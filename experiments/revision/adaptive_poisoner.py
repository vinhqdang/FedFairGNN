"""Adaptive Stealth Fairness-Poisoner Evaluation (Issue I4 & I5, Stanford Q4-Q6).

Formulates an omniscient adaptive attacker that explicitly evades distance-to-median
screening by projecting its update within the safe radius of benign updates:
    g_stealth = median(G) + rho * (g_poison - median(G)) / ||...|| * r_safe
while falsely reporting DPD_hat = 0.0 and Perf = 0.99.

Evaluates breakdown curves of 6 aggregators (robust_bfwa, krum, median, trimmed_mean, bfwa, fedavg)
across Byzantine corruption fractions f/K in {0.1, 0.2, 0.3, 0.4} on Bail (18.8k nodes).

Outputs:
  - results/revision/adaptive_poisoner_results.json
  - manuscript/tables/revision/adaptive_poisoner.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.aggregation import aggregate


def craft_stealth_poison_updates(updates: List[torch.Tensor], metas: List[dict],
                                 byzantine_ids: List[int], rho: float = 0.85) -> Tuple[List[torch.Tensor], List[dict]]:
    """Stealth projection: ensures adversarial updates stay within the benign median distance radius."""
    if not byzantine_ids:
        return updates, metas

    updates = [u.clone() for u in updates]
    metas = [dict(m) for m in metas]
    benign = [i for i in range(len(updates)) if i not in byzantine_ids]

    if not benign:
        return updates, metas

    bstack = torch.stack([updates[i].flatten() for i in benign])
    bmed = bstack.median(0).values
    bdist = ((bstack - bmed) ** 2).sum(1).sqrt()
    r_safe = float(bdist.median()) * rho  # Strictly within median benign distance

    for i in byzantine_ids:
        g = updates[i].flatten()
        direction = g - bmed
        dir_norm = direction.norm() + 1e-8
        # Project within safe radius
        g_stealth = bmed + (direction / dir_norm) * min(float(dir_norm), r_safe)
        updates[i] = g_stealth.view_as(updates[i])
        # Lie about fairness/utility to capture BFWA
        metas[i]["dpd"] = 0.0
        metas[i]["eod"] = 0.0
        metas[i]["perf"] = 0.99

    return updates, metas


def evaluate_adaptive_run(aggregator: str, byz_ratio: float, seed: int = 42, rounds: int = 15,
                          device: str = "cpu", dataset: str = "bail",
                          num_clients: int = 10) -> dict:
    t0 = time.perf_counter()
    num_byz = int(round(byz_ratio * num_clients))
    byz_indices = set(range(num_byz))

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator=aggregator,
        attack="fairness_poison",
        num_byzantine=num_byz,
        krum_f=num_byz,
        attack_intensity=10.0,
        dp_enabled=False,
    )

    trainer = FederatedTrainer(cfg)

    # Monkey patch round aggregation with stealth crafting
    orig_round = trainer._round

    def stealth_round(t):
        updates, metas = [], []
        for c in trainer.clients:
            c.set_flat(trainer.global_flat)
            c.train()
            g_k = trainer.global_flat - c.get_flat()
            updates.append(g_k)
            metas.append(c.meta())

        # Craft stealth updates
        updates, metas = craft_stealth_poison_updates(updates, metas, list(byz_indices), rho=0.85)

        # Aggregate. `state=trainer._agg_state` threads stateful-aggregator state
        # (BFWA's dual multiplier, fedgraphfair's lambda, ...) across rounds the
        # same way FederatedTrainer._round does -- omitting it silently restarts
        # BFWA's dual every round, which is exactly the bug that made tau inert
        # everywhere else in the repo (see aggregation.bfwa_weights).
        g_agg, info = aggregate(
            cfg.aggregator, updates, metas,
            tau=cfg.fairness_budget,
            fw_iters=cfg.fw_iterations,
            dual_step=cfg.dual_step_size,
            krum_f=cfg.krum_f,
            state=trainer._agg_state,
            bfwa_persist_dual=cfg.bfwa_persist_dual,
        )

        trainer.global_flat = trainer.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in trainer.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        if "kept" in info:
            rec["kept"] = info["kept"]
        return rec

    trainer._round = stealth_round
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    # Calculate w_adv -- the adversary's captured aggregation weight mass.
    #
    # NaN, NOT 0.0, when the rule exposes no weight vector. Coordinate median and
    # trimmed_mean are not expressible as client weights, so aggregate() returns
    # no info["weights"] for them and adv_weights stays empty. The old 0.0 default
    # made that empty list indistinguishable from a measured "the attacker
    # captured nothing", which is how median came to be reported as w_adv = 0.000
    # and described as "completely immune" -- a dict-lookup fallback, not a
    # measurement. Any aggregator returning no weights would have printed 0.000
    # under any attack, including none at all.
    hist = res.get("history", [])
    adv_weights = []
    for r_entry in hist:
        w_list = r_entry.get("agg_weights")
        if w_list is not None and len(w_list) == num_clients:
            byz_w = sum(w_list[i] for i in byz_indices)
            adv_weights.append(byz_w)

    mean_w_adv = float(np.mean(adv_weights)) if adv_weights else float("nan")
    final = res["final"]

    return {
        "aggregator": aggregator,
        "byz_ratio": byz_ratio,
        "num_byz": num_byz,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "w_adv": mean_w_adv,
        "wall_clock_s": wall_clock_s,
    }



# --------------------------------------------------------------------------- #
# Caption text, derived from the measurement
# --------------------------------------------------------------------------- #
# The final caption sentence used to be a fixed string asserting that "Krum and
# Median retain robust utility up to their theoretical breakdown limits ... while
# BFWA-based methods suffer fairness degradation" -- an interpretation written
# before the runs and printed regardless of what they produced. It is now
# derived from the records. Cut-offs, stated so they can be argued with:
#
#   |Delta AUC| from the lowest to the highest corruption ratio
#       < 0.02  -> "holds"      (inside seed noise for these datasets)
#       < 0.10  -> "degrades"
#      >= 0.10  -> "breaks down"
#   the same three bands, on DPD, decide whether fairness held.
AUC_DROP_BANDS = ((0.02, "holds"), (0.10, "degrades"), (float("inf"), "breaks down"))
DPD_RISE_BANDS = ((0.02, "holds"), (0.10, "degrades"), (float("inf"), "breaks down"))


def _band(value: float, bands) -> str:
    for hi, name in bands:
        if value < hi:
            return name
    return bands[-1][1]                                   # pragma: no cover


def summarise_breakdown(records: List[dict], aggregators: List[str],
                        byz_ratios: List[float]) -> Dict[str, dict]:
    """Per-aggregator change from the lowest to the highest corruption ratio."""
    lo, hi = min(byz_ratios), max(byz_ratios)
    out = {}
    for agg in aggregators:
        def at(r):
            m = [x for x in records
                 if x["aggregator"] == agg and abs(x["byz_ratio"] - r) < 1e-4]
            if not m:
                return None
            w = [x["w_adv"] for x in m if not math.isnan(x["w_adv"])]
            return (float(np.mean([x["auc"] for x in m])),
                    float(np.mean([x["dpd_hard"] for x in m])),
                    float(np.mean(w)) if w else float("nan"))
        a, b = at(lo), at(hi)
        if a is None or b is None:
            continue
        d_auc = a[0] - b[0]          # positive = utility lost as f grows
        d_dpd = b[1] - a[1]          # positive = fairness worsened as f grows
        out[agg] = {
            "auc_low": a[0], "auc_high": b[0], "auc_drop": d_auc,
            "dpd_low": a[1], "dpd_high": b[1], "dpd_rise": d_dpd,
            "w_adv_high": b[2],
            "utility_verdict": _band(abs(d_auc), AUC_DROP_BANDS),
            "fairness_verdict": _band(abs(d_dpd), DPD_RISE_BANDS),
        }
    return out


def _breakdown_caption_sentence(summary: Dict[str, dict], byz_ratios: List[float]) -> str:
    """Interpretive sentence, assembled from the measured breakdown summary."""
    if not summary:
        return "No aggregator produced a complete corruption sweep in this run."
    lo, hi = min(byz_ratios), max(byz_ratios)
    held = [a for a, v in summary.items() if v["utility_verdict"] == "holds"]
    lost = [a for a, v in summary.items() if v["utility_verdict"] != "holds"]
    # direction matters: dpd_rise > 0 means disparity got WORSE as f grew.
    unfair = [a for a, v in summary.items()
              if v["fairness_verdict"] != "holds" and v["dpd_rise"] > 0]
    fairer = [a for a, v in summary.items()
              if v["fairness_verdict"] != "holds" and v["dpd_rise"] < 0]

    def names(xs):
        xs = [x.replace("_", "\\_") for x in xs]
        if len(xs) == 1:
            return xs[0]
        return ", ".join(xs[:-1]) + " and " + xs[-1]

    parts = []
    if held:
        parts.append(f"utility holds (within $0.02$ AUC) for {names(held)}")
    if lost:
        worst = max(lost, key=lambda a: abs(summary[a]["auc_drop"]))
        parts.append(
            f"utility {summary[worst]['utility_verdict']} for {names(lost)} "
            f"(worst: {worst.replace('_', chr(92) + chr(92) + '_')}, "
            f"${summary[worst]['auc_drop']:+.3f}$ AUC)")
    if unfair:
        worst_f = max(unfair, key=lambda a: summary[a]["dpd_rise"])
        parts.append(
            f"disparity worsens for {names(unfair)} "
            f"(worst: {worst_f.replace('_', chr(92) + chr(92) + '_')}, "
            f"DPD ${summary[worst_f]['dpd_rise']:+.3f}$)")
    else:
        parts.append("no aggregator shows a disparity increase above $0.02$")
    if fairer:
        best_f = min(fairer, key=lambda a: summary[a]["dpd_rise"])
        parts.append(
            f"disparity in fact falls for {names(fairer)} "
            f"(largest: {best_f.replace('_', chr(92) + chr(92) + '_')}, "
            f"DPD ${summary[best_f]['dpd_rise']:+.3f}$), which at a higher "
            "corruption ratio reflects the attack flattening predictions rather "
            "than the defence improving")

    return (f"Measured from $f/K = {lo}$ to $f/K = {hi}$: " + "; ".join(parts) + ". "
            "Verdicts use fixed thresholds on the measured change "
            "($<0.02$ holds, $<0.10$ degrades, otherwise breaks down); "
            "they are read off this run, not assumed.")


def run_adaptive_experiment(out_json="results/revision/adaptive_poisoner_results.json",
                            out_tex="manuscript/tables/revision/adaptive_poisoner.tex",
                            aggregators=("fedavg", "bfwa", "krum", "median", "robust_bfwa"),
                            byz_ratios=(0.1, 0.2, 0.3, 0.4), seeds=(42,), rounds=15,
                            dataset="bail"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    aggregators = list(aggregators)
    byz_ratios = list(byz_ratios)
    seeds = list(seeds)

    records = []
    total = len(aggregators) * len(byz_ratios) * len(seeds)
    idx = 0

    print(f"[*] Running Adaptive Stealth Poisoner suite ({total} total runs)...", flush=True)

    for ratio in byz_ratios:
        for agg in aggregators:
            for s in seeds:
                idx += 1
                print(f"[{idx}/{total}] RUNNING: agg={agg} | ratio={ratio} | seed={s}...", flush=True)
                out = evaluate_adaptive_run(agg, ratio, seed=s, rounds=rounds,
                                            dataset=dataset)
                records.append(out)
                print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, w_adv={out['w_adv']:.3f} ({out['wall_clock_s']:.1f}s)", flush=True)
                with open(out_json, "w") as f:
                    json.dump(records, f, indent=2)

    summary = summarise_breakdown(records, aggregators, byz_ratios)
    with open(out_json.replace(".json", "_breakdown_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved adaptive poisoner JSON to {out_json}")

    # Generate summary LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Adaptive Stealth Poisoning Breakdown Point Analysis "
        f"({dataset.capitalize()}).}}",
        "Performance under an omniscient adversary that projects malicious updates within the benign median radius ($\\rho=0.85$) while falsifying $\\widehat{\\text{DPD}} = 0.0$.",
        "Reports AUC / DPD across corruption ratios $f/K \\in \\{"
        + ", ".join(f"{r:g}" for r in byz_ratios) + "\\}$"
        + f" over {len(seeds)} seed" + ("s" if len(seeds) != 1 else "") + ".",
        _breakdown_caption_sentence(
            summarise_breakdown(records, aggregators, byz_ratios), byz_ratios) + "}",
        "\\label{tab:adaptive_poisoner}",
        "\\begin{tabular}{l" + "c" * len(byz_ratios) + "}",
        "\\toprule",
        "\\textbf{Aggregator} & " + " & ".join(
            f"\\textbf{{$f={r:g}$ ({int(round(r * 10))}/10)}}" for r in byz_ratios) + " \\\\",
        " & " + " & ".join(["AUC / DPD"] * len(byz_ratios)) + " \\\\",
        "\\midrule",
    ]

    for agg in aggregators:
        row_cells = []
        for r in byz_ratios:
            matched = [x for x in records if x["aggregator"] == agg and abs(x["byz_ratio"] - r) < 1e-4]
            if matched:
                auc_m = np.mean([m["auc"] for m in matched])
                dpd_m = np.mean([m["dpd_hard"] for m in matched])
                cell = f"{auc_m:.3f} / {dpd_m:.3f}"
            else:
                cell = "--"
            row_cells.append(cell)
        agg_name = "\\textbf{robust\\_bfwa (ours)}" if agg == "robust_bfwa" else agg.replace("_", "\\_")
        lines.append(f"{agg_name} & " + " & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX adaptive poisoner table to {out_tex}")


def main():
    ap = argparse.ArgumentParser(description="Adaptive stealth fairness poisoner.")
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--aggregators", nargs="+",
                    default=["fedavg", "bfwa", "krum", "median", "robust_bfwa"])
    ap.add_argument("--byz-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out-json", default="results/revision/adaptive_poisoner_results.json")
    ap.add_argument("--out-tex", default="manuscript/tables/revision/adaptive_poisoner.tex")
    a = ap.parse_args()
    run_adaptive_experiment(out_json=a.out_json, out_tex=a.out_tex,
                            aggregators=a.aggregators, byz_ratios=a.byz_ratios,
                            seeds=a.seeds, rounds=a.rounds, dataset=a.dataset)


if __name__ == "__main__":
    main()
