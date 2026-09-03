"""Extended robustness evaluation for Issue I5: Multi-seed Byzantine & Fairness-Poisoning Evaluation.

Evaluates 7 aggregators across 3 attacks (gaussian, alie, fairness_poison) x 10 seeds {42..51}
on Bail (18.8k nodes) with Byzantine ratio sweep f/K in {0.1, 0.2, 0.3, 0.4}.

Aggregators:
  - fedavg
  - bfwa
  - krum
  - multikrum
  - median
  - trimmed_mean
  - robust_bfwa (ours)

Computes:
  - AUC (mean +/- std)
  - DPD_hard (mean +/- std)
  - EOD (mean +/- std)
  - Attacker weight share w_adv
  - False positive screening rate (fraction of benign clients dropped by screening)
  - Breakdown point identification across f/K

Outputs:
  - results/revision/robustness_multiseed.json
  - manuscript/tables/revision/robustness_v2.tex
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated import FederatedTrainer


AGGREGATORS = ["fedavg", "bfwa", "krum", "multikrum", "median", "trimmed_mean", "robust_bfwa"]
ATTACKS = ["gaussian", "alie", "fairness_poison"]
DEFAULT_SEEDS = list(range(42, 52))
DEFAULT_BYZ_RATIOS = [0.1, 0.2, 0.3, 0.4]


def get_git_provenance() -> Tuple[str, bool]:
    commit = os.environ.get("FEDFAIR_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    dirty_env = os.environ.get("FEDFAIR_GIT_DIRTY")
    if commit:
        dirty = (dirty_env in ("1", "true", "True"))
        return commit.strip(), dirty
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return c, bool(status)
    except Exception:
        return "unknown", False


def evaluate_single_byz_run(aggregator: str, attack: str, byz_ratio: float, seed: int, device: str, dp_enabled: bool = False) -> dict:
    t0 = time.perf_counter()
    num_clients = 10
    num_byz = int(round(byz_ratio * num_clients))
    byz_indices = set(range(num_byz))
    benign_indices = set(range(num_byz, num_clients))

    cfg = ExperimentConfig.canonical(
        dataset="bail",
        seed=seed,
        num_clients=num_clients,
        rounds=50,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator=aggregator,
        attack=attack,
        num_byzantine=num_byz,
        krum_f=num_byz,
        attack_intensity=10.0,
        dp_enabled=dp_enabled,
    )

    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    # Calculate attacker weight share and false positive screening rate
    hist = res.get("history", [])
    adv_weights = []
    fp_rates = []

    for r_entry in hist:
        w_list = r_entry.get("agg_weights")
        if w_list is not None and len(w_list) == num_clients:
            byz_w = sum(w_list[i] for i in byz_indices)
            adv_weights.append(byz_w)

        # Screening inspection (for robust_bfwa)
        if "kept" in r_entry:
            kept = set(r_entry["kept"])
            # False positive: benign client dropped
            benign_dropped = benign_indices - kept
            fp_rate = len(benign_dropped) / len(benign_indices) if benign_indices else 0.0
            fp_rates.append(fp_rate)

    mean_w_adv = float(np.mean(adv_weights)) if adv_weights else float("nan")  # NaN, not 0.0: an aggregator that exposes no weight vector
    # (coordinate median, trimmed_mean) never populates adv_weights, and a 0.0
    # there reads as a measured "attacker captured nothing". See
    # experiments/revision/adaptive_poisoner.py for the full note.
    mean_fp_rate = float(np.mean(fp_rates)) if fp_rates else 0.0

    final = res["final"]
    return {
        "aggregator": aggregator,
        "attack": attack,
        "byz_ratio": byz_ratio,
        "num_byzantine": num_byz,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_soft": float(final["dpd_soft"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "pred_std": float(final.get("pred_std", 0.0)),
        "w_adv": mean_w_adv,
        "false_positive_screening_rate": mean_fp_rate,
        "wall_clock_s": float(wall_clock_s),
    }


def generate_robustness_v2_table(results_store: dict, target_ratio: float = 0.2, out_tex: str = "manuscript/tables/revision/robustness_v2.tex"):
    """Generate publication-ready LaTeX table for robustness at target_ratio (default 2/10)."""
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    raw = [r for r in results_store["raw_runs"] if abs(r["byz_ratio"] - target_ratio) < 1e-4]
    if not raw:
        return

    COLLAPSE_AUC = 0.60
    stats = {}
    for agg in AGGREGATORS:
        for atk in ATTACKS:
            matched = [r for r in raw if r["aggregator"] == agg and r["attack"] == atk]
            if matched:
                stats[(agg, atk)] = {
                    "auc_m": float(np.mean([m["auc"] for m in matched])),
                    "auc_s": float(np.std([m["auc"] for m in matched])),
                    "dpd_m": float(np.mean([m["dpd_hard"] for m in matched])),
                    "dpd_s": float(np.std([m["dpd_hard"] for m in matched])),
                    "eod_m": float(np.mean([m["eod"] for m in matched])),
                    "eod_s": float(np.std([m["eod"] for m in matched])),
                    "n": len(matched),
                }

    pretty_atk = {"gaussian": "Gaussian ($\\sigma=10$)", "alie": "ALIE ($z=1.5$)", "fairness_poison": "Fair-poison"}
    header_cols = " & ".join(f"\\multicolumn{{3}}{{c}}{{{pretty_atk[a]}}}" for a in ATTACKS)
    sub_headers = " & ".join("AUC $\\uparrow$ & DPD $\\downarrow$ & EOD $\\downarrow$" for _ in ATTACKS)

    lines = [
        "\\begin{tabular}{l" + "ccc" * len(ATTACKS) + "}",
        "\\toprule",
        f" & {header_cols} \\\\",
        f"Aggregator & {sub_headers} \\\\",
        "\\midrule",
    ]

    # Find best among non-collapsed
    best = {}
    for atk in ATTACKS:
        present = [g for g in AGGREGATORS if (g, atk) in stats]
        if present:
            best[(atk, "auc")] = max(present, key=lambda g: stats[(g, atk)]["auc_m"])
            healthy = [g for g in present if stats[(g, atk)]["auc_m"] >= COLLAPSE_AUC]
            if healthy:
                best[(atk, "dpd")] = min(healthy, key=lambda g: stats[(g, atk)]["dpd_m"])
                best[(atk, "eod")] = min(healthy, key=lambda g: stats[(g, atk)]["eod_m"])

    for agg in AGGREGATORS:
        cells = []
        for atk in ATTACKS:
            s = stats.get((agg, atk))
            if not s:
                cells += ["--", "--", "--"]
                continue
            collapsed = s["auc_m"] < COLLAPSE_AUC
            
            # Format AUC
            auc_str = f"{s['auc_m']:.3f}$\\pm${s['auc_s']:.3f}"
            if best.get((atk, "auc")) == agg:
                auc_str = f"\\textbf{{{auc_str}}}"
            
            # Format DPD
            dpd_str = f"{s['dpd_m']:.3f}$\\pm${s['dpd_s']:.3f}"
            if collapsed:
                dpd_str += "$^{\\dagger}$"
            elif best.get((atk, "dpd")) == agg:
                dpd_str = f"\\textbf{{{dpd_str}}}"

            # Format EOD
            eod_str = f"{s['eod_m']:.3f}$\\pm${s['eod_s']:.3f}"
            if collapsed:
                eod_str += "$^{\\dagger}$"
            elif best.get((atk, "eod")) == agg:
                eod_str = f"\\textbf{{{eod_str}}}"

            cells += [auc_str, dpd_str, eod_str]

        name = "\\textbf{robust\\_bfwa (ours)}" if agg == "robust_bfwa" else agg.replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved updated LaTeX robustness table to {out_tex}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed Byzantine sweep on Bail.")
    parser.add_argument("--aggregators", type=str, default=",".join(AGGREGATORS))
    parser.add_argument("--attacks", type=str, default=",".join(ATTACKS))
    parser.add_argument("--byz-ratios", type=str, default="0.2",
                        help="Comma-separated ratios, e.g. 0.1,0.2,0.3,0.4 (default 0.2 for Table 9)")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--device", type=str, default=os.environ.get("FEDFAIR_DEVICE", "cpu"))
    parser.add_argument("--dp-enabled", action="store_true", default=False,
                        help="Enable DP during robustness evaluation (default False to isolate robustness from DP noise)")
    parser.add_argument("--output", type=str, default="results/revision/robustness_multiseed.json")
    parser.add_argument("--out-tex", type=str, default="manuscript/tables/revision/robustness_v2.tex")
    args = parser.parse_args()

    aggregators = [a.strip() for a in args.aggregators.split(",") if a.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    byz_ratios = [float(r.strip()) for r in args.byz_ratios.split(",") if r.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = args.device
    dp_enabled = args.dp_enabled
    out_file = args.output

    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)
    git_commit, git_dirty = get_git_provenance()

    results_store = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results_store = json.load(f)
        except Exception:
            results_store = {}

    if "_manifest" not in results_store:
        results_store["_manifest"] = {
            "experiment": "robustness_multiseed",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "device": device,
            "dp_enabled": dp_enabled,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    if "raw_runs" not in results_store:
        results_store["raw_runs"] = []

    completed_keys = set()
    for run in results_store["raw_runs"]:
        completed_keys.add((run["aggregator"], run["attack"], round(run["byz_ratio"], 4), run["seed"]))

    total_tasks = len(aggregators) * len(attacks) * len(byz_ratios) * len(seeds)
    done_count = len(completed_keys)
    print(f"[*] Total robustness runs: {total_tasks} | Already completed: {done_count}")

    run_idx = 0
    for ratio in byz_ratios:
        for atk in attacks:
            for agg in aggregators:
                for s in seeds:
                    run_idx += 1
                    key = (agg, atk, round(ratio, 4), s)
                    if key in completed_keys:
                        continue

                    print(f"[{run_idx}/{total_tasks}] RUNNING: agg={agg} | atk={atk} | byz_ratio={ratio} | seed={s} (dp={dp_enabled})...", flush=True)
                    out = evaluate_single_byz_run(agg, atk, ratio, s, device, dp_enabled=dp_enabled)
                    out["git_commit"] = git_commit
                    out["git_dirty"] = git_dirty
                    out["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    results_store["raw_runs"].append(out)
                    completed_keys.add(key)

                    print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, EOD={out['eod']:.4f}, w_adv={out['w_adv']:.3f} ({out['wall_clock_s']:.1f}s)", flush=True)

                    with open(out_file, "w") as f:
                        json.dump(results_store, f, indent=2)

    # Generate updated table
    generate_robustness_v2_table(results_store, target_ratio=0.2, out_tex=args.out_tex)
    print(f"\n[DONE] Completed robustness multi-seed runs. Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
