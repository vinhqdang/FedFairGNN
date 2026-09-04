"""Dirichlet Heterogeneity & Client Scalability Sweep (Issue I6, Stanford Q7).

Evaluates TrustFedGNN vs FedAvg across varying non-IID graph partition skews:
    alpha in {0.1, 0.3, 0.5, 1.0}
and client counts:
    K in {5, 10, 20}
on Bail Recidivism (18.8k nodes).

Measures:
  - AUC-ROC
  - DPD_hard
  - EOD
  - Weight oscillation Omega_w
  - Edge retention sum_k |E_k| / |E| (how much of the graph survives the split)

Outputs:
  - results/revision/dirichlet_sweep.json
  - manuscript/tables/revision/dirichlet_sweep.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.metrics import weight_oscillation
from experiments.fairshare_common import partition_edge_retention


ALPHAS = [0.1, 0.3, 0.5, 1.0]
CLIENT_COUNTS = [5, 10, 20]
MODELS = ["trustfedgnn", "fedavg"]
SEEDS = [42, 43]


def evaluate_dirichlet_run(model_name: str, alpha: float, num_clients: int, seed: int,
                           dataset: str = "bail", rounds: int = 15) -> dict:
    t0 = time.perf_counter()
    is_ours = (model_name == "trustfedgnn")
    
    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=alpha,
        device="cpu",
        model="trustfedgnn" if is_ours else "gcn",
        aggregator="bfwa" if is_ours else "fedavg",
        local_fairness=is_ours,
        dp_enabled=is_ours,
        dp_epsilon=8.0 if is_ours else 8.0,
        dp_delta=1e-5,
    )

    trainer = FederatedTrainer(cfg)
    # Edge budget of this split, measured before training: the induced client
    # subgraphs drop every cross-client edge, so alpha and K change not just the
    # label mix but how much of the graph the federation can still see. Without
    # this column a drop in AUC at K=20 cannot be told apart from "the model
    # struggles under skew" versus "there is almost no graph left".
    part = partition_edge_retention(trainer)
    res = trainer.run(verbose=False)
    wall_clock = time.perf_counter() - t0

    # Weight oscillation
    weights_hist = []
    for r in res.get("history", []):
        w = r.get("agg_weights")
        if w is not None and len(w) == num_clients:
            weights_hist.append(w)
    omega = float(weight_oscillation(weights_hist))  # NaN when unmeasurable

    final = res["final"]
    return {
        "model": model_name,
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "omega_w": omega,
        "edge_retention": float(part["edge_retention"]),
        "edge_retention_post_holdout": float(part["edge_retention_post_holdout"]),
        "expected_retention_iid": float(part["expected_retention_iid"]),
        "original_edges": int(part["original_edges"]),
        "client_edges": int(part["client_edges"]),
        "wall_clock_s": float(wall_clock),
    }



def _edge_retention_table(records: List[dict]) -> List[str]:
    """Companion table: how much of the graph survives each (K, alpha) split.

    Reported alongside the utility/fairness sweep because the two are
    confounded -- an induced-subgraph partition discards every edge that
    crosses a client boundary, so K and alpha move the edge budget as well as
    the label skew. ``sum_k p_k^2`` is what a topology-blind split of the same
    node shares would retain in expectation; a measured retention close to it
    means the partition ignores the graph's structure.
    """
    lines = [
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Edge Budget of the Dirichlet Partitions.}",
        "Fraction of the global graph's edges retained inside the induced client subgraphs, "
        "$\\sum_k |E_k| / |E|$, for each client count $K$ and skew $\\alpha$. "
        "$\\sum_k p_k^2$ is the retention a topology-blind split of the same node shares "
        "would achieve; measured values close to it indicate the partition cuts the graph "
        "essentially at random, so cross-client structure is unavailable to every method "
        "in the comparison alike.}",
        "\\label{tab:dirichlet_edge_retention}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Setting} & \\textbf{Retained edges} & \\textbf{Retention (\\%)} & "
        "\\textbf{$\\sum_k p_k^2$ (\\%)} \\\\",
        "\\midrule",
    ]
    for k in CLIENT_COUNTS:
        lines.append("\\multicolumn{4}{l}{\\textit{Client Population $K = " + str(k) + "$}} \\\\")
        for a in ALPHAS:
            m = [x for x in records
                 if x["num_clients"] == k and abs(x["alpha"] - a) < 1e-4
                 and "edge_retention" in x]
            if not m:
                lines.append(f"$\\alpha = {a}$ & -- & -- & -- \\\\")
                continue
            kept = np.mean([x["client_edges"] for x in m])
            tot = np.mean([x["original_edges"] for x in m])
            ret = 100 * np.mean([x["edge_retention"] for x in m])
            iid = 100 * np.mean([x["expected_retention_iid"] for x in m])
            lines.append(
                f"$\\alpha = {a}$ & {kept:,.0f} / {tot:,.0f} & {ret:.1f}\\% & {iid:.1f}\\% \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return lines


def run_dirichlet_experiment(out_json="results/revision/dirichlet_sweep.json",
                             out_tex="manuscript/tables/revision/dirichlet_sweep.tex"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    records = []
    total = len(ALPHAS) * len(CLIENT_COUNTS) * len(MODELS) * len(SEEDS)
    idx = 0

    print(f"[*] Running Dirichlet & Client Count Sweep ({total} total runs)...", flush=True)

    for k in CLIENT_COUNTS:
        for a in ALPHAS:
            for m in MODELS:
                for s in SEEDS:
                    idx += 1
                    print(f"[{idx}/{total}] RUNNING: K={k} | alpha={a} | model={m} | seed={s}...", flush=True)
                    out = evaluate_dirichlet_run(m, a, k, s, rounds=15)
                    records.append(out)
                    print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, EOD={out['eod']:.4f}, "
                          f"Omega={out['omega_w']:.4f}, edge_ret={out['edge_retention'] * 100:.1f}% "
                          f"(iid expect {out['expected_retention_iid'] * 100:.1f}%) ({out['wall_clock_s']:.1f}s)", flush=True)

                    with open(out_json, "w") as f:
                        json.dump(records, f, indent=2)

    print(f"[+] Saved Dirichlet sweep JSON to {out_json}")

    # Generate LaTeX table
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Sensitivity to Data Heterogeneity (Dirichlet $\\alpha$) and Client Scaling ($K$) on Bail Recidivism.}",
        "Comparison of TrustFedGNN vs FedAvg across non-IID skew $\\alpha \\in \\{0.1, 0.3, 0.5, 1.0\\}$ and client counts $K \\in \\{5, 10, 20\\}$.",
        "Results reported as $\\text{Mean} \\pm \\text{Std}$ over random seeds. Lower $\\alpha$ indicates more severe label/sensitive distribution skew.}",
        "\\label{tab:dirichlet_sweep}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        " & \\multicolumn{4}{c}{\\textbf{TrustFedGNN (Ours)}} & \\multicolumn{4}{c}{\\textbf{FedAvg (Baseline)}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "\\textbf{Setting} & \\textbf{AUC $\\uparrow$} & \\textbf{DPD $\\downarrow$} & \\textbf{EOD $\\downarrow$} & \\textbf{$\\Omega_w$ $\\downarrow$} & \\textbf{AUC $\\uparrow$} & \\textbf{DPD $\\downarrow$} & \\textbf{EOD $\\downarrow$} & \\textbf{$\\Omega_w$ $\\downarrow$} \\\\",
        "\\midrule",
    ]

    for k in CLIENT_COUNTS:
        lines.append("\\multicolumn{9}{l}{\\textit{Client Population $K = " + str(k) + "$}} \\\\")
        for a in ALPHAS:
            matched_ours = [x for x in records if x["num_clients"] == k and abs(x["alpha"] - a) < 1e-4 and x["model"] == "trustfedgnn"]
            matched_base = [x for x in records if x["num_clients"] == k and abs(x["alpha"] - a) < 1e-4 and x["model"] == "fedavg"]

            def fmt_stats(m_list):
                if not m_list:
                    return "-- & -- & -- & --"
                auc_m, auc_s = np.mean([x["auc"] for x in m_list]), np.std([x["auc"] for x in m_list])
                dpd_m, dpd_s = np.mean([x["dpd_hard"] for x in m_list]), np.std([x["dpd_hard"] for x in m_list])
                eod_m, eod_s = np.mean([x["eod"] for x in m_list]), np.std([x["eod"] for x in m_list])
                om_m = np.mean([x["omega_w"] for x in m_list])
                return f"{auc_m:.3f}$\\pm${auc_s:.3f} & {dpd_m:.3f}$\\pm${dpd_s:.3f} & {eod_m:.3f}$\\pm${eod_s:.3f} & {om_m:.3f}"

            str_ours = fmt_stats(matched_ours)
            str_base = fmt_stats(matched_base)
            lines.append(f"$\\alpha = {a}$ & {str_ours} & {str_base} \\\\")
        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    lines += _edge_retention_table(records)

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX Dirichlet sweep table to {out_tex}")


if __name__ == "__main__":
    run_dirichlet_experiment()
