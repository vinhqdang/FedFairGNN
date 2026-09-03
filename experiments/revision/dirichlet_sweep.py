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
    res = trainer.run(verbose=False)
    wall_clock = time.perf_counter() - t0

    # Weight oscillation
    weights_hist = []
    for r in res.get("history", []):
        w = r.get("agg_weights")
        if w is not None and len(w) == num_clients:
            weights_hist.append(w)
    omega = float(weight_oscillation(weights_hist)) if weights_hist else 0.0

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
        "wall_clock_s": float(wall_clock),
    }


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
                    print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, EOD={out['eod']:.4f}, Omega={out['omega_w']:.4f} ({out['wall_clock_s']:.1f}s)", flush=True)

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

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX Dirichlet sweep table to {out_tex}")


if __name__ == "__main__":
    run_dirichlet_experiment()
