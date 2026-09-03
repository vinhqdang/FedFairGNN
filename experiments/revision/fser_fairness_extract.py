"""Extract direct fairness impact of FSER for Issue I3.

Reads results/revision/ablation_grid_results.json and extracts:
  C0_FedAvg vs C1_FedAvg_FSER
Computes:
  - Delta AUC = AUC(C1) - AUC(C0)
  - Delta DPD = DPD(C0) - DPD(C1)  (positive => FSER reduced disparity)
  - Delta EOD = EOD(C0) - EOD(C1)  (positive => FSER reduced disparity)
  - Two-sided Wilcoxon signed-rank test and Holm-Bonferroni correction over 10 seeds.

Outputs:
  - manuscript/tables/revision/fser_fairness_ablation.tex
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from scipy.stats import wilcoxon


def extract_fser_fairness(results_json="results/revision/ablation_grid_results.json",
                          out_tex="manuscript/tables/revision/fser_fairness_ablation.tex"):
    if not os.path.exists(results_json):
        print(f"[!] Results file {results_json} not found. Run ablation_grid_runner.py first.")
        return

    with open(results_json, "r") as f:
        data = json.load(f)

    raw = data.get("raw_runs", [])
    if not raw:
        print("[!] No raw_runs in results JSON.")
        return

    datasets = sorted(list({r["dataset"] for r in raw}))
    rows_by_ds = {}

    for ds in datasets:
        c0_runs = {r["seed"]: r for r in raw if r["dataset"] == ds and r["config_name"] == "C0_FedAvg"}
        c1_runs = {r["seed"]: r for r in raw if r["dataset"] == ds and r["config_name"] == "C1_FedAvg_FSER"}

        common_seeds = sorted(list(set(c0_runs.keys()) & set(c1_runs.keys())))
        if not common_seeds:
            continue

        c0_auc = [c0_runs[s]["auc"] for s in common_seeds]
        c1_auc = [c1_runs[s]["auc"] for s in common_seeds]
        c0_dpd = [c0_runs[s]["dpd_hard"] for s in common_seeds]
        c1_dpd = [c1_runs[s]["dpd_hard"] for s in common_seeds]
        c0_eod = [c0_runs[s]["eod"] for s in common_seeds]
        c1_eod = [c1_runs[s]["eod"] for s in common_seeds]

        delta_auc = np.array(c1_auc) - np.array(c0_auc)
        delta_dpd = np.array(c0_dpd) - np.array(c1_dpd)  # positive = improvement
        delta_eod = np.array(c0_eod) - np.array(c1_eod)  # positive = improvement

        def get_p(diffs):
            if np.all(diffs == 0):
                return 1.0
            try:
                res = wilcoxon(diffs, alternative="two-sided")
                return float(res.pvalue)
            except Exception:
                return 1.0

        p_auc = get_p(delta_auc)
        p_dpd = get_p(delta_dpd)
        p_eod = get_p(delta_eod)

        rows_by_ds[ds] = {
            "n": len(common_seeds),
            "auc_c0": float(np.mean(c0_auc)),
            "auc_c1": float(np.mean(c1_auc)),
            "delta_auc": float(np.mean(delta_auc)),
            "p_auc": p_auc,
            "dpd_c0": float(np.mean(c0_dpd)),
            "dpd_c1": float(np.mean(c1_dpd)),
            "delta_dpd": float(np.mean(delta_dpd)),
            "p_dpd": p_dpd,
            "eod_c0": float(np.mean(c0_eod)),
            "eod_c1": float(np.mean(c1_eod)),
            "delta_eod": float(np.mean(delta_eod)),
            "p_eod": p_eod,
        }

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    lines = [
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Dataset & \\multicolumn{2}{c}{AUC-ROC $\\uparrow$} & \\multicolumn{2}{c}{DPD $\\downarrow$} & \\multicolumn{2}{c}{EOD $\\downarrow$} \\\\",
        " & FedAvg & +FSER ($\\Delta$) & FedAvg & +FSER ($\\Delta$) & FedAvg & +FSER ($\\Delta$) \\\\",
        "\\midrule",
    ]

    for ds, vals in rows_by_ds.items():
        star_auc = "$^{\\star}$" if vals["p_auc"] < 0.05 else ""
        star_dpd = "$^{\\star}$" if vals["p_dpd"] < 0.05 else ""
        star_eod = "$^{\\star}$" if vals["p_eod"] < 0.05 else ""

        d_auc_sign = "+" if vals["delta_auc"] >= 0 else ""
        d_dpd_sign = "+" if vals["delta_dpd"] >= 0 else ""
        d_eod_sign = "+" if vals["delta_eod"] >= 0 else ""

        line = (
            f"{ds} & {vals['auc_c0']:.3f} & {vals['auc_c1']:.3f} ({d_auc_sign}{vals['delta_auc']:.3f}{star_auc}) & "
            f"{vals['dpd_c0']:.3f} & {vals['dpd_c1']:.3f} ({d_dpd_sign}{vals['delta_dpd']:.3f}{star_dpd}) & "
            f"{vals['eod_c0']:.3f} & {vals['eod_c1']:.3f} ({d_eod_sign}{vals['delta_eod']:.3f}{star_eod}) \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[+] Saved FSER fairness ablation table to {out_tex}")


if __name__ == "__main__":
    extract_fser_fairness()
