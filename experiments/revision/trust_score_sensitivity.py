"""Sensitivity Analysis of the Composite Trust Score (Issue I11, GPT 7, GLM M5).

Perturbs composite trust score weights by +/-25% (and Dirichlet sampling over 2,000 configurations)
to assess ranking stability across federated baselines.

Measures:
  - Spearman rank correlation rho_s (mean +/- std, 95% CI)
  - Kendall's tau (mean +/- std, 95% CI)
  - Rank-1 retention probability for TrustFedGNN

Outputs:
  - results/revision/trust_score_sensitivity.json
  - manuscript/tables/revision/trust_score_sensitivity.tex
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.trust.trust_score import TrustWeights, sub_scores, trust_score


# Standard benchmark methods with representative metrics from main results
BASELINES = {
    "FedAvg-GCN": {"auc": 0.7272, "dpd": 0.0400, "eod": 0.0588, "epsilon": None, "clean_auc": 0.7272, "ece": 0.082},
    "FairGNN": {"auc": 0.5839, "dpd": 0.0200, "eod": 0.0075, "epsilon": None, "clean_auc": 0.5839, "ece": 0.145},
    "FairSIN": {"auc": 0.7212, "dpd": 0.0341, "eod": 0.0370, "epsilon": None, "clean_auc": 0.7212, "ece": 0.078},
    "FairFed": {"auc": 0.7232, "dpd": 0.0065, "eod": 0.0168, "epsilon": None, "clean_auc": 0.7232, "ece": 0.065},
    "FairGFL": {"auc": 0.7318, "dpd": 0.0399, "eod": 0.0618, "epsilon": None, "clean_auc": 0.7318, "ece": 0.071},
    "FedGraph-Fair": {"auc": 0.7159, "dpd": 0.0367, "eod": 0.0567, "epsilon": None, "clean_auc": 0.7159, "ece": 0.085},
    "CGSV": {"auc": 0.7390, "dpd": 0.0435, "eod": 0.0540, "epsilon": None, "clean_auc": 0.7390, "ece": 0.069},
    "DP-FedAvg": {"auc": 0.5230, "dpd": 0.0100, "eod": 0.0080, "epsilon": 8.0, "clean_auc": 0.5230, "ece": 0.215},
    "Krum": {"auc": 0.7100, "dpd": 0.0320, "eod": 0.0220, "epsilon": None, "clean_auc": 0.7100, "ece": 0.090},
    "Multi-Krum": {"auc": 0.7180, "dpd": 0.0450, "eod": 0.0300, "epsilon": None, "clean_auc": 0.7180, "ece": 0.084},
    "Geometric Median": {"auc": 0.7050, "dpd": 0.0380, "eod": 0.0200, "epsilon": None, "clean_auc": 0.7050, "ece": 0.088},
    "Ours w/o FSER": {"auc": 0.7662, "dpd": 0.0161, "eod": 0.0246, "epsilon": 8.0, "clean_auc": 0.7662, "ece": 0.058},
    "TrustFedGNN (Ours)": {"auc": 0.7862, "dpd": 0.0149, "eod": 0.0219, "epsilon": 8.0, "clean_auc": 0.7862, "ece": 0.043},
}


def run_trust_sensitivity_analysis(num_samples=2000, perturbation_pct=0.25,
                                   out_json="results/revision/trust_score_sensitivity.json",
                                   out_tex="manuscript/tables/revision/trust_score_sensitivity.tex"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    rng = np.random.default_rng(42)
    methods = list(BASELINES.keys())

    # Baseline scores with uniform weights
    w_base = TrustWeights(utility=1.0, fairness=1.0, privacy=1.0, robustness=1.0, calibration=1.0)
    base_scores = np.array([trust_score(BASELINES[m], weights=w_base, p=1.0) for m in methods])
    base_ranks = np.argsort(-base_scores)

    # 1. Uniform perturbation +/-25%
    spearman_pert = []
    kendall_pert = []
    ours_rank_pert = []

    for _ in range(num_samples):
        # Sample weights in [1 - delta, 1 + delta]
        pert = rng.uniform(1.0 - perturbation_pct, 1.0 + perturbation_pct, size=5)
        w = TrustWeights(utility=pert[0], fairness=pert[1], privacy=pert[2], robustness=pert[3], calibration=pert[4])
        scores = np.array([trust_score(BASELINES[m], weights=w, p=1.0) for m in methods])
        
        rho, _ = spearmanr(base_scores, scores)
        tau, _ = kendalltau(base_scores, scores)
        spearman_pert.append(rho)
        kendall_pert.append(tau)

        # Rank of TrustFedGNN (0-indexed rank 0 = Rank 1)
        rank_idx = int(np.where(np.argsort(-scores) == methods.index("TrustFedGNN (Ours)"))[0][0])
        ours_rank_pert.append(rank_idx + 1)

    # 2. Extreme Dirichlet sampling (Dirichlet(1, 1, 1, 1, 1))
    spearman_diri = []
    kendall_diri = []
    ours_rank_diri = []

    for _ in range(num_samples):
        d_weights = rng.dirichlet(np.ones(5)) * 5.0
        w = TrustWeights(utility=d_weights[0], fairness=d_weights[1], privacy=d_weights[2], robustness=d_weights[3], calibration=d_weights[4])
        scores = np.array([trust_score(BASELINES[m], weights=w, p=1.0) for m in methods])

        rho, _ = spearmanr(base_scores, scores)
        tau, _ = kendalltau(base_scores, scores)
        spearman_diri.append(rho)
        kendall_diri.append(tau)

        rank_idx = int(np.where(np.argsort(-scores) == methods.index("TrustFedGNN (Ours)"))[0][0])
        ours_rank_diri.append(rank_idx + 1)

    results = {
        "num_samples": num_samples,
        "perturbation_range": f"+/-{int(perturbation_pct * 100)}%",
        "perturbation_regime": {
            "spearman_mean": float(np.mean(spearman_pert)),
            "spearman_std": float(np.std(spearman_pert)),
            "spearman_ci95": [float(np.percentile(spearman_pert, 2.5)), float(np.percentile(spearman_pert, 97.5))],
            "kendall_mean": float(np.mean(kendall_pert)),
            "kendall_std": float(np.std(kendall_pert)),
            "kendall_ci95": [float(np.percentile(kendall_pert, 2.5)), float(np.percentile(kendall_pert, 97.5))],
            "ours_rank1_pct": float(np.mean(np.array(ours_rank_pert) == 1)) * 100.0,
        },
        "dirichlet_regime": {
            "spearman_mean": float(np.mean(spearman_diri)),
            "spearman_std": float(np.std(spearman_diri)),
            "spearman_ci95": [float(np.percentile(spearman_diri, 2.5)), float(np.percentile(spearman_diri, 97.5))],
            "kendall_mean": float(np.mean(kendall_diri)),
            "kendall_std": float(np.std(kendall_diri)),
            "kendall_ci95": [float(np.percentile(kendall_diri, 2.5)), float(np.percentile(kendall_diri, 97.5))],
            "ours_rank1_pct": float(np.mean(np.array(ours_rank_diri) == 1)) * 100.0,
        }
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved trust score sensitivity JSON to {out_json}")

    # Generate LaTeX table
    p_reg = results["perturbation_regime"]
    d_reg = results["dirichlet_regime"]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Composite Trust Score Sensitivity Analysis under Weight Perturbations (Issue I11).}",
        "Robustness of relative baseline rankings under $2{,}000$ Monte Carlo weight configurations perturbing the five trust sub-score weights (Utility, Fairness, Privacy, Robustness, Calibration).",
        "Reports Spearman's $\\rho_s$, Kendall's $\\tau$, and Rank-1 retention probability for TrustFedGNN across uniform $\\pm 25\\%$ perturbations and unconstrained Dirichlet(1) priors.}",
        "\\label{tab:trust_score_sensitivity}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Perturbation Model} & \\textbf{Spearman $\\rho_s$ (95\\% CI)} & \\textbf{Kendall $\\tau$ (95\\% CI)} & \\textbf{Rank-1 Retention (\\%)} \\\\",
        "\\midrule",
        f"Uniform $\\pm 25\\%$ Perturbation & {p_reg['spearman_mean']:.3f} [{p_reg['spearman_ci95'][0]:.3f}, {p_reg['spearman_ci95'][1]:.3f}] & {p_reg['kendall_mean']:.3f} [{p_reg['kendall_ci95'][0]:.3f}, {p_reg['kendall_ci95'][1]:.3f}] & \\textbf{{{p_reg['ours_rank1_pct']:.1f}\\%}} \\\\",
        f"Unconstrained $\\text{{Dirichlet}}(\\mathbf{{1}})$ Prior & {d_reg['spearman_mean']:.3f} [{d_reg['spearman_ci95'][0]:.3f}, {d_reg['spearman_ci95'][1]:.3f}] & {d_reg['kendall_mean']:.3f} [{d_reg['kendall_ci95'][0]:.3f}, {d_reg['kendall_ci95'][1]:.3f}] & \\textbf{{{d_reg['ours_rank1_pct']:.1f}\\%}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX trust sensitivity table to {out_tex}")


if __name__ == "__main__":
    run_trust_sensitivity_analysis()
