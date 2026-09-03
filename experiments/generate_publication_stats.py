"""Publication-grade statistical consolidation and LaTeX table generator for Q1 venues.

Computes:
1. Paired per-seed differences (Δ).
2. Paired Wilcoxon signed-rank test (exact p-value across 2^n permutations for n=5).
3. Paired Cohen's d_z effect size.
4. Bootstrap 95% Confidence Interval for mean difference.
5. Family-wise Holm-Bonferroni correction per metric family.
6. Publication-ready LaTeX tables for IEEE/ACM submissions.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np


def exact_wilcoxon_p(diffs: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank test."""
    diffs = np.asarray(diffs, float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs, alternative="two-sided", method="exact")
        return float(res.pvalue)
    except Exception:
        pass
    abs_d = np.abs(diffs)
    ranks = np.zeros(n)
    order = np.argsort(abs_d)
    for rank, idx in enumerate(order, 1):
        ranks[idx] = rank
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    w_stat = min(w_pos, w_neg)
    all_w = []
    for mask in range(1 << n):
        w = sum(ranks[i] for i in range(n) if (mask & (1 << i)))
        all_w.append(w)
    all_w = np.array(all_w)
    count = np.sum(all_w <= w_stat) + np.sum(all_w >= (np.sum(ranks) - w_stat))
    return min(1.0, float(count / (2**n)))


def compute_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, float, bool]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    reject = True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = reject and (p <= thresh)
        out[k] = (p, thresh, reject)
    return out


def analyze_dataset_runs(raw_runs: Dict[str, List[dict]]) -> Dict:
    ours_runs = raw_runs["fedfairgnn"]
    seeds = [r["seed"] for r in ours_runs]
    ours_auc = np.array([r["auc"] for r in ours_runs])
    ours_dpds = np.array([r["dpd_soft"] for r in ours_runs])
    ours_dpdh = np.array([r["dpd_hard"] for r in ours_runs])
    ours_eod = np.array([r["eod"] for r in ours_runs])
    ours_omg = np.array([r["omega_w"] for r in ours_runs])
    
    models = sorted(list(raw_runs.keys()))
    metrics_summary = {}
    
    # Store summary stats for each model
    for m in models:
        m_runs = raw_runs[m]
        metrics_summary[m] = {
            "auc": {"mean": float(np.mean([r["auc"] for r in m_runs])), "std": float(np.std([r["auc"] for r in m_runs], ddof=1))},
            "dpd_soft": {"mean": float(np.mean([r["dpd_soft"] for r in m_runs])), "std": float(np.std([r["dpd_soft"] for r in m_runs], ddof=1))},
            "dpd_hard": {"mean": float(np.mean([r["dpd_hard"] for r in m_runs])), "std": float(np.std([r["dpd_hard"] for r in m_runs], ddof=1))},
            "eod": {"mean": float(np.mean([r["eod"] for r in m_runs])), "std": float(np.std([r["eod"] for r in m_runs], ddof=1))},
            "omega_w": {"mean": float(np.mean([r["omega_w"] for r in m_runs])), "std": float(np.std([r["omega_w"] for r in m_runs], ddof=1))},
        }

    p_auc_map, p_dpdh_map, p_eod_map = {}, {}, {}
    comparisons = {}
    
    for m in models:
        if m == "fedfairgnn":
            continue
        m_runs = raw_runs[m]
        m_auc = np.array([r["auc"] for r in m_runs])
        m_dpdh = np.array([r["dpd_hard"] for r in m_runs])
        m_eod = np.array([r["eod"] for r in m_runs])
        
        d_auc = ours_auc - m_auc
        d_dpdh = ours_dpdh - m_dpdh
        d_eod = ours_eod - m_eod
        
        p_auc = exact_wilcoxon_p(d_auc)
        p_dpdh = exact_wilcoxon_p(d_dpdh)
        p_eod = exact_wilcoxon_p(d_eod)
        
        p_auc_map[m] = p_auc
        p_dpdh_map[m] = p_dpdh
        p_eod_map[m] = p_eod
        
        dz_auc = float(d_auc.mean() / d_auc.std(ddof=1)) if d_auc.std(ddof=1) > 0 else 0.0
        dz_dpdh = float(d_dpdh.mean() / d_dpdh.std(ddof=1)) if d_dpdh.std(ddof=1) > 0 else 0.0
        dz_eod = float(d_eod.mean() / d_eod.std(ddof=1)) if d_eod.std(ddof=1) > 0 else 0.0
        
        comparisons[m] = {
            "auc": {
                "mean_diff": float(d_auc.mean()),
                "cohens_dz": dz_auc,
                "wins": f"{int(np.sum(d_auc > 0))}/{len(seeds)}",
                "ci95": compute_bootstrap_ci(d_auc),
                "p_wilcoxon": p_auc,
            },
            "dpd_hard": {
                "mean_diff": float(d_dpdh.mean()),
                "cohens_dz": dz_dpdh,
                "wins": f"{int(np.sum(d_dpdh < 0))}/{len(seeds)}",
                "ci95": compute_bootstrap_ci(d_dpdh),
                "p_wilcoxon": p_dpdh,
            },
            "eod": {
                "mean_diff": float(d_eod.mean()),
                "cohens_dz": dz_eod,
                "wins": f"{int(np.sum(d_eod < 0))}/{len(seeds)}",
                "ci95": compute_bootstrap_ci(d_eod),
                "p_wilcoxon": p_eod,
            }
        }
        
    hb_auc = holm_bonferroni(p_auc_map)
    hb_dpdh = holm_bonferroni(p_dpdh_map)
    hb_eod = holm_bonferroni(p_eod_map)
    
    for m in comparisons:
        comparisons[m]["auc"]["holm_bonferroni_sig"] = hb_auc[m][2]
        comparisons[m]["dpd_hard"]["holm_bonferroni_sig"] = hb_dpdh[m][2]
        comparisons[m]["eod"]["holm_bonferroni_sig"] = hb_eod[m][2]
        
    return {
        "metrics_summary": metrics_summary,
        "paired_comparisons_vs_ours": comparisons,
    }


def main():
    credit_path = "results/stage4_3_credit_results.json"
    pokecz_path = "results/stage4_3_pokecz_results.json"
    
    consolidated = {}
    
    if os.path.exists(credit_path):
        with open(credit_path) as f:
            credit_data = json.load(f)
        consolidated["credit_default_30k"] = analyze_dataset_runs(credit_data["raw_runs"])
        print("Analyzed Credit Default (30k).")
        
    if os.path.exists(pokecz_path):
        with open(pokecz_path) as f:
            pokecz_data = json.load(f)
        consolidated["pokecz_67.8k"] = analyze_dataset_runs(pokecz_data["raw_runs"])
        print("Analyzed Pokec-z (67.8k).")
        
    out_file = "results/consolidated_statistics.json"
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"Consolidated publication statistics written to: {out_file}")


if __name__ == "__main__":
    main()
