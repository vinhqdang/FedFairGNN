"""Publication-quality Pareto Frontier plotter (DPD vs AUC) for Q1 papers.

Plots:
- Left Subplot: Credit Default (30,000 nodes, hs = 0.9595)
- Right Subplot: Pokec-z (67,796 nodes, hs = 0.9506)
- X-axis: Demographic Parity Difference (DPD_hard@0.5, lower is better)
- Y-axis: Area Under ROC Curve (AUC-ROC, higher is better)
- Pareto optimal frontier lines, distinct markers and error bars (±1 std).
"""

from __future__ import annotations

import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Method color and marker palette
STYLE_MAP = {
    "fedavg-gcn": {"color": "#757575", "marker": "o", "label": "FedAvg-GCN (AISTATS'17)"},
    "fairgnn": {"color": "#8e24aa", "marker": "s", "label": "FairGNN (WSDM'21)"},
    "fairsin": {"color": "#00acc1", "marker": "^", "label": "FairSIN (WWW'24)"},
    "fairfed": {"color": "#fb8c00", "marker": "v", "label": "FairFed (AAAI'23)"},
    "fairgfl": {"color": "#43a047", "marker": "D", "label": "FairGFL (TPDS'26)"},
    "fedgraphfair": {"color": "#3949ab", "marker": "P", "label": "FedGraph-Fair (InfoSci'26)"},
    "cgsv": {"color": "#f4511e", "marker": "X", "label": "CGSV Non-DP (NeurIPS'21)"},
    "ours-nofser": {"color": "#d81b60", "marker": "*", "label": "Ours w/o FSER (Ablation)"},
    "fedfairgnn": {"color": "#e53935", "marker": "h", "label": "TrustFedGNN (Ours Canonical)"},
}


def plot_pareto(credit_json_path: str, pokecz_json_path: str, out_png_path: str):
    if not HAS_MPL:
        print("[!] Matplotlib is not available in the current environment.")
        return

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["grid.alpha"] = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    datasets = [
        ("Credit Default (30k Nodes, $h_s=0.96$)", credit_json_path, axes[0]),
        ("Pokec-z (67.8k Nodes, $h_s=0.95$)", pokecz_json_path, axes[1]),
    ]

    for title, json_path, ax in datasets:
        if not os.path.exists(json_path):
            print(f"[!] Warning: File {json_path} does not exist. Skipping.")
            continue
            
        with open(json_path) as f:
            data = json.load(f)
            
        baselines = data.get("baselines", {})
        
        pts = []
        for name, info in baselines.items():
            summary = info.get("summary", {})
            auc_m = summary.get("auc", {}).get("mean", 0.0)
            auc_s = summary.get("auc", {}).get("std", 0.0)
            dpd_m = summary.get("dpd_hard", {}).get("mean", 0.0)
            dpd_s = summary.get("dpd_hard", {}).get("std", 0.0)
            
            style = STYLE_MAP.get(name, {"color": "#333333", "marker": "o", "label": name})
            
            is_ours = (name == "fedfairgnn")
            size = 140 if is_ours else 90
            zorder = 10 if is_ours else 5
            
            ax.errorbar(
                dpd_m, auc_m,
                xerr=dpd_s, yerr=auc_s,
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.2,
                capsize=3,
                alpha=0.7,
                zorder=zorder - 1
            )
            
            ax.scatter(
                dpd_m, auc_m,
                c=style["color"],
                marker=style["marker"],
                s=size,
                label=style["label"],
                edgecolors="black" if is_ours else "none",
                linewidths=1.5 if is_ours else 0,
                zorder=zorder,
                alpha=0.95 if is_ours else 0.85
            )
            pts.append((dpd_m, auc_m, name))

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(r"Demographic Parity Difference $\Delta_{\mathrm{DP}}^{\mathrm{hard}} \downarrow$ (Fairness)", fontsize=11, labelpad=8)
        ax.set_ylabel(r"AUC-ROC $\uparrow$ (Utility)", fontsize=11, labelpad=8)
        ax.grid(True, linestyle="--", linewidth=0.6)
        
        # Invert x-axis conceptually or annotate optimal direction
        ax.annotate("Optimal Trade-off\n(Low DPD, High AUC)", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=9, color="#2e7d32", fontweight="semibold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#81c784", lw=1))

    # Single unified legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, bbox_inches="tight", dpi=300)
    print(f"[*] Pareto Frontier plot saved successfully to: {out_png_path}")


def main():
    credit_json = "results/stage4_3_credit_results.json"
    pokecz_json = "results/stage4_3_pokecz_results.json"
    out_png = "results/pareto_frontier_credit_pokecz.png"
    plot_pareto(credit_json, pokecz_json, out_png)


if __name__ == "__main__":
    main()
