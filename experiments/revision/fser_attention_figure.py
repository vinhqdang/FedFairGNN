"""Generate attention audit figure for FSER (Issue I3).

Compares cross-group attention bias ratio before and after FSER:
  bias_ratio = mean_attn(cross_group_edges) / mean_attn(same_group_edges)
across GNN layers on Pokec-z and Bail.

Outputs:
  - manuscript/figures/revision/fser_attention_audit.pdf
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated import FederatedTrainer
from src.trust.explain import fser_edge_attention


def generate_attention_figure(out_pdf="manuscript/figures/revision/fser_attention_audit.pdf"):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # Compare on Pokec-z or Bail
    datasets = ["bail", "credit"]
    results = {}

    for ds in datasets:
        # 1. Without FSER
        cfg_nofser = ExperimentConfig.canonical(
            dataset=ds, seed=42, rounds=20, num_clients=5,
            model="gat", aggregator="fedavg", local_fairness=False, dp_enabled=False
        )
        trainer_nofser = FederatedTrainer(cfg_nofser)
        trainer_nofser.run(verbose=False)
        # Audit attention on global data
        att_nofser = fser_edge_attention(trainer_nofser.server.model, trainer_nofser.global_data)

        # 2. With FSER
        cfg_fser = ExperimentConfig.canonical(
            dataset=ds, seed=42, rounds=20, num_clients=5,
            model="trustfedgnn", aggregator="fedavg", local_fairness=False, dp_enabled=False
        )
        trainer_fser = FederatedTrainer(cfg_fser)
        trainer_fser.run(verbose=False)
        att_fser = fser_edge_attention(trainer_fser.server.model, trainer_fser.global_data)

        results[ds] = {
            "nofser": att_nofser.get("attention_bias_ratio", 1.0),
            "fser": att_fser.get("attention_bias_ratio", 1.0),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.arange(len(datasets))
    width = 0.3

    r_nofser = [results[d]["nofser"] for d in datasets]
    r_fser = [results[d]["fser"] for d in datasets]

    rects1 = ax.bar(x - width/2, r_nofser, width, label="Without FSER (Standard GAT)", color="#d73027", alpha=0.85)
    rects2 = ax.bar(x + width/2, r_fser, width, label="With FSER (TrustFedGNN)", color="#1b7837", alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Parity baseline (ratio = 1.0)")

    ax.set_ylabel("Cross-Group Attention Bias Ratio\n(Attn Mass / Edge Prevalence)", fontsize=10)
    ax.set_title("FSER Structural Regularization: Suppressing Biased Attention", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.set_ylim(0, max(max(r_nofser), max(r_fser)) * 1.25)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved FSER attention audit figure to {out_pdf}")


if __name__ == "__main__":
    generate_attention_figure()
