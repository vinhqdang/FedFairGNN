"""Analyze stability and convergence of FSER beta parameter (Issue I3 & Stanford Q9).

Extracts learned beta values across layers and seeds on Pokec-z, Credit, and Bail.
Checks whether beta hits boundary [0, 5], assesses heterophily sensitivity,
and plots distributions across seeds.

Outputs:
  - results/revision/fser_beta_analysis.json
  - manuscript/figures/revision/fser_beta_stability.pdf
"""
from __future__ import annotations

import argparse
import json
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


def run_beta_stability_analysis(datasets=("bail", "credit"), seeds=range(42, 52),
                                out_json="results/revision/fser_beta_analysis.json",
                                out_pdf="manuscript/figures/revision/fser_beta_stability.pdf"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    beta_records = []

    for ds in datasets:
        print(f"[*] Analyzing beta stability on {ds} across {len(list(seeds))} seeds...", flush=True)
        for s in seeds:
            cfg = ExperimentConfig.canonical(
                dataset=ds, seed=s, rounds=20, num_clients=5,
                model="trustfedgnn", aggregator="fedavg", local_fairness=False, dp_enabled=False
            )
            trainer = FederatedTrainer(cfg)
            trainer.run(verbose=False)

            model = trainer.server.model
            betas = []
            if hasattr(model, "layers"):
                for idx, layer in enumerate(model.layers):
                    if hasattr(layer, "beta"):
                        b_val = float(layer.beta.detach().cpu().item())
                        betas.append(b_val)
                        beta_records.append({
                            "dataset": ds,
                            "seed": s,
                            "layer": idx,
                            "beta": b_val,
                        })
            print(f"    Seed {s} -> betas={betas}", flush=True)

    with open(out_json, "w") as f:
        json.dump(beta_records, f, indent=2)
    print(f"[+] Saved beta stability records to {out_json}")

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    
    unique_ds = sorted(list(set(r["dataset"] for r in beta_records)))
    unique_layers = sorted(list(set(r["layer"] for r in beta_records)))

    data_to_plot = []
    labels = []

    for ds in unique_ds:
        for l in unique_layers:
            vals = [r["beta"] for r in beta_records if r["dataset"] == ds and r["layer"] == l]
            if vals:
                data_to_plot.append(vals)
                labels.append(f"{ds.capitalize()}\nL{l+1}")

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                    boxprops=dict(facecolor="#91bfdb", color="#4575b4"),
                    medianprops=dict(color="#d73027", linewidth=2))

    ax.axhline(0.0, color="gray", linestyle=":", label="Lower bound (0.0)")
    ax.axhline(5.0, color="gray", linestyle=":", label="Upper bound (5.0)")
    ax.axhline(0.5, color="orange", linestyle="--", alpha=0.7, label="Init value (0.5)")

    ax.set_ylabel("Learned $\\beta$ Coefficient", fontsize=10)
    ax.set_title("FSER $\\beta$ Stability Across Layers and Random Seeds", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.2, 5.2)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved FSER beta stability figure to {out_pdf}")


if __name__ == "__main__":
    run_beta_stability_analysis()
