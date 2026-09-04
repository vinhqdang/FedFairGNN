"""Generate the attention audit figure for FSER (Issue I3).

Compares the cross-group attention bias ratio

    bias_ratio = (attention mass on cross-group edges) / (share of cross-group edges)

with and without the FSER penalty. A ratio of 1.0 means attention is spread
across the sensitive-attribute boundary exactly in proportion to how common
such edges are; below 1.0 means the model systematically discounts cross-group
neighbours, which is the structural mechanism FSER is meant to suppress.

What the two arms are
---------------------
The "without FSER" arm is ``model="trustfedgnn"`` with ``freeze_beta=True`` and
``beta_init=0.0``. At beta = 0 an ``FSERLayer`` is *exactly* GAT attention, so
this holds the entire backbone (BN, residual, skip-concat, layer widths, the
attention read-out itself) fixed and removes only the fairness term. The
earlier version of this script used ``model="gat"`` instead, which removes FSER
*and* the whole scaffold at once -- and, worse, plain GAT exposes no
``edge_attention()`` method, so ``fser_edge_attention`` returned
``{"available": False}`` and the plotted "without FSER" bar was the hardcoded
``.get(..., 1.0)`` fallback rather than a measurement. The bars are now asserted
to be real measurements; the script fails loudly if a model cannot report its
attention.

Where the model and the data come from
--------------------------------------
``FederatedTrainer`` has no ``server`` object and no ``global_data`` attribute
(the previous version referenced both and could never have run). The trained
global model is recovered by loading ``trainer.global_flat`` into
``trainer.ref_model`` -- the same thing ``evaluate_global`` does -- and the
audit graph is the full pre-partition graph, since attention over
sensitive-attribute boundaries is only meaningful on a graph that still
contains cross-client edges (the induced client subgraphs drop them all).

Outputs:
  - manuscript/figures/revision/fser_attention_audit.pdf
  - results/revision/fser_attention_audit.json
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
from src.federated.client import load_flat_state
from src.trust.explain import fser_edge_attention
from experiments.fairshare_common import full_graph


def _audit_attention(cfg: ExperimentConfig) -> dict:
    """Train under ``cfg`` and measure cross-group attention on the full graph."""
    trainer = FederatedTrainer(cfg)
    trainer.run(verbose=False)

    # The trained global model: exactly what evaluate_global scores with.
    load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
    trainer.ref_model.eval()

    data = full_graph(cfg).to(trainer.device)
    with torch.no_grad():
        att = fser_edge_attention(trainer.ref_model, data)

    if not att.get("available", False):
        raise RuntimeError(
            f"model={cfg.model!r} exposes no edge_attention(); the attention "
            f"audit cannot be measured for this arm. (This is what silently "
            f"produced a hardcoded bias ratio of 1.0 in the previous version "
            f"of this script.)")
    att["mean_beta"] = None if att.get("mean_beta") is None else float(att["mean_beta"])
    return att


def collect_attention_results(datasets=("bail", "credit"), seed=42, rounds=20,
                              num_clients=5) -> dict:
    results = {}
    for ds in datasets:
        common = dict(dataset=ds, seed=seed, rounds=rounds, num_clients=num_clients,
                      aggregator="fedavg", local_fairness=False, dp_enabled=False)

        # 1. Without FSER: same architecture, fairness term switched off.
        print(f"[*] {ds}: training w/o FSER (trustfedgnn, beta frozen at 0.0)...",
              flush=True)
        cfg_nofser = ExperimentConfig.canonical(
            model="trustfedgnn", beta_init=0.0, freeze_beta=True, **common)
        att_nofser = _audit_attention(cfg_nofser)

        # 2. With FSER: beta trained from its canonical initialisation.
        print(f"[*] {ds}: training w/ FSER (trustfedgnn, beta trainable)...",
              flush=True)
        cfg_fser = ExperimentConfig.canonical(model="trustfedgnn", **common)
        att_fser = _audit_attention(cfg_fser)

        results[ds] = {
            "nofser": att_nofser["attention_bias_ratio"],
            "fser": att_fser["attention_bias_ratio"],
            "cross_group_edge_fraction": att_fser["cross_group_edge_fraction"],
            "nofser_detail": att_nofser,
            "fser_detail": att_fser,
        }
        print(f"    -> bias ratio  w/o FSER = {results[ds]['nofser']:.4f} | "
              f"w/ FSER = {results[ds]['fser']:.4f} "
              f"(cross-group edges = {results[ds]['cross_group_edge_fraction'] * 100:.1f}% "
              f"of E)", flush=True)
    return results


def generate_attention_figure(out_pdf="manuscript/figures/revision/fser_attention_audit.pdf",
                              out_json="results/revision/fser_attention_audit.json",
                              datasets=("bail", "credit"), seed=42, rounds=20,
                              num_clients=5):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    datasets = list(datasets)
    results = collect_attention_results(datasets, seed=seed, rounds=rounds,
                                        num_clients=num_clients)

    with open(out_json, "w") as f:
        json.dump({"seed": seed, "rounds": rounds, "num_clients": num_clients,
                   "results": results}, f, indent=2)
    print(f"[+] Saved FSER attention audit JSON to {out_json}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x = np.arange(len(datasets))
    width = 0.3

    r_nofser = [results[d]["nofser"] for d in datasets]
    r_fser = [results[d]["fser"] for d in datasets]

    rects1 = ax.bar(x - width / 2, r_nofser, width,
                    label=r"Without FSER ($\beta$ frozen at 0)", color="#d73027", alpha=0.85)
    rects2 = ax.bar(x + width / 2, r_fser, width,
                    label=r"With FSER ($\beta$ trained)", color="#1b7837", alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
               label="Parity baseline (ratio = 1.0)")

    ax.set_ylabel("Cross-Group Attention Bias Ratio\n(Attn Mass / Edge Prevalence)", fontsize=10)
    ax.set_title("FSER Structural Regularization: Cross-Group Attention", fontsize=11,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.set_ylim(0, max(max(r_nofser), max(r_fser), 1.0) * 1.25)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved FSER attention audit figure to {out_pdf}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["bail", "credit"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--out-pdf", default="manuscript/figures/revision/fser_attention_audit.pdf")
    ap.add_argument("--out-json", default="results/revision/fser_attention_audit.json")
    a = ap.parse_args()
    generate_attention_figure(out_pdf=a.out_pdf, out_json=a.out_json,
                              datasets=a.datasets, seed=a.seed, rounds=a.rounds,
                              num_clients=a.num_clients)


if __name__ == "__main__":
    main()
