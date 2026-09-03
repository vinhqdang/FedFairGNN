"""FS-WI-3: explainability figures for FairShare-GNN.

Produces two figures from a fairshare run's per-round history:
  1. phi_util vs phi_fair scatter (last round) -- good clients in Q-I, biased
     clients with negative fairness credit, Byzantine clients gated (phi<0).
  2. stacked-area weight trajectory w_k(t) -- FairShare should be smooth versus
     BFWA's one-hot oscillation (visualises finding F6 / the V-3 vulnerability).

Reads a logged run if --run_id is given, else trains a short one on the fly.

    python -m experiments.plot_shapley --dataset german --rounds 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history_from_run(run_id, results="results"):
    p = os.path.join(results, "runs", f"{run_id}.json")
    with open(p) as f:
        return json.load(f)["history"]


def history_from_fresh(dataset, seed, rounds, alpha):
    from experiments.fairshare_common import make_trainer
    tr = make_trainer(dataset=dataset, seed=seed, num_clients=5, rounds=rounds,
                      method="fairshare", alpha=alpha)
    return tr.run(verbose=False)["history"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default=None)
    p.add_argument("--dataset", default="german")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", default="results/fairshare/figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    H = history_from_run(args.run_id) if args.run_id else \
        history_from_fresh(args.dataset, args.seed, args.rounds, args.alpha)
    H = [h for h in H if h.get("agg_weights")]
    if not H:
        print("no weighted rounds to plot"); return

    # ---- Fig 1: decomposition scatter (last round with phi) ----
    dec = [h for h in H if "phi_util" in h and "phi_fair" in h]
    if dec:
        last = dec[-1]
        u = np.array(last["phi_util"]); fr = np.array(last["phi_fair"])
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.scatter(u, fr, s=60, c=np.where(u > 0, "#1b7837", "#d73027"), zorder=3)
        for k, (xu, yf) in enumerate(zip(u, fr)):
            ax.annotate(f"C{k}", (xu, yf), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(r"$\phi_k^{\mathrm{util}}$"); ax.set_ylabel(r"$\phi_k^{\mathrm{fair}}$")
        ax.set_title("Shapley contribution decomposition")
        f1 = os.path.join(args.out, f"shapley_scatter__{args.dataset}.png")
        fig.savefig(f1); plt.close(fig); print("wrote", f1)

    # ---- Fig 2: weight trajectory (stacked area) ----
    W = np.array([h["agg_weights"] for h in H])
    rounds = [h["round"] for h in H]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.stackplot(rounds, W.T, alpha=0.85)
    ax.set_xlabel("round"); ax.set_ylabel(r"aggregation weight $w_k$")
    ax.set_title(f"Weight trajectory (osc={np.mean(np.var(np.diff(W,axis=0),axis=0)):.4f})")
    ax.set_ylim(0, 1); ax.margins(x=0)
    f2 = os.path.join(args.out, f"weight_trajectory__{args.dataset}.png")
    fig.savefig(f2); plt.close(fig); print("wrote", f2)


if __name__ == "__main__":
    main()
