"""FS-WI-7: "why a graph method?" -- relate per-client graph structure to the
fairness Shapley credit.

For a fairshare run we compute, per client, structural descriptors (edge label
homophily, average degree, sensitive-boundary-edge ratio) and the client's
time-averaged fairness credit phi_fair, then correlate. The hypothesis (plan):
clients with lower homophily / more cross-group boundary edges carry higher
fairness credit -- a signal invisible to non-graph incentive methods.

    python -m experiments.topology_shapley_analysis --dataset german --rounds 20
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.fairshare_common import make_trainer, topology_metrics, pearson_spearman


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--out", default="results/fairshare")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    tr = make_trainer(dataset=args.dataset, seed=args.seed, num_clients=args.num_clients,
                      rounds=args.rounds, method="fairshare", alpha=args.alpha)
    res = tr.run(verbose=False)
    H = [h for h in res["history"] if "phi_fair" in h]
    if not H:
        print("no phi_fair logged"); return
    phi_fair = np.mean([h["phi_fair"] for h in H], axis=0)     # time-averaged per client

    topo = [topology_metrics(d) for d in tr.clients_data]
    rows = []
    for k, tm in enumerate(topo):
        rows.append({"client": k, "phi_fair_mean": round(float(phi_fair[k]), 5), **tm})

    path = os.path.join(args.out, f"topology_shapley__{args.dataset}__s{args.seed}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # correlations of each structural descriptor with fairness credit
    print(f"{'metric':16s} pearson spearman")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, key in zip(axes, ["homophily", "avg_degree", "boundary_ratio"]):
        xs = [r[key] for r in rows]; ys = [r["phi_fair_mean"] for r in rows]
        pr, sr = pearson_spearman(xs, ys)
        print(f"{key:16s} {pr:7.3f} {sr:7.3f}")
        ax.scatter(xs, ys, c="#1b7837", s=50)
        ax.set_xlabel(key); ax.set_ylabel(r"$\bar\phi^{\mathrm{fair}}_k$")
        ax.set_title(f"r={pr:.2f}")
    fig.tight_layout()
    fpng = os.path.join(args.out, "figures", f"topology_shapley__{args.dataset}.png")
    fig.savefig(fpng); plt.close(fig)
    print(f"wrote {path} and {fpng}")


if __name__ == "__main__":
    main()
