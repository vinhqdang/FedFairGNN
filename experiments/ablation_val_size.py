"""FS-WI-5: how small can the server validation set be, and how sensitive is the
FU-Shapley score to the validation distribution?

At a fixed round we extract the K client pseudo-gradients once, then recompute
the target gradient (and hence phi_k) under:
  * shrinking validation subsamples |D_val| in {50, 100, 200, full}, and
  * distribution shifts: iid subsample, sensitive-skewed, 10% label noise,
correlating each phi against the full-clean phi. A high correlation at ~100
nodes is the answer to the reviewer's "the server having D_val is a strong
assumption" (plan risk table).

    python -m experiments.ablation_val_size --dataset german --sizes 50 100 full
"""
from __future__ import annotations

import argparse
import csv
import os

import torch

from src.federated.client import load_flat_state
from src.trust.incentive import get_server_target_gradients_pooled, compute_fu_weights
from experiments.fairshare_common import (
    make_trainer, warm_rounds, client_pseudo_grads, pearson_spearman,
)


def _subsample_val(clients_data, size, mode, seed):
    """Return a shallow copy of clients_data with val_mask reduced to ~`size`
    total nodes according to `mode`. `size='full'` keeps everything."""
    g = torch.Generator().manual_seed(seed)
    out = []
    # gather global pool of (client, node) val indices
    per_client = []
    total = 0
    for d in clients_data:
        idx = d.val_mask.nonzero(as_tuple=True)[0]
        per_client.append(idx); total += idx.numel()
    keep_total = total if size == "full" else min(int(size), total)
    for d, idx in zip(clients_data, per_client):
        d2 = d.clone()
        if size == "full" or idx.numel() == 0:
            out.append(d2); continue
        share = max(1, round(keep_total * idx.numel() / max(1, total)))
        if mode == "s_skew":                       # bias toward group 1 nodes
            s = d.sensitive_attr[idx]
            order = torch.argsort((s == 1).float(), descending=True)
            sel = idx[order[:share]]
        else:
            perm = torch.randperm(idx.numel(), generator=g)[:share]
            sel = idx[perm]
        new_mask = torch.zeros_like(d.val_mask)
        new_mask[sel] = True
        d2.val_mask = new_mask
        if mode == "label_noise":                  # flip 10% of val labels
            d2 = d2.clone()
            y = d2.y.clone()
            flip = sel[torch.rand(sel.numel(), generator=g) < 0.10]
            y[flip] = 1 - y[flip]
            d2.y = y
        out.append(d2)
    return out


def phi_for(trainer, grads, clients_data, alpha):
    load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
    tg = get_server_target_gradients_pooled(trainer.ref_model, clients_data,
                                            trainer.device, alpha,
                                            fair_surrogate=trainer.cfg.fu_fair_surrogate)
    g_target_cpu = tg[0].cpu()
    _, phi, _ = compute_fu_weights(grads, g_target_cpu, normalize="none")
    return phi.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--sizes", nargs="+", default=["50", "100", "full"])
    p.add_argument("--out", default="results/fairshare")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trainer = make_trainer(dataset=args.dataset, seed=args.seed, num_clients=5,
                           rounds=args.rounds + 2, method="fairshare", alpha=args.alpha)
    warm_rounds(trainer, args.rounds)
    grads = client_pseudo_grads(trainer)

    ref = phi_for(trainer, grads, trainer.clients_data, args.alpha)   # full clean
    rows = []
    # (a) size sweep, clean iid subsample
    for size in args.sizes:
        cd = _subsample_val(trainer.clients_data, size, "iid", args.seed)
        phi = phi_for(trainer, grads, cd, args.alpha)
        pr, sr = pearson_spearman(phi, ref)
        n = sum(int(d.val_mask.sum()) for d in cd)
        rows.append({"setting": f"size={size}", "val_nodes": n,
                     "pearson_vs_full": round(pr, 4), "spearman_vs_full": round(sr, 4)})
        print(f"size={size:>4s} nodes={n:4d} pearson={pr:.3f}")
    # (b) distribution-shift at a fixed moderate size
    for mode in ["iid", "s_skew", "label_noise"]:
        cd = _subsample_val(trainer.clients_data, "100", mode, args.seed)
        phi = phi_for(trainer, grads, cd, args.alpha)
        pr, sr = pearson_spearman(phi, ref)
        n = sum(int(d.val_mask.sum()) for d in cd)
        rows.append({"setting": f"shift={mode}", "val_nodes": n,
                     "pearson_vs_full": round(pr, 4), "spearman_vs_full": round(sr, 4)})
        print(f"shift={mode:11s} nodes={n:4d} pearson={pr:.3f}")

    path = os.path.join(args.out, f"ablation_val_size__{args.dataset}__s{args.seed}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["setting", "val_nodes", "pearson_vs_full", "spearman_vs_full"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
