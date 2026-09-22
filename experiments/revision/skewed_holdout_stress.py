"""T15 -- Skewed Holdout Representativeness Stress Test (P3-D1).

Audits how sensitive TrustFedGNN's reference vector (g_target) is to demographic,
topological, and label skew in the server holdout set D_root (|D_root| = 125 nodes).

Evaluates 4 Holdout Extraction Regimes:
    1. balanced: 50% s=0, 50% s=1 (Global Demographic Parity baseline)
    2. sensitive_skew_80_20: 80% s=1, 20% s=0 (Demographic Disparity Skew)
    3. community_skew: 125 nodes sampled exclusively from a single Louvain community
    4. label_skew: 75% y=1, 25% y=0 (Severe Label Shift)

Tested under:
    - Benign operational regime
    - Fairness poisoning attack (intensity = 1.0, 1 Byzantine client)

Outputs:
    results/revision/skewed_holdout_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import ExperimentConfig
from src.data.datasets import load_dataset
from src.data.partition import _induced, partition_graph, partition_stats
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest, resolve_actual_device

HOLDOUT_REGIMES = ["balanced", "sensitive_skew_80_20", "community_skew", "label_skew"]
DEFAULT_SEEDS = list(range(42, 52))


def carve_skewed_holdout(
    data: Data, size: int = 125, regime: str = "balanced", seed: int = 42
) -> Tuple[Data, Data]:
    """Extract a 125-node holdout from data under specific skew constraints."""
    val_idx = torch.nonzero(data.val_mask, as_tuple=False).flatten()
    s = data.sensitive_attr[val_idx].cpu().numpy()
    y = data.y[val_idx].cpu().numpy()
    gen = np.random.RandomState(seed)

    target_size = min(int(size), int(len(val_idx) // 2))

    if regime == "balanced":
        idx_0 = val_idx[s == 0].tolist()
        idx_1 = val_idx[s == 1].tolist()
        n_half = target_size // 2
        pick_0 = gen.choice(idx_0, size=min(n_half, len(idx_0)), replace=False)
        pick_1 = gen.choice(idx_1, size=min(target_size - len(pick_0), len(idx_1)), replace=False)
        picked = np.concatenate([pick_0, pick_1])

    elif regime == "sensitive_skew_80_20":
        idx_0 = val_idx[s == 0].tolist()
        idx_1 = val_idx[s == 1].tolist()
        n_maj = int(0.80 * target_size)
        n_min = target_size - n_maj
        pick_1 = gen.choice(idx_1, size=min(n_maj, len(idx_1)), replace=False)
        pick_0 = gen.choice(idx_0, size=min(n_min, len(idx_0)), replace=False)
        picked = np.concatenate([pick_1, pick_0])

    elif regime == "community_skew":
        # Extract from largest Louvain community in val set
        edge_index = data.edge_index.cpu().numpy()
        g = nx.Graph()
        g.add_nodes_from(range(data.num_nodes))
        edges = list(zip(edge_index[0], edge_index[1]))
        g.add_edges_from(edges)
        communities = nx.community.louvain_communities(g, seed=seed)

        val_set = set(val_idx.tolist())
        # Find community with most validation nodes
        best_comm = []
        for comm in communities:
            c_val = list(comm.intersection(val_set))
            if len(c_val) > len(best_comm):
                best_comm = c_val

        if len(best_comm) >= target_size:
            picked = gen.choice(best_comm, size=target_size, replace=False)
        else:
            remaining = list(val_set.difference(set(best_comm)))
            fill = gen.choice(remaining, size=target_size - len(best_comm), replace=False)
            picked = np.concatenate([best_comm, fill])

    elif regime == "label_skew":
        idx_0 = val_idx[y == 0].tolist()
        idx_1 = val_idx[y == 1].tolist()
        n_maj = int(0.75 * target_size)
        n_min = target_size - n_maj
        pick_1 = gen.choice(idx_1, size=min(n_maj, len(idx_1)), replace=False)
        pick_0 = gen.choice(idx_0, size=min(n_min, len(idx_0)), replace=False)
        picked = np.concatenate([pick_1, pick_0])
    else:
        raise ValueError(f"Unknown regime: {regime}")

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[picked] = True

    holdout = _induced(data, torch.nonzero(mask, as_tuple=False).flatten())
    holdout.val_mask = torch.ones(holdout.num_nodes, dtype=torch.bool)
    holdout.train_mask = torch.zeros(holdout.num_nodes, dtype=torch.bool)
    holdout.test_mask = torch.zeros(holdout.num_nodes, dtype=torch.bool)

    rest = _induced(data, torch.nonzero(~mask, as_tuple=False).flatten())
    return holdout, rest


def run_skewed_holdout_seed(
    regime: str, seed: int, dataset: str = "german",
    num_clients: int = 5, rounds: int = 20, attack: str = "fairness_poison",
    device: str = "cpu"
) -> Dict:
    t0 = time.perf_counter()

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator="fu_shapley",
        attack=attack,
        num_byzantine=1 if attack != "none" else 0,
        fu_alpha=0.1,
        fu_grad_clip=10.0,
        dp_enabled=True,
    )

    # Load full dataset and carve server holdout using the specified regime
    data_full = load_dataset(dataset)
    holdout, rest_graph = carve_skewed_holdout(data_full, size=125, regime=regime, seed=seed)

    trainer = FederatedTrainer(cfg)
    trainer.server_holdout = holdout
    trainer.clients_data = partition_graph(
        rest_graph, cfg.num_clients, method=cfg.partition,
        alpha=cfg.dirichlet_alpha, by=cfg.partition_by, seed=cfg.seed
    )
    trainer.partition_stats = partition_stats(trainer.clients_data)

    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    history = res.get("history", [])
    w_adv_list = []
    for r in history:
        w = r.get("agg_weights")
        if w is not None and len(w) == num_clients and attack != "none":
            w_adv_list.append(w[0])

    final = res.get("final", {})
    return {
        "regime": regime,
        "seed": seed,
        "attack": attack,
        "w_adv": float(np.mean(w_adv_list)) if w_adv_list else 0.0,
        "auc": float(final.get("auc", float("nan"))),
        "dpd_hard": float(final.get("dpd_hard", float("nan"))),
        "eod": float(final.get("eod", float("nan"))),
        "wall_clock_s": wall_clock_s,
    }


def main():
    parser = argparse.ArgumentParser(description="T15 Skewed Holdout Evaluation Suite")
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--attack", type=str, default="fairness_poison")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed, 2 rounds smoke test")
    parser.add_argument("--output", type=str, default="results/revision/skewed_holdout_results.json")
    args = parser.parse_args()

    seeds = [42] if args.smoke else DEFAULT_SEEDS
    rounds = 2 if args.smoke else args.rounds

    print(f"=== T15 Skewed Holdout Representativeness Stress Test ===")
    print(f"Dataset: {args.dataset}, Clients: {args.num_clients}, Rounds: {rounds}, Attack: {args.attack}")

    results_by_regime = {reg: [] for reg in HOLDOUT_REGIMES}

    for reg in HOLDOUT_REGIMES:
        print(f"\nEvaluating Holdout Regime: {reg}")
        for s in seeds:
            res = run_skewed_holdout_seed(
                regime=reg, seed=s, dataset=args.dataset,
                num_clients=args.num_clients, rounds=rounds, attack=args.attack,
                device=args.device
            )
            results_by_regime[reg].append(res)
            print(f"  Seed {s:2d} -> w_adv: {res['w_adv']:.4f}, AUC: {res['auc']:.4f}, DPD: {res['dpd_hard']:.4f}")

    summary = {}
    for reg in HOLDOUT_REGIMES:
        items = results_by_regime[reg]
        w_vals = [r["w_adv"] for r in items if not math.isnan(r["w_adv"])]
        aucs = [r["auc"] for r in items if not math.isnan(r["auc"])]
        dpds = [r["dpd_hard"] for r in items if not math.isnan(r["dpd_hard"])]
        eods = [r["eod"] for r in items if not math.isnan(r["eod"])]

        summary[reg] = {
            "w_adv_mean": float(np.mean(w_vals)) if w_vals else None,
            "w_adv_std": float(np.std(w_vals, ddof=1)) if len(w_vals) > 1 else 0.0,
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "dpd_mean": float(np.mean(dpds)) if dpds else None,
            "dpd_std": float(np.std(dpds, ddof=1)) if len(dpds) > 1 else 0.0,
            "eod_mean": float(np.mean(eods)) if eods else None,
            "eod_std": float(np.std(eods, ddof=1)) if len(eods) > 1 else 0.0,
        }

    manifest = build_manifest(dataset=args.dataset, num_seeds=len(seeds), rounds=rounds)
    manifest["device"] = resolve_actual_device(args.device)
    manifest["task"] = "T15_skewed_holdout_stress"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"manifest": manifest, "summary": summary, "raw_runs": results_by_regime}, f, indent=2)

    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
