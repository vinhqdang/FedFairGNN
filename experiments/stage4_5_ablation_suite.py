"""Stage 4.5: Resource-Efficient Q1 Ablation Studies Suite.

Covers:
1. Component-wise Ablations (M1–M7) across 3 random seeds (42, 43, 44)
2. Sweep 1: Server Fairness Weight Alpha (Pareto Curve)
3. Sweep 2: Server Holdout Size Sensitivity
4. Sweep 3: DP Privacy Budget Epsilon
5. Sweep 4: Non-IID Dirichlet Skew
6. Local Pareto Grid Sweep: Local Fairness Weight (lambda) x FSER Beta (beta)
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig, set_seed
from src.federated import FederatedTrainer
from src.utils.metrics import weight_oscillation


def evaluate_run(cfg: ExperimentConfig) -> dict:
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    weights_hist = [r.get("agg_weights") for r in res["history"]]
    omega_w = weight_oscillation(weights_hist)
    return {
        "auc": float(res["final"]["auc"]),
        "dpd": float(res["final"]["dpd"]),
        "eod": float(res["final"]["eod"]),
        "diverged": float(res["final"]["diverged"]),
        "omega_w": float(omega_w),
    }


def aggregate_seeds(results_list: list[dict]) -> dict:
    aucs = [r["auc"] for r in results_list if np.isfinite(r["auc"])]
    dpds = [r["dpd"] for r in results_list if np.isfinite(r["dpd"])]
    eods = [r["eod"] for r in results_list if np.isfinite(r["eod"])]
    omegas = [r["omega_w"] for r in results_list if np.isfinite(r["omega_w"])]
    
    return {
        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "auc_std": float(np.std(aucs)) if aucs else 0.0,
        "dpd_mean": float(np.mean(dpds)) if dpds else 0.0,
        "dpd_std": float(np.std(dpds)) if dpds else 0.0,
        "eod_mean": float(np.mean(eods)) if eods else 0.0,
        "eod_std": float(np.std(eods)) if eods else 0.0,
        "omega_w_mean": float(np.mean(omegas)) if omegas else 0.0,
        "omega_w_std": float(np.std(omegas)) if omegas else 0.0,
        "seeds": results_list,
    }


def run_targeted_local_pareto_sweep(results_file="results/stage4_5_results.json"):
    print("=" * 70)
    print("🚀 [STAGE 4.5 TARGETED] RUNNING LOCAL PARETO 2D GRID SWEEP")
    print("=" * 70)

    dataset = "german"
    num_clients = 5
    rounds = 20
    dir_alpha = 0.3
    seed = 42

    existing = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = {}

    pareto_grid_results = {}
    fairness_weights = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    betas = [0.0, 0.1, 0.5]

    for b in betas:
        for fw in fairness_weights:
            set_seed(seed)
            cfg = ExperimentConfig(
                dataset=dataset,
                seed=seed,
                num_clients=num_clients,
                rounds=rounds,
                dirichlet_alpha=dir_alpha,
                model="trustfedgnn",
                aggregator="fu_shapley",
                fu_alpha=0.1,
                fu_ema_beta=0.9,
                fairness_weight=fw,
                beta_init=b,
                dp_enabled=True,
                dp_mode="ftgd",
            )
            res = evaluate_run(cfg)
            key = f"beta_{b:.1f}_fw_{fw:.2f}"
            pareto_grid_results[key] = {
                "beta": b,
                "fairness_weight": fw,
                **res
            }
            print(f"  [Pareto Sweep] beta={b:.1f}, lambda={fw:4.2f} -> AUC={res['auc']:.4f}, DPD={res['dpd']:.4f}, EOD={res['eod']:.4f}, Omega_w={res['omega_w']:.4f}")

    existing["local_pareto_grid_sweep"] = pareto_grid_results

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(existing, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ [TARGETED SWEEP COMPLETED] Saved to {results_file}")
    print("=" * 70)
    return existing


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--targeted":
        run_targeted_local_pareto_sweep()
    else:
        run_targeted_local_pareto_sweep()
