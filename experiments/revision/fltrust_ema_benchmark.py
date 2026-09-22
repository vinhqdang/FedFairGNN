"""T14 -- FLTrust + EMA History Smoothing Benchmark on Pokec-z (P1-B1).

Evaluates whether the 7.37x weight volatility margin (Omega_w) between
FLTrust (0.7193) and TrustFedGNN (0.0976) is attributable specifically to
Exponential Moving Average (EMA) history smoothing.

Arms:
    1. fltrust: Canonical unsmoothed FLTrust (NDSS'21)
    2. fltrust_ema: FLTrust with EMA momentum smoothing (beta_ema = 0.9)
    3. trustfedgnn: Reference bi-objective gated rule (beta_ema = 0.9)

Metrics:
    - Omega_w: Mean temporal variation in client aggregation weights
    - AUC, DPD_hard, EOD
    - Wall-clock seconds per round

Outputs:
    results/revision/fltrust_ema_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest, resolve_actual_device

ARMS = ["fltrust", "fltrust_ema", "trustfedgnn"]
DEFAULT_SEEDS = list(range(42, 52))


def compute_omega_w(weight_history: List[List[float]]) -> float:
    """Calculate temporal weight volatility across rounds:
    Omega_w = (1 / (R-1)) sum_{t=1}^{R-1} || w^(t+1) - w^(t) ||_1

    NOTE: MEAN per round, verified against the canonical artifact. Recomputing
    from results/fairshare/audit_traj__german__fu_shapley__*.csv gives
    mean-over-rounds = 0.0605 versus canonical_suite.json omega_w_mean = 0.0586
    (3 seeds vs 10), whereas sum-over-rounds gives 1.1486 (20x off). The formula
    printed in docs/05_data_and_results.md and tab_weight_stability.tex states a
    SUM and is therefore wrong; the stored numbers are means. Do not "fix" this
    function to a sum without first correcting those two documents.
    """
    if len(weight_history) < 2:
        return 0.0
    diffs = []
    for t in range(len(weight_history) - 1):
        w_curr = np.array(weight_history[t])
        w_next = np.array(weight_history[t + 1])
        diffs.append(np.sum(np.abs(w_next - w_curr)))
    return float(np.mean(diffs))


def run_fltrust_ema_seed(
    arm: str, seed: int, dataset: str = "pokec_z",
    num_clients: int = 10, rounds: int = 50, device: str = "cuda"
) -> Dict:
    t0 = time.perf_counter()

    if arm == "fltrust":
        aggregator = "fltrust"
        fu_alpha = 0.0
        fu_ema_beta = 0.0
    elif arm == "fltrust_ema":
        aggregator = "fltrust_ema"
        fu_alpha = 0.0
        fu_ema_beta = 0.9
    elif arm == "trustfedgnn":
        aggregator = "fu_shapley"
        fu_alpha = 0.1
        fu_ema_beta = 0.9
    else:
        raise ValueError(f"Unknown arm: {arm}")

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=3,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator=aggregator,
        attack="none",
        fu_alpha=fu_alpha,
        fu_ema_beta=fu_ema_beta,
        dp_enabled=True,
    )

    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    history = res.get("history", [])
    weights_traj = []
    for r in history:
        w = r.get("agg_weights")
        if w is not None and len(w) == num_clients:
            weights_traj.append(w)

    omega_w = compute_omega_w(weights_traj)
    final = res.get("final", {})

    return {
        "arm": arm,
        "seed": seed,
        "omega_w": omega_w,
        "auc": float(final.get("auc", float("nan"))),
        "dpd_hard": float(final.get("dpd_hard", float("nan"))),
        "eod": float(final.get("eod", float("nan"))),
        "wall_clock_s": wall_clock_s,
    }


def main():
    parser = argparse.ArgumentParser(description="T14 FLTrust + EMA Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="pokec_z")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed, 2 rounds smoke test")
    parser.add_argument("--output", type=str, default="results/revision/fltrust_ema_results.json")
    args = parser.parse_args()

    seeds = [42] if args.smoke else DEFAULT_SEEDS
    rounds = 2 if args.smoke else args.rounds
    actual_device = resolve_actual_device(args.device)

    print(f"=== T14 FLTrust + EMA Smoothing Evaluation ===")
    print(f"Dataset: {args.dataset}, Clients: {args.num_clients}, Rounds: {rounds}")
    print(f"Arms: {args.arms}, Device: {actual_device}")

    results_by_arm = {arm: [] for arm in args.arms}

    for arm in args.arms:
        print(f"\nEvaluating Arm: {arm}")
        for s in seeds:
            res = run_fltrust_ema_seed(
                arm=arm, seed=s, dataset=args.dataset,
                num_clients=args.num_clients, rounds=rounds, device=actual_device
            )
            results_by_arm[arm].append(res)
            print(f"  Seed {s:2d} -> Omega_w: {res['omega_w']:.4f}, AUC: {res['auc']:.4f}, DPD: {res['dpd_hard']:.4f}")

    summary = {}
    for arm in args.arms:
        items = results_by_arm[arm]
        omegas = [r["omega_w"] for r in items if not math.isnan(r["omega_w"])]
        aucs = [r["auc"] for r in items if not math.isnan(r["auc"])]
        dpds = [r["dpd_hard"] for r in items if not math.isnan(r["dpd_hard"])]
        eods = [r["eod"] for r in items if not math.isnan(r["eod"])]

        summary[arm] = {
            "omega_w_mean": float(np.mean(omegas)) if omegas else None,
            "omega_w_std": float(np.std(omegas, ddof=1)) if len(omegas) > 1 else 0.0,
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "dpd_mean": float(np.mean(dpds)) if dpds else None,
            "dpd_std": float(np.std(dpds, ddof=1)) if len(dpds) > 1 else 0.0,
            "eod_mean": float(np.mean(eods)) if eods else None,
            "eod_std": float(np.std(eods, ddof=1)) if len(eods) > 1 else 0.0,
        }

    manifest = build_manifest(dataset=args.dataset, num_seeds=len(seeds), rounds=rounds)
    manifest["device"] = actual_device
    manifest["task"] = "T14_fltrust_ema_benchmark"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"manifest": manifest, "summary": summary, "raw_runs": results_by_arm}, f, indent=2)

    print(f"\n[OK] Results written to {args.output}")


if __name__ == "__main__":
    main()
