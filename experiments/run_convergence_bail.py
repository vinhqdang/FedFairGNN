"""Stage S3 Runner: Training convergence across communication rounds on Bail.

Generates:
  - results/convergence_bail.json (used by experiments/make_figures.py -> Figure 5)

Runs TrustFedGNN (canonical config on Bail) across 10 seeds {42...51} for 20 rounds,
logging full history (AUC, DPD_hard, EOD, etc.) with mean and std per round.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

# Ensure FedFairGNN root is on path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

SEEDS = list(range(42, 52))  # 10 seeds: 42 to 51
ROUNDS = 20
DATASET = "bail"


def run_convergence_bail(out_path: str = "results/convergence_bail.json") -> dict:
    t_start = time.perf_counter()
    print("=" * 80)
    print(f"  RUN CONVERGENCE BAIL (10 SEEDS: {SEEDS})")
    print(f"  Dataset: {DATASET} | Rounds: {ROUNDS} | Model: TrustFedGNN (FTGD)")
    print("=" * 80, flush=True)

    all_seed_histories: List[List[dict]] = []
    final_metrics_list: List[dict] = []

    for i, seed in enumerate(SEEDS, 1):
        t0 = time.perf_counter()
        print(f"[{i:02d}/{len(SEEDS):02d}] Running seed {seed}...", end="", flush=True)
        cfg = ExperimentConfig.canonical(
            dataset=DATASET,
            rounds=ROUNDS,
            local_epochs=1,
            seed=seed,
            dp_enabled=True,
            dp_mode="ftgd",
        )
        trainer = FederatedTrainer(cfg)
        res = trainer.run(verbose=False)
        dur = time.perf_counter() - t0
        print(f" done in {dur:.1f}s | Final AUC: {res['final']['auc']:.4f}, DPD_hard: {res['final']['dpd_hard']:.4f}", flush=True)

        all_seed_histories.append(res["history"])
        final_metrics_list.append(res["final"])

    total_time = time.perf_counter() - t_start

    # Compute mean and std per round across all seeds
    aggregated_history = []
    for r in range(ROUNDS):
        round_idx = r + 1
        aucs = [h[r]["g_auc"] for h in all_seed_histories]
        dpds = [h[r]["g_dpd_hard"] for h in all_seed_histories]
        dpd_softs = [h[r]["g_dpd_soft"] for h in all_seed_histories]
        eods = [h[r].get("g_eod", 0.0) for h in all_seed_histories]
        aps = [h[r].get("g_ap", 0.0) for h in all_seed_histories]
        f1s = [h[r].get("g_f1", 0.0) for h in all_seed_histories]

        aggregated_history.append({
            "round": round_idx,
            "g_auc": float(np.mean(aucs)),
            "g_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "g_dpd": float(np.mean(dpds)),
            "g_dpd_std": float(np.std(dpds, ddof=1)) if len(dpds) > 1 else 0.0,
            "g_dpd_hard": float(np.mean(dpds)),
            "g_dpd_hard_std": float(np.std(dpds, ddof=1)) if len(dpds) > 1 else 0.0,
            "g_dpd_soft": float(np.mean(dpd_softs)),
            "g_dpd_soft_std": float(np.std(dpd_softs, ddof=1)) if len(dpd_softs) > 1 else 0.0,
            "g_eod": float(np.mean(eods)),
            "g_eod_std": float(np.std(eods, ddof=1)) if len(eods) > 1 else 0.0,
            "g_ap": float(np.mean(aps)),
            "g_f1": float(np.mean(f1s)),
        })

    # Summary of final metrics
    final_aucs = [f["auc"] for f in final_metrics_list]
    final_dpds = [f["dpd_hard"] for f in final_metrics_list]
    final_eods = [f["eod"] for f in final_metrics_list]

    final_summary = {
        "auc_mean": float(np.mean(final_aucs)),
        "auc_std": float(np.std(final_aucs, ddof=1)),
        "dpd_hard_mean": float(np.mean(final_dpds)),
        "dpd_hard_std": float(np.std(final_dpds, ddof=1)),
        "eod_mean": float(np.mean(final_eods)),
        "eod_std": float(np.std(final_eods, ddof=1)),
    }

    manifest = build_manifest(extra={
        "experiment": "convergence_bail",
        "dataset": DATASET,
        "rounds": ROUNDS,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "wall_clock_s": total_time,
    })

    payload = {
        "manifest": manifest,
        "final": final_summary,
        "history": aggregated_history,
        "per_seed_final": final_metrics_list,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  CONVERGENCE BAIL COMPLETE in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Final AUC: {final_summary['auc_mean']:.4f} +/- {final_summary['auc_std']:.4f}")
    print(f"  Final DPD: {final_summary['dpd_hard_mean']:.4f} +/- {final_summary['dpd_hard_std']:.4f}")
    print(f"  Saved artifact to: {out_path}")
    print("=" * 80, flush=True)

    return payload


if __name__ == "__main__":
    run_convergence_bail()
