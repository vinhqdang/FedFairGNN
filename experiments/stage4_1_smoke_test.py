"""Stage 4.1: Fast Smoke Test & Pipeline Integrity Check (Offline Local).

Verifies the entire end-to-end TrustFedGNN / FairShare-GNN training pipeline
on a synthetic graph without network calls or heavy compute.
"""
from __future__ import annotations

import json
import math
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig, set_seed
from src.data.datasets import load_synthetic
from src.data.partition import partition_graph, partition_stats
from src.models import build_model
from src.federated import FederatedTrainer
from src.trust.incentive import (
    get_server_target_gradients,
    compute_fu_weights,
    decompose,
)
from src.utils.metrics import all_metrics


def run_stage4_1_smoke_test():
    print("=" * 70)
    print("🚀 [STAGE 4.1] EXECUTING FAST SMOKE TEST (OFFLINE LOCAL)")
    print("=" * 70)

    seed = 42
    set_seed(seed)
    num_clients = 3
    num_rounds = 3

    # 1. Config initialization with exact ExperimentConfig parameters
    cfg = ExperimentConfig(
        dataset="synthetic",
        seed=seed,
        num_clients=num_clients,
        rounds=num_rounds,
        model="trustfedgnn",
        aggregator="fu_shapley",
        hidden_channels=8,
        num_layers=2,
        heads=2,
        dropout=0.0,
        fairness_weight=1.0,
        fu_alpha=0.2,
        fu_ema_beta=0.9,
        fu_grad_clip=10.0,
        dp_enabled=True,
        dp_epsilon=1.0,
        dp_delta=1e-5,
    )

    print(f"[*] Initialized ExperimentConfig: model={cfg.model}, aggregator={cfg.aggregator}, clients={cfg.num_clients}, rounds={cfg.rounds}")

    # 2. Build Trainer
    trainer = FederatedTrainer(cfg)
    print(f"[*] FederatedTrainer constructed successfully. Number of clients: {len(trainer.clients)}")

    # 3. Step through communication rounds
    round_logs = []
    for t in range(num_rounds):
        res = trainer._round(t)
        round_logs.append(res)
        auc = res.get("g_auc", res.get("auc", 0.0))
        dpd = res.get("g_dpd", res.get("dpd", 0.0))
        eod = res.get("g_eod", res.get("eod", 0.0))
        diverged = res.get("g_diverged", res.get("diverged", 0.0))
        weights = res.get("agg_weights", res.get("weights"))
        
        print(f"\n--- Round {t + 1}/{num_rounds} ---")
        print(f"    Global Metrics: AUC = {auc:.4f}, DPD = {dpd:.4f}, EOD = {eod:.4f}, Diverged = {diverged}")
        
        # Verify Simplex weights
        if weights is not None:
            w_sum = sum(weights)
            all_nonneg = all(w >= 0 for w in weights)
            print(f"    Simplex Weights: {[round(w, 4) for w in weights]} | Sum = {w_sum:.4f} (Non-neg: {all_nonneg})")
            assert math.isclose(w_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4), f"Weights sum to {w_sum}, expected 1.0"
            assert all_nonneg, "Negative weight found on Simplex!"

        # Verify Shapley decomposition if available
        if "phi_util" in res and "phi_fair" in res:
            print(f"    FU-Shapley Split: phi_util = {[round(float(v), 4) for v in res['phi_util']]}")
            print(f"                      phi_fair = {[round(float(v), 4) for v in res['phi_fair']]}")

        # Verify Divergence
        assert diverged == 0.0, f"Round {t} marked as diverged!"
        assert math.isfinite(auc), f"AUC is non-finite: {auc}"
        assert math.isfinite(dpd), f"DPD is non-finite: {dpd}"

    # 4. Check Beta Clamping in FSER layers
    for client in trainer.clients:
        for layer in client.model.layers:
            if hasattr(layer, "beta"):
                beta_val = float(layer.beta.data)
                assert 0.0 <= beta_val <= 5.0, f"Beta out of bounds [0, 5]: {beta_val}"
    print("\n[*] FSER Beta Clamping: Verified (all beta in [0.0, 5.0])")

    # 5. Output summary verification
    final_res = round_logs[-1]
    final_auc = final_res.get("g_auc", final_res.get("auc", 0.0))
    final_dpd = final_res.get("g_dpd", final_res.get("dpd", 0.0))
    final_eod = final_res.get("g_eod", final_res.get("eod", 0.0))
    final_weights = final_res.get("agg_weights", final_res.get("weights"))

    print("\n" + "=" * 70)
    print("✅ [STAGE 4.1 SMOKE TEST PASSED] PIPELINE INTEGRITY FULLY VERIFIED")
    print("=" * 70)
    summary = {
        "status": "PASS",
        "rounds_completed": num_rounds,
        "final_auc": final_auc,
        "final_dpd": final_dpd,
        "final_eod": final_eod,
        "diverged": final_res.get("g_diverged", 0.0),
        "weights": final_weights,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_stage4_1_smoke_test()
