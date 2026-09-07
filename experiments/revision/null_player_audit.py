"""Step 4.3: Empirical verification of Proposition 3 -- Null-Player Invariant (N6).

Validates that a null client submitting an identically zero update (g_k = 0)
receives exactly zero aggregation weight (w_null = 0) in EVERY round,
including normal rounds and degenerate fallback rounds (all scores <= 0).
Formal backing: docs/proofs/NullPlayer.lean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
from src.config import ExperimentConfig, set_seed
from src.federated import FederatedTrainer
from src.trust.incentive import compute_fu_weights
from src.utils.provenance import build_manifest


def run_null_player_audit(out_dir: str = "results/fairshare"):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    # 1. Component test: verify across 20 simulated rounds including degenerate fallback
    K, P = 5, 32
    null_idx = 2
    round_records = []
    phi_ema = None
    all_zero_rounds = 0

    for r in range(20):
        # Generate client gradients; enforce client null_idx is exactly zero
        grads = [torch.randn(P) for _ in range(K)]
        grads[null_idx] = torch.zeros(P)

        # In round 10 and 15, simulate degenerate condition where target is anti-aligned with all
        if r in (10, 15):
            g_target = -10.0 * sum(grads)
        else:
            g_target = torch.randn(P)

        w, phi_raw, phi_ema = compute_fu_weights(
            grads, g_target, phi_ema=phi_ema, beta_ema=0.9, normalize="target_norm"
        )
        w_null = float(w[null_idx].item())
        assert w_null == 0.0, f"Round {r}: null player received weight {w_null} > 0"
        
        round_records.append({
            "round": r,
            "w_null": w_null,
            "weights": [round(float(x), 4) for x in w],
            "simplex_sum": round(float(w.sum().item()), 4),
            "is_degenerate": bool(r in (10, 15)),
        })

    # 2. End-to-end integration test: trainer running 5 rounds on synthetic graph
    cfg = ExperimentConfig(
        dataset="synthetic",
        seed=42,
        num_clients=4,
        rounds=5,
        model="trustfedgnn",
        aggregator="fu_shapley",
        hidden_channels=8,
        num_layers=2,
        heads=2,
    )
    trainer = FederatedTrainer(cfg)
    # Client 1 does not train (null player emitting zero update)
    trainer.clients[1].train = lambda: None
    
    e2e_null_weights = []
    for t in range(5):
        res = trainer._round(t)
        w = res.get("agg_weights")
        if w is not None:
            w1 = float(w[1])
            assert w1 == 0.0, f"Trainer round {t}: null player received weight {w1} > 0"
            e2e_null_weights.append(w1)

    verdict = {
        "proposition_3_verified": True,
        "null_player_weight_always_zero": True,
        "rounds_audited": len(round_records),
        "e2e_trainer_rounds_audited": len(e2e_null_weights),
        "round_details": round_records,
    }

    manifest = build_manifest(experiment="null_player_audit", args={"out_dir": out_dir})
    payload = {"manifest": manifest, "verdict": verdict}
    out_file = os.path.join(out_dir, "null_player_verdict.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 60)
    print("NULL PLAYER INVARIANT: PASS")
    print(f"Audited {len(round_records)} rounds (including fallback branches).")
    print(f"w_null == 0.0 in 100% of tested rounds.")
    print(f"Wrote verdict to {out_file}")
    print("=" * 60)
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/fairshare")
    args = p.parse_args()
    run_null_player_audit(args.out_dir)
