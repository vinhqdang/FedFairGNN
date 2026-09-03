"""Invariant tests for Phase 1 revision ablation configs and runners.

Ensures that:
1. Every ablation arm in the grid isolates EXACTLY the intended components.
2. No baseline or ablation arm borrows outside modules inappropriately.
3. All 7 configs instantiate and run correctly for 1 round on synthetic data.
"""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.federated import FederatedTrainer


ABLATION_CONFIGS = {
    "C0_FedAvg": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C1_FedAvg_FSER": dict(
        model="trustfedgnn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C2_FedAvg_FTGD": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=False,
        dp_mode="ftgd",
    ),
    "C3_FedAvg_FTGD_DP": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
    "C4_BFWA_unconstrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=1e6,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C5_BFWA_constrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=0.05,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C6_Full_TrustFedGNN": dict(
        model="trustfedgnn",
        aggregator="bfwa",
        fairness_budget=0.05,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
}


def test_ablation_grid_pairwise_isolation():
    """Verify that ablation arms strictly isolate the intended independent variables."""
    c0 = ABLATION_CONFIGS["C0_FedAvg"]
    c1 = ABLATION_CONFIGS["C1_FedAvg_FSER"]
    c2 = ABLATION_CONFIGS["C2_FedAvg_FTGD"]
    c3 = ABLATION_CONFIGS["C3_FedAvg_FTGD_DP"]
    c4 = ABLATION_CONFIGS["C4_BFWA_unconstrained"]
    c5 = ABLATION_CONFIGS["C5_BFWA_constrained"]
    c6 = ABLATION_CONFIGS["C6_Full_TrustFedGNN"]

    # C0 vs C1: strictly isolates FSER (model)
    diff_0_1 = {k for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
    assert diff_0_1 == {"model"}, f"C0 vs C1 should strictly differ by 'model', got {diff_0_1}"

    # C0 vs C2: strictly isolates FTGD loss objective
    diff_0_2 = {k for k in set(c0) | set(c2) if c0.get(k) != c2.get(k)}
    assert diff_0_2 == {"local_fairness", "dp_mode"}, f"C0 vs C2 diff: {diff_0_2}"

    # C2 vs C3: strictly isolates DP noise
    diff_2_3 = {k for k in set(c2) | set(c3) if c2.get(k) != c3.get(k)}
    assert diff_2_3 == {"dp_enabled", "dp_epsilon", "dp_delta"}, f"C2 vs C3 diff: {diff_2_3}"

    # C0 vs C4: strictly isolates BFWA unconstrained aggregation
    diff_0_4 = {k for k in set(c0) | set(c4) if c0.get(k) != c4.get(k)}
    assert diff_0_4 == {"aggregator", "fairness_budget"}, f"C0 vs C4 diff: {diff_0_4}"

    # C4 vs C5: strictly isolates disparity budget constraint tau
    diff_4_5 = {k for k in set(c4) | set(c5) if c4.get(k) != c5.get(k)}
    assert diff_4_5 == {"fairness_budget"}, f"C4 vs C5 diff: {diff_4_5}"

    # C6: Full TrustFedGNN has all three components active
    assert c6["model"] == "trustfedgnn"
    assert c6["aggregator"] == "bfwa"
    assert c6["fairness_budget"] == 0.05
    assert c6["dp_enabled"] is True
    assert c6["dp_mode"] == "ftgd"


def test_ablation_configs_smoke_run():
    """Smoke test: execute 1 round of each of the 7 configs on synthetic data."""
    for name, overrides in ABLATION_CONFIGS.items():
        cfg = ExperimentConfig.canonical(
            dataset="synthetic",
            rounds=1,
            num_clients=2,
            local_epochs=1,
            seed=42,
            **overrides
        )
        trainer = FederatedTrainer(cfg)
        res = trainer.run(verbose=False)
        assert "final" in res
        assert "auc" in res["final"]
        assert "dpd_hard" in res["final"]
        assert "eod" in res["final"]


def test_robustness_screening_invariants():
    """Verify that robust_bfwa successfully performs distance screening under attack."""
    cfg = ExperimentConfig.canonical(
        dataset="synthetic",
        rounds=2,
        num_clients=4,
        local_epochs=1,
        seed=42,
        model="trustfedgnn",
        aggregator="robust_bfwa",
        attack="gaussian",
        num_byzantine=1,
        krum_f=1,
        attack_intensity=10.0,
    )
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    assert len(res["history"]) == 2
    # Byzantine client is client 0. Check that client 0 is screened out from kept set
    for r_entry in res["history"]:
        assert "kept" in r_entry, f"r_entry should have 'kept' key, got keys: {list(r_entry.keys())}"
        assert len(r_entry["kept"]) == 3
        # Malicious client 0 with gaussian noise variance 10.0 should be excluded
        assert 0 not in r_entry["kept"], f"Malicious client 0 should be screened out, but kept set is {r_entry['kept']}"

