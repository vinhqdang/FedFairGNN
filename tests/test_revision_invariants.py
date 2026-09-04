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



# --------------------------------------------------------------------------- #
# BFWA: the fairness budget tau must actually bind, and the dual multiplier
# must survive across communication rounds.
#
# Both were broken. `gamma = 2/(t+2)` evaluated at t=0 is 1.0, so the first
# Frank-Wolfe step discarded the uniform iterate and jumped onto the vertex
# argmin(-perf + mu*dpd); with mu=0 and a gradient that does not depend on w
# (the objective is linear), every later step re-selected that same vertex.
# `mu` was also a local restarted at 0 on every call, so it never reached the
# value at which the vertex flips. Net effect: sweeping tau over
# {1e6, 0.10, 0.05, 0.02, 0.0} at the shipped fw_iters=20 returned the
# IDENTICAL weight vector every time -- tau had no effect anywhere in the repo.
# --------------------------------------------------------------------------- #
import torch

from src.federated.aggregation import aggregate, bfwa_weights

# A client set with a real utility/fairness trade-off: the most accurate client
# (0) is also the most unfair, the second-most accurate (2) is middling, and the
# fairest clients (1, 3) are the least accurate.
_PERF = torch.tensor([0.90, 0.75, 0.85, 0.70, 0.80])
_DPD = torch.tensor([0.20, 0.02, 0.15, 0.01, 0.10])
_TAUS = [1e6, 0.10, 0.05, 0.02, 0.0]


def test_bfwa_tau_actually_changes_the_weights():
    """Different tau -> different weights at the shipped iteration budget."""
    ws = [bfwa_weights(_PERF, _DPD, tau, iters=20, dual_step=0.1) for tau in _TAUS]
    for i in range(len(ws) - 1):
        assert not torch.allclose(ws[i], ws[i + 1], atol=1e-6), (
            f"tau={_TAUS[i]} and tau={_TAUS[i+1]} gave identical weights "
            f"{ws[i].tolist()} -- the fairness budget is not binding")
    # ... and tightening the budget must move the achieved disparity DOWN.
    gaps = [float(w.dot(_DPD)) for w in ws]
    assert all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1)), \
        f"weighted DPD must be non-increasing as tau tightens, got {gaps}"
    assert gaps[0] > gaps[-1] + 1e-3, \
        f"tau=0 must buy a strictly smaller disparity than tau=1e6, got {gaps}"


def test_bfwa_reports_constraint_residual_and_feasibility():
    """The per-round residual/feasibility the manuscript promises to report."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    for method in ("bfwa", "robust_bfwa"):
        _, info = aggregate(method, updates, meta, tau=0.05, fw_iters=20,
                            dual_step=0.1, krum_f=1, state={})
        for key in ("constraint_residual", "feasible",
                    "constraint_residual_preclamp", "feasible_preclamp"):
            assert key in info, f"{method} must report {key}"
        w = torch.tensor(info["weights"])
        # post-clamp residual is computed on the returned weights
        assert abs(info["constraint_residual"] - (float(w.dot(_DPD)) - 0.05)) < 1e-5
        assert info["feasible"] == (info["constraint_residual"] <= 0.0)
        assert info["feasible_preclamp"] == (info["constraint_residual_preclamp"] <= 0.0)

    # A budget nothing can violate is reported feasible; tau=0 with a strictly
    # positive disparity everywhere cannot be.
    _, loose = aggregate("bfwa", updates, meta, tau=1e6, state={})
    _, tight = aggregate("bfwa", updates, meta, tau=0.0, state={})
    assert loose["feasible"] is True and tight["feasible"] is False


def test_bfwa_dual_persists_across_rounds():
    """mu must accumulate across rounds through `state`, like fedgraphfair_lambda."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    kw = dict(tau=0.02, fw_iters=20, dual_step=0.1)

    state = {}
    persisted = [aggregate("bfwa", updates, meta, state=state, **kw)[1]
                 for _ in range(2)]
    assert "bfwa_mu" in state and state["bfwa_mu"] > 0.0
    assert persisted[1]["bfwa_mu"] > persisted[0]["bfwa_mu"], \
        "the dual multiplier must keep ascending across rounds"

    independent = [aggregate("bfwa", updates, meta, state=None, **kw)[1]
                   for _ in range(2)]
    assert independent[0]["weights"] == independent[1]["weights"], \
        "without state, every round restarts the dual from 0 (sanity)"
    assert persisted[1]["weights"] != independent[1]["weights"], \
        "persisting mu must change round 2's weights -- otherwise it does nothing"
    # Carrying the dual over tightens the constraint, it does not loosen it.
    assert persisted[1]["constraint_residual"] < independent[1]["constraint_residual"]

    # bfwa_persist_dual=False reproduces the old reset-every-round behaviour.
    off_state = {}
    off = [aggregate("bfwa", updates, meta, state=off_state,
                     bfwa_persist_dual=False, **kw)[1] for _ in range(2)]
    assert "bfwa_mu" not in off_state
    assert off[0]["weights"] == off[1]["weights"] == independent[1]["weights"]


def test_bfwa_and_robust_bfwa_keep_separate_duals():
    """The two rules solve different subproblems and must not share mu."""
    updates = [torch.randn(8) for _ in range(5)]
    meta = [{"n": 100, "perf": float(p), "dpd": float(d), "loss": 1.0 - float(p)}
            for p, d in zip(_PERF, _DPD)]
    state = {}
    aggregate("bfwa", updates, meta, tau=0.02, state=state)
    aggregate("robust_bfwa", updates, meta, tau=0.02, krum_f=1, state=state)
    assert {"bfwa_mu", "robust_bfwa_mu"} <= set(state)
    assert state["bfwa_mu"] != state["robust_bfwa_mu"]
