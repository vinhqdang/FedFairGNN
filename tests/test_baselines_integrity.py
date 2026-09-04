"""Unit test to ensure 100% of baseline models and aggregators instantiate and execute cleanly."""
import pytest
import torch
from src.config import ExperimentConfig
from src.models import build_model
from src.federated.aggregation import aggregate
from experiments.methods import METHODS, apply_method


def test_all_methods_instantiation_and_forward():
    """Verify that every method defined in METHODS builds and executes a forward pass."""
    x = torch.randn(12, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    s = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    
    for m_name in METHODS:
        cfg = ExperimentConfig.canonical()
        apply_method(cfg, m_name)
        model = build_model(cfg.model, in_channels=16, config=cfg)
        assert model is not None, f"Failed to build model for method {m_name}"
        
        # Test forward pass with sensitive_attr
        out = model(x, edge_index, s)
        assert out.shape == (12,), f"Output shape mismatch for {m_name}: got {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in output for {m_name}"
        assert not torch.isinf(out).any(), f"Inf in output for {m_name}"


def test_all_aggregators():
    """Verify that custom aggregators compute valid simplex weights without NaN."""
    K = 5
    updates = [torch.randn(32) for _ in range(K)]
    meta = [
        {"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2},
        {"n": 120, "perf": 0.75, "dpd": 0.08, "loss": 0.25},
        {"n": 80, "perf": 0.85, "dpd": 0.02, "loss": 0.15},
        {"n": 150, "perf": 0.7, "dpd": 0.12, "loss": 0.3},
        {"n": 90, "perf": 0.9, "dpd": 0.04, "loss": 0.1},
    ]
    
    for agg_name in ["fedavg", "fairfed", "bfwa", "fairgfl", "fedgraphfair", "popets_fairfed", "krum", "multikrum", "median", "trimmed_mean"]:
        state = {}
        agg, info = aggregate(agg_name, updates, meta, state=state)
        assert agg.shape == (32,), f"Agg shape mismatch for {agg_name}"
        assert not torch.isnan(agg).any(), f"NaN in agg for {agg_name}"
        if "weights" in info and info["weights"] is not None:
            w = torch.tensor(info["weights"])
            assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-4), f"Weight sum != 1 for {agg_name}: {w.sum()}"


def test_bfwa_variants_report_the_feasibility_contract():
    """Every BFWA round must expose the constraint residual and feasibility.

    The manuscript states that the per-round constraint residual and the
    empirical feasibility rate are reported; before the Frank-Wolfe fix nothing
    in the repo computed either quantity (and tau did not bind at all).
    """
    K = 5
    torch.manual_seed(7)
    updates = [torch.randn(32) for _ in range(K)]
    perf = [0.90, 0.75, 0.85, 0.70, 0.80]
    dpd = [0.20, 0.02, 0.15, 0.01, 0.10]
    meta = [{"n": 100, "perf": p, "dpd": d, "loss": 1.0 - p}
            for p, d in zip(perf, dpd)]

    for agg_name in ("bfwa", "robust_bfwa"):
        _, info = aggregate(agg_name, updates, meta, tau=0.05, krum_f=1, state={})
        for key in ("constraint_residual", "feasible",
                    "constraint_residual_preclamp", "feasible_preclamp", "tau"):
            assert key in info, f"{agg_name} must report {key}"
        assert isinstance(info["feasible"], bool)
        w = torch.tensor(info["weights"])
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-4)


def test_fu_aggregators_never_pay_a_client_that_submitted_nothing():
    """Null-player axiom, at the aggregator boundary."""
    torch.manual_seed(8)
    P = 32
    g_target = torch.randn(P)
    updates = [-g_target, -g_target, torch.randn(P), torch.zeros(P)]
    meta = [{"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2} for _ in range(4)]

    for agg_name in ("fu_shapley", "robust_fu_shapley"):
        _, info = aggregate(agg_name, updates, meta, state={}, krum_f=1,
                            g_target=g_target, fu_normalize="none")
        w = info["weights"]
        if w is None:
            continue            # a median fallback pays nobody by construction
        assert w[3] == 0.0, f"{agg_name} paid the null client: {w}"
        assert abs(sum(w) - 1.0) < 1e-4
