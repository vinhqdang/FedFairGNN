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
