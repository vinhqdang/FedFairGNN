"""Regression test for FSER beta_init parameter wiring and gradient flow.

Guards against B-01 bug where beta_init in ExperimentConfig was ignored and
hardcoded to 0.5.
"""
import pytest
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models import build_model
from src.models.gnn import FSERLayer, TrustFedGNN


def test_fser_layer_beta_init():
    """Verify FSERLayer initializes beta correctly and preserves it across resets."""
    for b in [0.0, 0.1, 0.3, 0.8, 2.5]:
        layer = FSERLayer(in_channels=16, out_channels=4, heads=2, beta_init=b)
        assert layer.beta.item() == pytest.approx(b, rel=1e-5), f"Expected beta={b}, got {layer.beta.item()}"
        
        # Test reset_parameters preserves configured beta_init
        layer.reset_parameters()
        assert layer.beta.item() == pytest.approx(b, rel=1e-5), f"Expected beta={b} after reset, got {layer.beta.item()}"


def test_build_model_trustfedgnn_beta_init():
    """Verify build_model forwards config.beta_init to all FSER layers."""
    for b in [0.05, 0.25, 0.75, 1.5]:
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=b, num_layers=3, heads=2)
        model = build_model("trustfedgnn", in_channels=10, config=cfg)
        assert isinstance(model, TrustFedGNN)
        assert len(model.layers) == 3
        for i, layer in enumerate(model.layers):
            assert layer.beta.item() == pytest.approx(b, rel=1e-5), (
                f"Layer {i} beta expected {b}, got {layer.beta.item()}"
            )


def test_fser_beta_receives_gradient():
    """Verify that self.beta receives non-zero gradients across all FSER modes."""
    from src.data.datasets import load_synthetic
    data = load_synthetic(seed=42, num_nodes=50)
    for mode in ["sub", "add", "same_penalize"]:
        torch.manual_seed(42)
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=0.5, fser_mode=mode, num_layers=1, heads=2, dropout=0.0)
        model = build_model("trustfedgnn", in_channels=data.x.shape[1], config=cfg)
        model.train()
        
        out = model(data.x, data.edge_index, sensitive_attr=data.sensitive_attr)
        loss = out[data.train_mask].sum()
        loss.backward()
        
        beta_grad = model.layers[0].beta.grad
        assert beta_grad is not None, f"Mode '{mode}': beta.grad should not be None"
        assert beta_grad.abs().item() > 1e-6, f"Mode '{mode}': beta.grad should be non-zero, got {beta_grad.item()}"




def test_fser_modes():
    """Verify FSERLayer supports sub, add, and same_penalize modes."""
    for mode in ["sub", "add", "same_penalize"]:
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=0.5, fser_mode=mode)
        model = build_model("trustfedgnn", in_channels=8, config=cfg)
        assert model.fser_mode == mode
        assert model.layers[0].fser_mode == mode


def test_fser_mode_defaults_agree():
    """The default fser_mode must be the canonical 'sub' at EVERY construction site.

    A direct ``TrustFedGNN(...)`` call bypasses build_model/ExperimentConfig, so a
    divergent constructor default would silently train a different method with no
    error -- the layer, the backbone and the canonical config must all say 'sub'.
    """
    assert FSERLayer(in_channels=8, out_channels=4, heads=2).fser_mode == "sub"
    backbone = TrustFedGNN(in_channels=8, hidden_channels=8, heads=2, num_layers=1)
    assert backbone.fser_mode == "sub"
    assert backbone.layers[0].fser_mode == "sub"
    assert ExperimentConfig().fser_mode == "sub"
    assert ExperimentConfig.canonical().fser_mode == "sub"


def test_canonical_matches_manifest():
    """Verify ExperimentConfig.canonical() initializes with the frozen canonical hyperparameters."""
    cfg = ExperimentConfig.canonical(seed=42)
    assert cfg.fser_mode == "sub", f"Expected sub, got {cfg.fser_mode}"
    assert cfg.dp_enabled is True
    assert cfg.dp_mode == "ftgd"
    assert cfg.model == "trustfedgnn"
    assert cfg.aggregator == "fu_shapley"
    assert cfg.fu_alpha == pytest.approx(0.1)
    assert cfg.fu_ema_beta == pytest.approx(0.9)

