"""Invariants for how ``dp_mode`` and ``dp_enabled`` interact on the client.

``dp_mode`` names the local *training algorithm*; ``dp_enabled`` says whether the
privacy *mechanism* (clip + Gaussian noise) is live. Conflating the two is how a
config ends up clipping gradients to ``dp_clip`` while adding zero noise -- a
variant that is neither the published private method nor plain SGD, and that is
silently reportable as "DP" because the mode name still says so.

These tests pin the contract:
  * ``dp_active`` is the single switch every clip/noise site gates on;
  * FTGD survives ``dp_enabled=False`` at sigma=0 (the `fedfairgnn-nodp` arm);
  * a privacy *mechanism* asked to run without privacy warns loudly and does not
    clip.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ExperimentConfig
from src.data.datasets import load_synthetic
from src.federated.client import Client


def _client(**overrides):
    data = load_synthetic(seed=0, num_nodes=60)
    cfg = ExperimentConfig(**overrides)
    return Client(0, data, cfg)


def test_dp_disabled_means_no_noise_anywhere():
    """dp_enabled=False must leave sigma at exactly zero for every mode."""
    for mode in ["auto", "none", "ftgd", "gradient", "puffle"]:
        with pytest.warns(RuntimeWarning) if mode in ("gradient", "puffle") \
                else _noop_ctx():
            c = _client(model="gcn", dp_enabled=False, dp_mode=mode)
        assert c.dp_active is False, f"mode={mode}: dp_active must be False"
        assert c.dp_sigma == 0.0, f"mode={mode}: dp_sigma must be 0"
        assert c.noise_multiplier == 0.0, f"mode={mode}: noise_multiplier must be 0"


def test_ftgd_survives_dp_disabled():
    """`fedfairgnn-nodp` = FTGD gradient geometry at sigma=0, no warning.

    The arm must differ from `fedfairgnn` by dp_enabled alone, so the mode has to
    stay 'ftgd' rather than collapsing to 'none'.
    """
    c = _client(model="trustfedgnn", dp_enabled=False, dp_mode="ftgd")
    assert c.dp_mode == "ftgd"
    assert c.dp_active is False


def test_dp_mechanism_without_dp_warns():
    """'gradient'/'puffle' ARE the privacy mechanism: refuse to run it silently."""
    for mode in ["gradient", "puffle"]:
        with pytest.warns(RuntimeWarning, match="not differentially private"):
            c = _client(model="gcn", dp_enabled=False, dp_mode=mode)
        assert c.dp_active is False


def test_privatise_grads_is_a_noop_when_dp_inactive():
    """No clipping without noise: gradients must pass through byte-identically."""
    with pytest.warns(RuntimeWarning):
        c = _client(model="gcn", dp_enabled=False, dp_mode="gradient", dp_clip=1e-6)
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)      # way past any sane clip norm
    before = [p.grad.clone() for p in c.model.parameters()]
    c._privatise_grads()
    for b, p in zip(before, c.model.parameters()):
        assert torch.equal(b, p.grad), "dp_active=False must not touch gradients"


def test_privatise_grads_clips_to_dp_clip_when_active():
    """Clipping bounds ||g|| by dp_clip. Isolated from the noise (sigma forced 0)
    so the bound is exact -- with noise live the perturbation legitimately
    dominates and no norm bound on the released vector holds."""
    c = _client(model="gcn", dp_enabled=True, dp_mode="gradient",
                dp_clip=1.0, dp_epsilon=8.0, dp_delta=1e-5)
    assert c.dp_active is True and c.dp_sigma > 0
    c.dp_sigma = 0.0                     # isolate the clip
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)
    c._privatise_grads()
    flat = torch.cat([p.grad.flatten() for p in c.model.parameters()])
    assert flat.norm(2).item() == pytest.approx(1.0, rel=1e-5), \
        "clipped gradient norm must equal dp_clip"


def test_privatise_grads_actually_adds_noise_when_active():
    """With the mechanism live the released gradient must be perturbed."""
    c = _client(model="gcn", dp_enabled=True, dp_mode="gradient",
                dp_clip=1.0, dp_epsilon=8.0, dp_delta=1e-5)
    for p in c.model.parameters():
        p.grad = torch.full_like(p, 100.0)
    c._privatise_grads()
    flat = torch.cat([p.grad.flatten() for p in c.model.parameters()])
    assert torch.isfinite(flat).all()
    assert flat.norm(2).item() > 1.0, "noise must move the vector off the clip ball"


def test_auto_mode_still_resolves_as_before():
    """dp_mode='auto' keeps its old meaning: ftgd for fair backbones, else gradient."""
    fair = _client(model="trustfedgnn", dp_enabled=True, dp_mode="auto")
    assert fair.dp_mode == "ftgd"
    plain = _client(model="gcn", dp_enabled=True, dp_mode="auto", local_fairness=False)
    assert plain.dp_mode == "gradient"
    off = _client(model="gcn", dp_enabled=False, dp_mode="auto")
    assert off.dp_mode == "none"


class _noop_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
