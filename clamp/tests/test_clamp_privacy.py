"""Tiny-data checks for the CLAMP privacy experiments (synthetic data, 2 rounds)."""
import math

import pytest

from experiments.clamp_privacy import (_train, attack_balanced_accuracy, dp_bound,
                                       probe_auc, release_inputs, sigma_for_eps)


@pytest.fixture(scope="module")
def trained():
    cfg, trainer = _train("clamp", "synthetic", 0, rounds=2)
    return cfg, trainer


def test_attack_is_near_perfect_without_noise(trained):
    _, trainer = trained
    inputs = release_inputs(trainer)
    res = attack_balanced_accuracy(inputs, sigma=0.0, releases=1, n_targets=400,
                                   trials=40, seed=0)
    assert res["mc"] > 0.999
    assert res["analytic"] == 1.0


def test_attack_is_near_chance_at_small_epsilon(trained):
    cfg, trainer = trained
    inputs = release_inputs(trainer)
    sig = sigma_for_eps(0.05, R=2, delta=cfg.dp_delta)
    res = attack_balanced_accuracy(inputs, sigma=sig, releases=1, n_targets=400,
                                   trials=40, seed=1)
    assert abs(res["mc"] - 0.5) < 0.05
    assert res["mc"] <= dp_bound(0.05, cfg.dp_delta) + 0.03


def test_attack_respects_the_dp_bound_for_composed_releases(trained):
    cfg, trainer = trained
    inputs = release_inputs(trainer)
    R = 2
    sig = sigma_for_eps(1.0, R=R, delta=cfg.dp_delta)
    res = attack_balanced_accuracy(inputs, sigma=sig, releases=R, n_targets=400,
                                   trials=40, seed=2)
    assert res["mc"] <= dp_bound(1.0, cfg.dp_delta) + 0.03


def test_monte_carlo_matches_the_closed_form(trained):
    cfg, trainer = trained
    inputs = release_inputs(trainer)
    sig = sigma_for_eps(8.0, R=2, delta=cfg.dp_delta)
    res = attack_balanced_accuracy(inputs, sigma=sig, releases=1, n_targets=400,
                                   trials=200, seed=3)
    assert abs(res["mc"] - res["analytic"]) < 0.02


def test_probe_auc_is_a_valid_score(trained):
    _, trainer = trained
    auc = probe_auc(trainer, seed=0)
    assert math.isnan(auc) or 0.0 <= auc <= 1.0


def test_dp_bound_values():
    assert dp_bound(math.inf, 1e-5) == 1.0
    assert abs(dp_bound(0.0, 0.0) - 0.5) < 1e-12
