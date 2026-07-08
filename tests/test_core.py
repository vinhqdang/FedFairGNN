"""Fast, offline unit tests (synthetic data only -- no downloads).

Run: pytest -q
"""
import math

import numpy as np
import torch

from src.config import ExperimentConfig, set_seed
from src.data.datasets import load_synthetic
from src.data.partition import partition_graph, partition_stats
from src.models import build_model
from src.utils import metrics as M
from src.trust import privacy
from src.trust.trust_score import trust_score as compute_trust
from src.federated import aggregate


# ----------------------------- metrics ----------------------------- #
def test_perfect_and_random_auc():
    y = np.array([0, 0, 1, 1])
    assert M.auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert abs(M.auc_roc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_dpd_zero_when_equal():
    s = np.array([0, 0, 1, 1])
    assert M.demographic_parity_difference(np.array([0.5, 0.5, 0.5, 0.5]), s) == 0.0
    assert abs(M.demographic_parity_difference(np.array([1.0, 1.0, 0.0, 0.0]), s) - 1.0) < 1e-9


def test_metrics_nan_safe():
    y = np.array([0, 1, 0, 1])
    score = np.array([np.nan, np.inf, 0.2, 0.8])
    out = M.all_metrics(y, score, np.array([0, 1, 0, 1]))
    assert all(np.isfinite(v) for v in out.values())


def test_equalized_odds_bounds():
    y = np.array([1, 1, 0, 0]); s = np.array([0, 1, 0, 1])
    v = M.equalized_odds(y, np.array([0.9, 0.1, 0.1, 0.9]), s)
    assert 0.0 <= v <= 1.0


# ----------------------------- privacy ----------------------------- #
def test_epsilon_monotone_in_noise():
    e_lo = privacy.compute_epsilon(1.0, 50, 1e-5)
    e_hi = privacy.compute_epsilon(5.0, 50, 1e-5)
    assert e_hi < e_lo            # more noise -> smaller epsilon
    assert privacy.compute_epsilon(2.0, 100, 1e-5) > privacy.compute_epsilon(2.0, 10, 1e-5)


def test_calibration_hits_target():
    z = privacy.calibrate_noise_multiplier(4.0, 60, 1e-5)
    assert abs(privacy.compute_epsilon(z, 60, 1e-5) - 4.0) < 0.2


# --------------------------- aggregation --------------------------- #
def test_fedavg_is_weighted_mean():
    ups = [torch.ones(5), torch.zeros(5)]
    meta = [{"n": 3}, {"n": 1}]
    agg, _ = aggregate("fedavg", ups, meta)
    assert torch.allclose(agg, torch.full((5,), 0.75), atol=1e-6)


def test_krum_rejects_outlier():
    torch.manual_seed(0)
    honest = [torch.randn(8) * 0.01 for _ in range(5)]
    outlier = torch.ones(8) * 100.0
    ups = honest + [outlier]
    meta = [{"n": 1}] * 6
    agg, info = aggregate("krum", ups, meta, krum_f=1)
    assert info["selected"] != 5        # never selects the outlier
    assert agg.abs().max() < 1.0


def test_median_robust_to_outlier():
    ups = [torch.zeros(4), torch.zeros(4), torch.full((4,), 1e6)]
    agg, _ = aggregate("median", ups, [{"n": 1}] * 3)
    assert torch.allclose(agg, torch.zeros(4))


def test_bfwa_weights_simplex():
    from src.federated.aggregation import bfwa_weights
    w = bfwa_weights(torch.tensor([0.9, 0.6, 0.7]), torch.tensor([0.2, 0.01, 0.1]), tau=0.05)
    assert abs(float(w.sum()) - 1.0) < 1e-5 and (w >= 0).all()


# ------------------------------ trust ------------------------------ #
def test_trust_score_bounds():
    m = {"auc": 0.9, "dpd": 0.01, "eod": 0.02}
    t = compute_trust(m, epsilon=1.0, ece=0.02)
    assert 0.0 <= t <= 1.0


def test_trust_geomean_penalises_zero_axis():
    good = {"auc": 0.9, "dpd": 0.0, "eod": 0.0}
    unfair = {"auc": 0.9, "dpd": 0.5, "eod": 0.5}
    assert compute_trust(good, p=0.0) > compute_trust(unfair, p=0.0)


# ------------------------- data & partition ------------------------ #
def test_synthetic_shapes_and_bias():
    d = load_synthetic(seed=0, num_nodes=400, d=16)
    assert d.x.shape == (400, 16)
    assert d.y.shape[0] == 400 and set(d.y.unique().tolist()) <= {0.0, 1.0}
    assert d.sensitive_attr.shape[0] == 400


def test_partition_covers_all_nodes_and_both_classes():
    set_seed(0)
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    parts = partition_graph(d, 4, "dirichlet", 0.5, seed=0)
    assert sum(p.num_nodes for p in parts) == 600
    for p in parts:                       # floor guarantees both classes present
        assert len(p.y.unique()) == 2


# ------------------------------ models ----------------------------- #
def test_models_forward_probabilities():
    d = load_synthetic(seed=0, num_nodes=200, d=8)
    cfg = ExperimentConfig(hidden_channels=16, num_layers=2, heads=2)
    for name in ["gcn", "gat", "fedfairgnn", "fairgnn", "fairsin"]:
        model = build_model(name, 8, cfg)
        out = model(d.x, d.edge_index, d.sensitive_attr)
        assert out.shape[0] == 200
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
