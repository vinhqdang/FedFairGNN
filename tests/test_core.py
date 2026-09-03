"""Fast, offline unit tests (synthetic data only -- no downloads).

Run: pytest -q
"""
import math
import warnings

import numpy as np
import pytest
import torch

from src.config import ExperimentConfig, set_seed
from src.data.datasets import load_synthetic
from src.data.partition import (carve_server_holdout, partition_graph,
                                partition_stats)
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


def test_metrics_report_divergence_instead_of_hiding_it():
    """SPEC 4.0(c). This test previously asserted the opposite -- that every
    metric stays finite on NaN/inf input -- which made a bug into a requirement:
    non-finite scores were mapped onto the constant 0.5, so a diverged model
    scored auc=0.5 *and dpd=0.0*, i.e. chance accuracy and perfect fairness. It
    is a false positive in our own favour, and it is what the sign_flip rows of
    incentive_audit reported for 80% of rounds. A dead model has no defined
    utility or fairness; it must surface as NaN and be excluded from aggregates,
    not averaged in as a zero disparity."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0, 1, 0, 1])

    out = M.all_metrics(y, np.array([np.nan, np.inf, 0.2, 0.8]), s)
    assert out["diverged"] == 1.0
    assert not np.isfinite(out["auc"]), "diverged AUC must be NaN, never 0.5"
    assert not np.isfinite(out["dpd"]), "diverged DPD must be NaN, never 0.0"

    ok = M.all_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), s)
    assert ok["diverged"] == 0.0
    assert all(np.isfinite(v) for v in ok.values()), "healthy runs stay finite"


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
    # hold privacy fixed (eps) so the comparison isolates the fairness axis
    good = {"auc": 0.9, "dpd": 0.0, "eod": 0.0}
    unfair = {"auc": 0.9, "dpd": 0.5, "eod": 0.5}
    assert compute_trust(good, p=0.0, epsilon=1.0) > compute_trust(unfair, p=0.0, epsilon=1.0)


def test_no_privacy_scores_zero_privacy_axis():
    from src.trust.trust_score import sub_scores
    assert sub_scores({"auc": 0.9, "dpd": 0.01, "eod": 0.01})["privacy"] == 0.0
    assert sub_scores({"auc": 0.9, "dpd": 0.01, "eod": 0.01}, epsilon=1.0)["privacy"] > 0.5


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


# --------------------- server holdout (A1 / C16 / I5) --------------------- #
def _val_budget(d):
    """How many validation nodes carve_server_holdout is allowed to take."""
    return int(d.val_mask.sum()) // 2


def test_holdout_honours_request_within_budget():
    """A request the split can afford must be granted exactly -- and quietly."""
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    ask = _val_budget(d) // 2                       # comfortably inside budget
    assert ask > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # any warning fails the test
        holdout, rest = carve_server_holdout(d, ask, seed=0)
    assert holdout is not None
    assert holdout.num_nodes == ask
    assert holdout.granted_size == ask == holdout.requested_size
    assert holdout.truncated is False


def test_holdout_never_truncates_silently():
    """C16. The cap at half the validation split is legitimate; clipping the
    request *without saying so* is not. Before this test, german's 250-node val
    split silently turned fu_holdout_size=250 into 125, so a D7/D8 ablation
    would have reported 'a bigger holdout does not help' while measuring a
    holdout that never grew. Two things must hold: the caller is warned, and
    the granted size is readable from the returned object."""
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    budget = _val_budget(d)
    ask = budget * 2 + 10                           # impossible on purpose

    with pytest.warns(RuntimeWarning, match="server-holdout"):
        holdout, rest = carve_server_holdout(d, ask, seed=0)

    assert holdout is not None
    assert holdout.requested_size == ask
    assert holdout.granted_size < ask               # it WAS clipped
    assert holdout.granted_size == holdout.num_nodes    # and says by how much
    assert holdout.truncated is True


def test_holdout_is_node_disjoint_from_every_client():
    """I5 / A1. The server's yardstick must not be data any client -- least of
    all a Byzantine one -- also holds. Node ids are relabelled by the induced
    subgraph, so identity is carried by the (unique) synthetic feature rows."""
    set_seed(0)
    d = load_synthetic(seed=0, num_nodes=600, d=8)
    holdout, rest = carve_server_holdout(d, _val_budget(d) // 2, seed=0)
    assert holdout is not None
    parts = partition_graph(rest, 4, "dirichlet", 0.5, seed=0)

    # (a) the partition of `rest` accounts for every non-holdout node
    assert holdout.num_nodes + sum(p.num_nodes for p in parts) == d.num_nodes

    # (b) no feature row is present on both sides
    def rows(t):
        return {tuple(r) for r in t.tolist()}
    server_rows = rows(holdout.x)
    assert len(server_rows) == holdout.num_nodes        # rows really are unique
    for i, p in enumerate(parts):
        assert not (server_rows & rows(p.x)), f"client {i} shares holdout nodes"


# ------------------------------ models ----------------------------- #
def test_models_forward_probabilities():
    d = load_synthetic(seed=0, num_nodes=200, d=8)
    cfg = ExperimentConfig(hidden_channels=16, num_layers=2, heads=2)
    for name in ["gcn", "gat", "trustfedgnn", "fairgnn", "fairsin"]:
        model = build_model(name, 8, cfg)
        out = model(d.x, d.edge_index, d.sensitive_attr)
        assert out.shape[0] == 200
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
