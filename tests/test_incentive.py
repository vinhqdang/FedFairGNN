"""Gate-0 unit tests for the FU-Shapley incentive module (FairShare-GNN).

Each test pins one of the area-chair findings F1-F6 so a regression in the sign,
the coordinate layout, or the loss can never pass silently. Offline & fast
(synthetic graph, tiny model). Run: pytest tests/test_incentive.py -q
"""
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models import build_model
from src.data.datasets import load_synthetic
from src.federated.client import flatten_state, _weighted_bce
from src.trust.incentive import (
    get_server_target_gradients, get_server_target_gradients_pooled,
    compute_fu_weights, decompose,
)


def _fixture(seed=0, d=8, n=160):
    torch.manual_seed(seed)
    cfg = ExperimentConfig(hidden_channels=8, num_layers=2, heads=2, dropout=0.0)
    data = load_synthetic(num_nodes=n, d=d, seed=seed)
    model = build_model("trustfedgnn", d, cfg)
    return model, data


def _flat_grad_ref(model):
    """Independent replica of the state_dict-ordered flatten (for parity check)."""
    gm = {n: p.grad for n, p in model.named_parameters()}
    parts = []
    for k, v in model.state_dict().items():
        g = gm.get(k)
        parts.append((g.detach().flatten() if g is not None
                      else torch.zeros(v.numel())).to(torch.float32))
    return torch.cat(parts)


# --------------------------------------------------------------------------- #
# F2: g_target lines up coordinate-for-coordinate with flatten_state(g_k)
# --------------------------------------------------------------------------- #
def test_dim_match():
    model, data = _fixture()
    g_target, g_task, g_fair = get_server_target_gradients(model, data, alpha=0.1)
    P = flatten_state(model.state_dict()).numel()
    assert g_target.shape == (P,)
    assert g_task.shape == (P,) and g_fair.shape == (P,)
    assert g_target.dtype == torch.float32


# --------------------------------------------------------------------------- #
# F3: task target uses WEIGHTED BCE on val_mask (not plain BCE, not train_mask)
# --------------------------------------------------------------------------- #
def test_weighted_bce_parity():
    model, data = _fixture()
    g_target, g_task, g_fair = get_server_target_gradients(model, data, alpha=0.1)

    m = data.val_mask
    # reference 1: weighted BCE (what the function must use)
    model.eval()
    model.zero_grad(set_to_none=True)
    pred = model(data.x, data.edge_index, data.sensitive_attr)[m]
    _weighted_bce(pred, data.y[m].float()).backward()
    ref_weighted = _flat_grad_ref(model)

    # reference 2: plain BCE (what the buggy proposal used) -- must DIFFER
    model.zero_grad(set_to_none=True)
    pred = model(data.x, data.edge_index, data.sensitive_attr)[m]
    F.binary_cross_entropy(pred, data.y[m].float()).backward()
    ref_plain = _flat_grad_ref(model)

    assert torch.allclose(g_task, ref_weighted, atol=1e-6), "task grad must equal weighted-BCE grad"
    assert not torch.allclose(ref_weighted, ref_plain, atol=1e-4), \
        "weighted and plain BCE should differ on class-imbalanced data (sanity of the test)"


# --------------------------------------------------------------------------- #
# F1: fairness credit sign is PLUS -- a disparity-reducing update is rewarded
# --------------------------------------------------------------------------- #
def test_sign_fair():
    model, data = _fixture()
    _, g_task, g_fair = get_server_target_gradients(model, data, alpha=0.5)

    # A client that DESCENDS fairness loss has pseudo-grad aligned with the
    # ascent gradient g_fair; a bias-increasing client is anti-aligned.
    good = 1.0 * g_fair
    bad = -1.0 * g_fair
    phi_util, phi_fair = decompose([good, bad], g_task, g_fair, alpha=0.5)

    assert phi_fair[0] > 0 > phi_fair[1], "fair-improving client must get positive fairness credit"
    assert phi_fair[0] > phi_fair[1]

    # and the weight must favour the client aligned with the overall target
    g_target = g_task + 0.5 * g_fair
    w, phi_raw, _ = compute_fu_weights([g_target, -g_target], g_target,
                                       normalize="none")
    assert w[0] > w[1]
    assert torch.isclose(w[1], torch.tensor(0.0))


# --------------------------------------------------------------------------- #
# ReLU-gate: phi<0 -> w=0; all-negative -> uniform FedAvg fallback
# --------------------------------------------------------------------------- #
def test_relu_gate():
    torch.manual_seed(1)
    g_target = torch.randn(64)

    # one aligned (+), one anti-aligned (-)
    w, _, _ = compute_fu_weights([g_target, -g_target], g_target, normalize="none")
    assert torch.isclose(w[0], torch.tensor(1.0)) and torch.isclose(w[1], torch.tensor(0.0))

    # all anti-aligned -> total ReLU mass 0 -> uniform fallback
    w2, _, _ = compute_fu_weights([-g_target, -g_target, -g_target], g_target,
                                  normalize="none")
    assert torch.allclose(w2, torch.full((3,), 1.0 / 3.0))
    assert torch.isclose(w2.sum(), torch.tensor(1.0))


# --------------------------------------------------------------------------- #
# F6/F9: EMA smoothing reduces round-to-round weight variance under noise
# --------------------------------------------------------------------------- #
def test_ema_smoothness():
    torch.manual_seed(2)
    P = 50
    base = torch.randn(P)
    g0 = base.clone()               # client aligned with the target
    g1 = torch.randn(P)             # noisy competitor

    def trajectory(beta):
        phi_ema = None
        ws = []
        gen = torch.Generator().manual_seed(7)
        for _ in range(40):
            g_target = base + 0.8 * torch.randn(P, generator=gen)   # noisy target
            w, _, phi_ema = compute_fu_weights([g0, g1], g_target,
                                               phi_ema=phi_ema, beta_ema=beta,
                                               normalize="none")
            ws.append(w[0].item())
        return torch.tensor(ws)

    var_ema = trajectory(0.9).var()
    var_raw = trajectory(0.0).var()          # beta=0 -> no smoothing
    assert var_ema < var_raw, f"EMA should smooth weights (ema={var_ema:.4f} raw={var_raw:.4f})"


# =========================================================================== #
# P0-GAP: the five tests above all exercise get_server_target_gradients, which
# exists ONLY in this file -- production calls get_server_target_gradients_POOLED
# (trainer.py:221). The two are separate code paths: the non-pooled one runs one
# forward over a single graph and reads `.grad`; the pooled one concatenates a
# forward per client and uses torch.autograd.grad. Everything below pins F1/F2/F3
# on the path that actually ships.
# =========================================================================== #
def _pooled(model, clients, alpha=0.1, **kw):
    return get_server_target_gradients_pooled(model, clients, "cpu", alpha, **kw)


def _pooled_grad_ref(model, clients, loss_fn):
    """Independent replica of the pooled forward + state_dict-ordered flatten."""
    model.eval()
    preds, ys, ss = [], [], []
    for d in clients:
        m = d.val_mask
        preds.append(model(d.x, d.edge_index, d.sensitive_attr)[m])
        ys.append(d.y[m]); ss.append(d.sensitive_attr[m])
    pred, y, s = torch.cat(preds), torch.cat(ys), torch.cat(ss)
    model.zero_grad(set_to_none=True)
    loss_fn(pred, y, s).backward()
    return _flat_grad_ref(model)


# --------------------------------------------------------------------------- #
# F2 on the production path: pooled g_target lines up with flatten_state(g_k)
# --------------------------------------------------------------------------- #
def test_dim_match_pooled():
    model, data = _fixture()
    g_target, g_task, g_fair = _pooled(model, [data, data])
    P = flatten_state(model.state_dict()).numel()
    assert g_target.shape == (P,)
    assert g_task.shape == (P,) and g_fair.shape == (P,)
    assert g_target.dtype == torch.float32


# --------------------------------------------------------------------------- #
# F3 on the production path: WEIGHTED BCE over the pooled val nodes
# --------------------------------------------------------------------------- #
def test_bce_parity_pooled():
    model, data = _fixture()
    clients = [data, data]
    _, g_task, _ = _pooled(model, clients)

    ref_weighted = _pooled_grad_ref(
        model, clients, lambda p, y, s: _weighted_bce(p, y.float()))
    ref_plain = _pooled_grad_ref(
        model, clients, lambda p, y, s: F.binary_cross_entropy(p, y.float()))

    assert torch.allclose(g_task, ref_weighted, atol=1e-6), \
        "pooled task grad must equal the weighted-BCE grad over pooled val nodes"
    assert not torch.allclose(ref_weighted, ref_plain, atol=1e-4), \
        "weighted and plain BCE should differ (sanity of the test itself)"


# --------------------------------------------------------------------------- #
# F1 on the production path: fairness credit sign is PLUS
# --------------------------------------------------------------------------- #
def test_sign_fair_pooled():
    model, data = _fixture()
    _, g_task, g_fair = _pooled(model, [data, data], alpha=0.5)

    phi_util, phi_fair = decompose([1.0 * g_fair, -1.0 * g_fair],
                                   g_task, g_fair, alpha=0.5)
    assert phi_fair[0] > 0 > phi_fair[1], \
        "fair-improving client must get positive fairness credit on the pooled path"

    g_target = g_task + 0.5 * g_fair
    w, _, _ = compute_fu_weights([g_target, -g_target], g_target, normalize="none")
    assert w[0] > w[1] and torch.isclose(w[1], torch.tensor(0.0))


# --------------------------------------------------------------------------- #
# The lock: with ONE client holding the whole graph, the two code paths must
# agree. This is what makes the four tests above transfer to the non-pooled
# helper (and vice versa) instead of the two drifting apart silently.
# --------------------------------------------------------------------------- #
def test_pooled_matches_nonpooled():
    model, data = _fixture()
    a = get_server_target_gradients(model, data, alpha=0.3)
    b = _pooled(model, [data], alpha=0.3)
    for name, x, y in zip(("g_target", "g_task", "g_fair"), a, b):
        assert torch.allclose(x, y, atol=1e-5), f"{name} differs between code paths"


# --------------------------------------------------------------------------- #
# F15: phi decomposes additively under `dot` -- and NOT under `cosine`.
# The explainability claim (phi = phi_util + phi_fair) is only valid for `dot`;
# choosing cosine in the D11 sweep would silently break it.
# --------------------------------------------------------------------------- #
def test_decompose_additivity():
    torch.manual_seed(3)
    P, alpha = 64, 0.4
    g_task, g_fair = torch.randn(P), torch.randn(P)
    g_target = g_task + alpha * g_fair
    grads = [torch.randn(P) for _ in range(4)]

    _, phi_dot, _ = compute_fu_weights(grads, g_target, normalize="none", score="dot")
    u, f = decompose(grads, g_task, g_fair, alpha=alpha, score="dot")
    assert torch.allclose(phi_dot, u + f, atol=1e-5), \
        "dot-score phi must equal phi_util + phi_fair (explainability claim)"

    _, phi_cos, _ = compute_fu_weights(grads, g_target, normalize="none", score="cosine")
    u_c, f_c = decompose(grads, g_task, g_fair, alpha=alpha, score="cosine")
    assert not torch.allclose(phi_cos, u_c + f_c, atol=1e-3), \
        "cosine must NOT be additive -- if this ever passes, the F15 warning is moot"


# --------------------------------------------------------------------------- #
# SPEC 4.0 / proposal 1.3.1 -- the finiteness contract (tests 11-12)
# --------------------------------------------------------------------------- #
def test_nan_phi_holds_ema_and_sets_fallback_flag():
    """A non-finite g_k must not poison the EMA, and must be labelled.

    Pre-4.0 the recursion was `0.9*ema + 0.1*phi`, for which NaN is an absorbing
    state: one exploding round (routine under sign_flip) left phi_ema NaN for
    the rest of training, and since `nan > 0` is False the rule then quietly
    aggregated as FedAvg -- handing the attacker a full 1/K share while
    reporting no fallback at all. Three properties pin that shut."""
    P = 32
    g_target = torch.randn(P)
    good = g_target.clone()

    # --- round 1: everything finite, establishes a healthy EMA ---
    info1 = {}
    _, _, ema1 = compute_fu_weights([good, torch.randn(P)], g_target,
                                    normalize="none", info=info1)
    assert info1["fu_status"] == "ok"
    assert info1["phi_nan_frac"] == 0.0
    assert torch.isfinite(ema1).all()

    # --- round 2: client 1's update explodes ---
    blown = torch.full((P,), float("inf"))
    info2 = {}
    _, _, ema2 = compute_fu_weights([good, blown], g_target, phi_ema=ema1,
                                    normalize="none", info=info2)
    assert torch.isfinite(ema2).all(), "NaN must never enter the EMA state"
    assert info2["phi_nan_frac"] == 0.5, "the broken client must be counted"
    # D2: the poisoned coordinate holds its previous value; the healthy one moves.
    assert torch.isclose(ema2[1], ema1[1]), "non-finite score must hold prior EMA"

    # --- round 3: client 1 recovers -> the EMA must recover with it ---
    info3 = {}
    _, _, ema3 = compute_fu_weights([good, torch.randn(P)], g_target,
                                    phi_ema=ema2, normalize="none", info=info3)
    assert torch.isfinite(ema3).all()
    assert info3["fu_status"] == "ok", "mechanism must resume after a bad round"

    # --- D3: 'measurement broke' is labelled apart from 'all contributions <=0' ---
    info_nan = {}
    compute_fu_weights([blown, blown], g_target, normalize="none", info=info_nan)
    assert info_nan["fu_status"] == "degenerate_nan"
    info_neg = {}
    compute_fu_weights([-g_target, -g_target], g_target, normalize="none",
                       info=info_neg)
    assert info_neg["fu_status"] == "degenerate_nonpos"


def test_grad_clip_bounds_influence_without_touching_the_attack():
    """SPEC 4.0(b): cap ||g_k|| defensively rather than weakening the adversary."""
    P = 32
    g_target = torch.randn(P)
    huge = 1000.0 * g_target / g_target.norm()

    info = {}
    _, phi_clipped, _ = compute_fu_weights([g_target, huge], g_target,
                                           normalize="none", grad_clip=10.0,
                                           info=info)
    _, phi_raw, _ = compute_fu_weights([g_target, huge], g_target,
                                       normalize="none")
    assert info["n_clipped"] == 1
    assert phi_clipped[1] < phi_raw[1], "clipping must reduce the outlier's score"
    assert torch.isclose(phi_clipped[0], phi_raw[0]), "in-norm client untouched"


def test_grad_norms_are_logged_so_the_clip_threshold_is_falsifiable():
    """``n_clipped == 0`` is ambiguous, and that ambiguity hid a dead defence.

    A correctly-sized clip on a clean round reports ``n_clipped = 0``; so does a
    threshold expressed in the wrong units, which is what ``fu_grad_clip = 10.0``
    turned out to be -- it never bound anything for the whole of Phase 1 and
    nothing in the logs could say so. The norms themselves disambiguate: they are
    recorded BEFORE clipping, so the threshold can be checked against the scale
    it is supposed to act on (Phase 1 checkpoint N6)."""
    P = 32
    g_target = torch.randn(P)
    small = g_target / g_target.norm()                 # ||g|| = 1
    huge = 1000.0 * small                              # ||g|| = 1000

    info = {}
    compute_fu_weights([small, small, huge], g_target, normalize="none",
                       grad_clip=10.0, info=info)
    assert info["n_clipped"] == 1
    # Pre-clip scale: the max is what a threshold must be compared against.
    assert info["g_norm_max"] > 100.0, "norms must be recorded BEFORE clipping"
    assert abs(info["g_norm_median"] - 1.0) < 1e-4

    # A clean round: n_clipped is 0 here too -- only the norms tell the two apart.
    info_clean = {}
    compute_fu_weights([small, small, small], g_target, normalize="none",
                       grad_clip=10.0, info=info_clean)
    assert info_clean["n_clipped"] == 0
    assert info_clean["g_norm_max"] < 10.0, (
        "a clip that never binds on clean data must be visibly distinct from "
        "one whose threshold is in the wrong units")


def test_benign_zeroing_is_measured_so_attacker_zeroing_means_something():
    """``atk_w_mass = 0`` is only evidence of detection if honest clients survive.

    A gate that zeroes somebody every round produces the same 0.0 on the
    attacker as a gate that identified it -- same number, different claim.
    Phase 1 checkpoint D2 exists to separate them, and it needs this metric to be
    computable from the audit history."""
    from experiments.incentive_audit import attacker_weight_stats

    # Gate behaving: attacker zeroed, the three honest clients all keep weight.
    good = [{"agg_weights": [0.0, 0.34, 0.33, 0.33]} for _ in range(10)]
    st = attacker_weight_stats(good, [0], 4)
    assert st["atk_w_mass"] == 0.0
    assert st["benign_zeroed_frac"] == 0.0

    # Gate flailing: attacker zeroed, but an honest client is zeroed every round.
    bad = [{"agg_weights": [0.0, 0.0, 0.5, 0.5]} for _ in range(10)]
    st_bad = attacker_weight_stats(bad, [0], 4)
    assert st_bad["atk_w_mass"] == 0.0, "identical headline number ..."
    assert st_bad["benign_zeroed_frac"] == 1.0, "... but D2 must reject this"

    # The attack="none" row is what makes D2 readable, so the metric must be
    # defined when there is no attacker at all -- and with nobody excluded as
    # Byzantine, EVERY client counts as benign.
    st_none = attacker_weight_stats([{"agg_weights": [0.25] * 4}] * 10, [], 4)
    assert st_none["benign_zeroed_frac"] == 0.0
    # Same history, but read with client 0 declared Byzantine: client 0's zero
    # stops counting against the gate. The metric is relative to who is benign,
    # which is why D2 must be read off the attack="none" row and not inferred
    # from an attacked one.
    assert attacker_weight_stats(good, [], 4)["benign_zeroed_frac"] == 1.0
    assert attacker_weight_stats(good, [0], 4)["benign_zeroed_frac"] == 0.0


def test_diverged_model_reports_nan_not_auc_half():
    """SPEC 4.0(c): a dead model has no fairness, so it must not score as fair.

    `_scores` maps NaN predictions onto the constant 0.5, which reads out as
    `auc=0.5, dpd=0.0` -- chance accuracy and *perfect* fairness. That is a
    false positive in the method's own favour, and it is exactly what the
    sign_flip rows of the old incentive_audit reported."""
    from src.utils.metrics import all_metrics, diverged

    y = torch.tensor([0, 1, 0, 1])
    s = torch.tensor([0, 0, 1, 1])
    dead = torch.tensor([float("nan")] * 4)

    assert diverged(dead) and not diverged(torch.tensor([0.1, 0.9, 0.2, 0.8]))

    m = all_metrics(y, dead, s)
    assert m["diverged"] == 1.0
    assert m["auc"] != m["auc"], "AUC of a diverged model must be NaN, not 0.5"
    assert m["dpd"] != m["dpd"], "DPD of a diverged model must be NaN, not 0.0"

    ok = all_metrics(y, torch.tensor([0.1, 0.9, 0.2, 0.8]), s)
    assert ok["diverged"] == 0.0 and torch.isfinite(torch.tensor(ok["auc"]))
