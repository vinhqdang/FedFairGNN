"""CLAMP-RT tests (docs/clamp_rt_design.md, Sections 3-6). Offline, synthetic data.

Covers the client statistic d_k, the trimmed-mean server step, the T1 sensitivity
bound on the trimmed mean, the T3 bracket under every gap attack, the sigma_d
formula, the registrations, the gap attacks and an end-to-end trainer run.
"""
import math
import random
import warnings
from types import SimpleNamespace

import pytest
import torch

from experiments.methods import METHODS, apply_method
from src.config import ExperimentConfig
from src.federated.aggregation import secagg_trimmed
from src.federated.attacks import falsify_gaps, poison_updates
from src.federated.client import Client, client_floor
from src.federated.trainer import FederatedTrainer, _trimmed_diagnostics
from src.trust.privacy import calibrate_noise_multiplier

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _cfg(method, rounds=2, **kw):
    cfg = ExperimentConfig(dataset="synthetic", num_clients=4, rounds=rounds,
                           local_epochs=2, seed=7)
    apply_method(cfg, method)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _tm(gaps, b, *, updates=None, C=1.0, sigma_d=0.0, generator=None):
    """Pre-noise trimmed mean through the real server step (all updates valid)."""
    K = len(gaps)
    if updates is None:
        updates = [torch.zeros(3) for _ in range(K)]
    _, info = secagg_trimmed(updates, list(gaps), {}, C=C, tau=0.05, eta_lambda=0.5,
                             b=b, sigma_d=sigma_d, sigma_u=0.0, generator=generator,
                             release=True, direction_rule="sign")
    return info["tm_clean"]


def _stub_client(p, s, c):
    """A stand-in with the attributes local_gap reads, so d_k can be set exactly."""
    return SimpleNamespace(_val_blind_pred=lambda: (p, s),
                           cfg=SimpleNamespace(r2_floor_frac=c))


# --------------------------------------------------------------------------- #
# Client statistic (Section 3)
# --------------------------------------------------------------------------- #
def test_client_floor_is_ceil_with_float_guard_and_min_one():
    assert client_floor(30, 0.1) == 3          # 0.1 * 30 = 3.000...04 must not give 4
    assert client_floor(31, 0.1) == 4
    assert client_floor(1, 0.1) == 1
    assert client_floor(0, 0.1) == 1           # defined for an empty validation split
    assert client_floor(200, 0.05) == 10


def test_local_gap_matches_formula_and_is_clipped():
    p = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.3, 0.6], dtype=torch.float64)
    s = torch.tensor([1, 1, 0, 0, 0, 0])
    out = Client.local_gap(_stub_client(p, s, 0.1))
    n0 = client_floor(6, 0.1)                  # ceil(0.6) = 1
    m1 = (0.9 + 0.8) / max(2, n0)
    m0 = (0.1 + 0.2 + 0.3 + 0.6) / max(4, n0)
    assert out["d"] == pytest.approx(m1 - m0)
    assert (out["n0"], out["V0"], out["V1"]) == (n0, 4, 2)

    # A floor larger than the group shrinks the group mean toward 0:
    # n0 = 2: group 1 holds one node with p = 1 -> m1 = 1/2; group 0 holds p = 0 -> m0 = 0.
    out = Client.local_gap(_stub_client(torch.tensor([1.0, 0.0]), torch.tensor([1, 0]), 1.0))
    assert out["n0"] == 2
    assert out["d"] == pytest.approx(0.5)
    # Clipping: a difference can never leave [-1, 1].
    out = Client.local_gap(_stub_client(torch.tensor([1.0, 0.0]), torch.tensor([1, 0]), 0.0))
    assert -1.0 <= out["d"] <= 1.0


def test_local_gap_matches_real_client_forward_and_is_s_blind():
    trainer = FederatedTrainer(_cfg("clamp-rt-f"))
    client = trainer.clients[0]
    client.set_flat(trainer.global_flat)
    d = client.data
    mask = d.val_mask
    client.model.eval()
    with torch.no_grad():
        pred = client.model(d.x, d.edge_index,
                            None if client._model_uses_s else d.sensitive_attr)[mask].double()
    s = d.sensitive_attr[mask]
    n0 = client_floor(int(mask.sum()), client.cfg.r2_floor_frac)
    V = [int((s == a).sum()) for a in (0, 1)]
    m = [float(pred[s == a].sum()) / max(V[a], n0, 1) for a in (0, 1)]
    got = client.local_gap()
    assert got["d"] == pytest.approx(min(1.0, max(-1.0, m[1] - m[0])))
    assert (got["n0"], got["V0"], got["V1"]) == (n0, V[0], V[1])

    # s-blindness: flipping one validation node's attribute changes d_k only through
    # its group membership, so predictions are unchanged.
    v = int(mask.nonzero()[0])
    before = client.local_gap()["d"]
    d.sensitive_attr[v] = 1 - int(d.sensitive_attr[v])
    try:
        after = client.local_gap()["d"]
    finally:
        d.sensitive_attr[v] = 1 - int(d.sensitive_attr[v])
    n0_k = client_floor(int(mask.sum()), client.cfg.r2_floor_frac)
    assert abs(after - before) <= 2.0 / n0_k + 1e-12


# --------------------------------------------------------------------------- #
# Server step (Section 2, 6)
# --------------------------------------------------------------------------- #
def test_b0_trimmed_mean_is_the_plain_mean():
    rng = random.Random(3)
    for _ in range(50):
        K = rng.randint(2, 9)
        gaps = [rng.uniform(-1, 1) for _ in range(K)]
        assert _tm(gaps, 0) == pytest.approx(sum(gaps) / K, abs=1e-12)


def test_trimmed_mean_drops_b_each_side():
    gaps = [-1.0, -0.5, 0.0, 0.2, 0.4, 1.0]
    assert _tm(gaps, 1) == pytest.approx(sum([-0.5, 0.0, 0.2, 0.4]) / 4)
    assert _tm(gaps, 2) == pytest.approx((0.0 + 0.2) / 2)


def test_rejected_clients_gap_is_replaced_by_zero_and_update_excluded():
    C = 1.0
    updates = [torch.full((4,), 0.1), torch.full((4,), 9.0),   # client 1: norm > C
               torch.full((4,), 0.1), torch.full((4,), 0.1)]
    gaps = [0.6, 0.6, 1.5, 0.6]                                # client 2: gap out of range
    # Rejected clients 1 and 2 leave the update sum; their gaps enter the sort as 0:
    # sorted [0, 0, 0.6, 0.6], b = 0, mean = 0.3.
    state = {}
    g, info = secagg_trimmed(updates, gaps, state, C=C, tau=0.05, eta_lambda=0.5, b=0,
                             sigma_d=0.0, sigma_u=0.0, release=True)
    assert info["n_accepted"] == 2 and info["n_rejected"] == 2
    assert info["tm_clean"] == pytest.approx(0.3)
    assert torch.allclose(g, torch.full((4,), 0.1), atol=1e-7)

    # Gap outside the range alone, with a valid update, is also excluded.
    updates2 = [torch.full((4,), 0.1)] * 3
    _, info2 = secagg_trimmed(updates2, [0.2, 0.2, -1.2], {}, C=C, tau=0.05,
                              eta_lambda=0.5, b=0, sigma_d=0.0, sigma_u=0.0, release=True)
    assert info2["n_rejected"] == 1
    assert info2["tm_clean"] == pytest.approx(0.4 / 3)


def test_nan_gap_is_rejected():
    updates = [torch.zeros(3)] * 3
    _, info = secagg_trimmed(updates, [0.1, float("nan"), 0.1], {}, C=1.0, tau=0.05,
                             eta_lambda=0.5, b=0, sigma_d=0.0, sigma_u=0.0, release=True)
    assert info["n_rejected"] == 1 and math.isfinite(info["dhat"])


def test_release_false_leaves_lambda_and_direction_unchanged():
    state = {"r2_lambda": 0.3, "r2_dir": -1}
    updates = [torch.full((3,), 0.1)] * 3
    g, info = secagg_trimmed(updates, None, state, C=1.0, tau=0.05, eta_lambda=0.5, b=1,
                             sigma_d=0.0, sigma_u=0.0, release=False)
    assert state == {"r2_lambda": 0.3, "r2_dir": -1}
    assert info["lambda"] == 0.3 and info["direction"] == -1
    assert info["dhat"] is None and info["n_accepted"] == 3
    assert torch.allclose(g, torch.full((3,), 0.1), atol=1e-7)


def test_insufficient_clients_for_trimming_raises():
    with pytest.raises(ValueError):
        secagg_trimmed([torch.zeros(2)] * 4, [0.0] * 4, {}, C=1.0, tau=0.05, eta_lambda=0.5,
                       b=2, sigma_d=0.0, sigma_u=0.0, release=True)


def test_noise_is_two_independent_server_draws_on_the_trimmed_mean():
    gaps = [0.1, 0.3, -0.2, 0.4]
    gen = torch.Generator().manual_seed(5)
    _, info = secagg_trimmed([torch.zeros(2)] * 4, gaps, {}, C=1.0, tau=0.05,
                             eta_lambda=0.5, b=1, sigma_d=0.5, sigma_u=0.0,
                             generator=gen, release=True)
    xi = torch.randn(2, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    assert info["tm_clean"] == pytest.approx((0.1 + 0.3) / 2)
    assert info["gap"] == pytest.approx(0.2 + 0.5 * float(xi.sum()))
    assert info["dhat"] == pytest.approx(abs(info["gap"]))
    # Empirical variance is 2 sigma^2.
    gen = torch.Generator().manual_seed(0)
    vals = [secagg_trimmed([torch.zeros(2)] * 4, gaps, {}, C=1.0, tau=0.05, eta_lambda=0.5,
                           b=1, sigma_d=0.5, sigma_u=0.0, generator=gen,
                           release=True)[1]["gap"] - 0.2 for _ in range(4000)]
    var = sum(v * v for v in vals) / len(vals)
    assert var == pytest.approx(2 * 0.25, rel=0.1)


def test_t1_bound_is_attained_when_the_floor_binds():
    """A floor that binds and extreme predictions reach |Δd| = 2 / n0 exactly."""
    n, c = 20, 0.5                               # n0 = 10
    n0 = client_floor(n, c)
    s = torch.tensor([0] * 4 + [1] * 16)         # |V_0| = 4 < n0, |V_1| = 16 >= n0
    p = torch.tensor([1.0] * 4 + [0.0] * 16)
    d0 = Client.local_gap(_stub_client(p, s, c))["d"]
    s2 = s.clone(); s2[0] = 1                    # move a p = 1 node from group 0 to group 1
    d1 = Client.local_gap(_stub_client(p, s2, c))["d"]
    assert abs(d1 - d0) <= 2.0 / n0 + 1e-12
    assert abs(d1 - d0) >= 1.0 / n0 + 1.0 / 17 - 1e-12

# --------------------------------------------------------------------------- #
# T1: sensitivity of the trimmed mean to one flipped validation node
# --------------------------------------------------------------------------- #
def test_t1_trimmed_mean_sensitivity_bound_holds_under_node_flips():
    rng = random.Random(17)
    c = 0.1
    trials = 0
    for trial in range(60):
        K = rng.choice([3, 5, 6, 10])
        b = rng.choice([0, 1, 2]) if K >= 5 else 0
        if K - 2 * b < 1:
            continue
        sizes = [rng.randint(5, 60) for _ in range(K)]
        preds = [torch.tensor([rng.random() for _ in range(n)], dtype=torch.float64)
                 for n in sizes]
        groups = [torch.tensor([rng.randint(0, 1) for _ in range(n)]) for n in sizes]
        n0_min = min(client_floor(n, c) for n in sizes)
        bound = 2.0 / (n0_min * (K - 2 * b))

        def tm_of(preds_, groups_):
            gaps = [Client.local_gap(_stub_client(p, s, c))["d"]
                    for p, s in zip(preds_, groups_)]
            return _tm(gaps, b)

        base = tm_of(preds, groups)
        k = rng.randrange(K)
        v = rng.randrange(sizes[k])
        flipped = [s.clone() for s in groups]
        flipped[k][v] = 1 - flipped[k][v]
        moved = tm_of(preds, flipped)
        assert abs(moved - base) <= bound + 1e-12, (K, b, sizes, k, v)
        trials += 1
    assert trials >= 40


# --------------------------------------------------------------------------- #
# T3: bracket on the trimmed mean under falsified gaps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K,b", [(5, 1), (10, 2), (10, 1)])
def test_t3_bracket_holds_for_every_gap_attack_mode(K, b):
    rng = random.Random(29 + K + b)
    modes = ["hide", "inflate", "inlier", "zero", "random"]
    for _ in range(200):
        for f in range(1, b + 1):
            true = [rng.uniform(-1, 1) for _ in range(K)]
            byz = list(range(f))              # any f positions; honest are the rest
            honest = sorted(true[i] for i in range(K) if i not in byz)
            lo, hi = honest[b - f], honest[K - b - 1]   # h_(b-f+1), h_(K-b)
            for mode in modes:
                if mode == "random":
                    rep = list(true)
                    for i in byz:
                        rep[i] = rng.uniform(-1, 1)
                else:
                    rep = falsify_gaps(true, byz, mode)
                tm = _tm(rep, b)
                assert lo - 1e-12 <= tm <= hi + 1e-12, (mode, f, tm, lo, hi)


def test_trimmed_diagnostics_match_the_bracket_and_undefined_past_breakdown():
    true = [0.3, -0.4, 0.1, 0.7, -0.9]
    diag = _trimmed_diagnostics(true, [0], 1)
    honest = sorted([-0.4, 0.1, 0.7, -0.9])
    assert diag["honest_lo"] == honest[0] and diag["honest_hi"] == honest[3]
    assert diag["honest_spread"] == pytest.approx(honest[3] - honest[0])
    none = _trimmed_diagnostics(true, [0, 1], 1)     # f = 2 > b = 1: breakdown
    assert none == {"honest_lo": None, "honest_hi": None, "honest_spread": None}
    # f = 0: the bracket is [h_(b+1), h_(K-b)].
    clean = _trimmed_diagnostics(true, [], 1)
    srt = sorted(true)
    assert clean["honest_lo"] == srt[1] and clean["honest_hi"] == srt[3]


# --------------------------------------------------------------------------- #
# Gap attacks (Section 6)
# --------------------------------------------------------------------------- #
def test_falsify_gaps_modes():
    gaps = [0.3, 0.1, -0.2, 0.5, 0.0]
    byz = [0]                                 # honest: 0.1, -0.2, 0.5, 0.0 -> mean 0.1 > 0
    assert falsify_gaps(gaps, byz, "hide")[0] == -1.0
    assert falsify_gaps(gaps, byz, "inflate")[0] == 1.0
    assert falsify_gaps(gaps, byz, "inlier")[0] == -0.2   # min honest when g^H > 0
    assert falsify_gaps(gaps, byz, "zero")[0] == 0.0
    assert falsify_gaps(gaps, byz, "hide")[1:] == gaps[1:]  # honest entries untouched

    neg = [-0.3, -0.1, 0.2, -0.5]
    assert falsify_gaps(neg, [0], "hide")[0] == 1.0       # g^H < 0 (-0.1, 0.2, -0.5 -> -0.13)
    assert falsify_gaps(neg, [0], "inlier")[0] == 0.2     # max honest when g^H < 0
    with pytest.raises(ValueError):
        falsify_gaps(gaps, byz, "swap")


def test_gap_attacks_do_not_touch_updates():
    ups = [torch.randn(6) for _ in range(3)]
    metas = [{"perf": 0.9, "dpd": 0.1} for _ in range(3)]
    for attack in ("gap_hide", "gap_inflate", "gap_inlier", "gap_zero"):
        out_u, out_m = poison_updates(attack, ups, metas, [2], 10.0)
        assert all(torch.equal(a, b) for a, b in zip(ups, out_u))
        assert out_m == metas


# --------------------------------------------------------------------------- #
# Registrations, setup constants, trainer wiring
# --------------------------------------------------------------------------- #
def test_clamp_rt_registrations_mirror_clamp_settings():
    rt = {"clamp-rt": "clamp", "clamp-rt-f": "clamp-f", "clamp-rt-b0-f": "clamp-f",
          "clamp-rt-gcn-f": "clamp-gcn-f", "clamp-rt-ema-f": "clamp-f"}
    for name, base in rt.items():
        cfg, ref = ExperimentConfig(), ExperimentConfig()
        apply_method(cfg, name)
        apply_method(ref, base)
        assert cfg.aggregator == "secagg_lagrangian" and cfg.r2_stat_channel == "trimmed"
        assert cfg.fair_mode == "lagrangian" and cfg.dp_enabled is False
        for key in vars(ref):
            if key in ("exp_name", "r2_stat_channel", "r2_trim_b", "r2_gap_ema"):
                continue
            assert getattr(cfg, key) == getattr(ref, key), (name, key)
    assert METHODS["clamp-rt"].get("r2_output") is None
    assert METHODS["clamp-rt-f"]["r2_output"] == "feasible"
    assert METHODS["clamp-rt-b0-f"]["r2_trim_b"] == 0
    assert METHODS["clamp-rt-gcn-f"]["model"] == "gcn"
    cfg = ExperimentConfig()
    assert cfg.r2_stat_channel == "sums" and cfg.r2_trim_b == -1 and cfg.r2_floor_frac == 0.05 and cfg.r2_gap_ema == 1.0


def test_sums_channel_registrations_are_unchanged():
    for name in ("clamp", "clamp-f", "clamp-gcn-f"):
        assert "r2_stat_channel" not in METHODS[name]


def test_sigma_d_formula_and_setup_constants():
    from src.trust.privacy import calibrate_noise_multiplier as cal
    cfg = _cfg("clamp-rt-f", rounds=10, r2_stat_period=3)
    trainer = FederatedTrainer(cfg)
    K = 4
    n0_min = min(client_floor(int(d.val_mask.sum()), cfg.r2_floor_frac)
                 for d in trainer.clients_data)
    b = max(1, K // 5)
    z = cal(cfg.dp_epsilon, steps=math.ceil(cfg.rounds / cfg.r2_stat_period),
            delta=cfg.dp_delta)
    assert trainer._trim_b == b
    assert trainer._sigma_d == pytest.approx(2.0 / (n0_min * (K - 2 * b)) * z)
    info = trainer._trim_info
    assert info["n0_min"] == n0_min and info["trim_b"] == b and info["n_clients"] == K
    assert info["sigma_d"] == pytest.approx(trainer._sigma_d)
    assert info["n_small_groups"] == sum(
        1 for d in trainer.clients_data
        for a in (0, 1)
        if int(((d.val_mask) & (d.sensitive_attr == a)).sum())
        < client_floor(int(d.val_mask.sum()), cfg.r2_floor_frac))

    off = FederatedTrainer(_cfg("clamp-rt-f", r2_stat_dp=False))
    assert off._sigma_d == 0.0


def test_trim_b_override_and_breakdown_validation():
    assert FederatedTrainer(_cfg("clamp-rt-b0-f"))._trim_b == 0
    assert FederatedTrainer(_cfg("clamp-rt-f", r2_trim_b=1))._trim_b == 1
    with pytest.raises(ValueError):                      # K - 2b = 0
        FederatedTrainer(_cfg("clamp-rt-f", r2_trim_b=2))
    with pytest.raises(ValueError):
        FederatedTrainer(_cfg("clamp-rt-f", r2_stat_channel="median"))


def test_gap_attack_requires_trimmed_channel():
    with pytest.raises(ValueError):
        FederatedTrainer(_cfg("clamp-f", num_byzantine=1, attack="gap_hide"))
    with pytest.raises(ValueError):
        FederatedTrainer(_cfg("clamp-rt-f", num_byzantine=1, attack="stat_falsify"))


def test_end_to_end_clamp_rt_logs_round_fields_and_bracket(monkeypatch):
    import src.federated.trainer as tr
    seen = []
    orig = tr.falsify_gaps

    def spy(gaps, byz, mode):
        out = orig(gaps, byz, mode)
        seen.append((list(gaps), list(out), mode))
        return out

    monkeypatch.setattr(tr, "falsify_gaps", spy)
    cfg = _cfg("clamp-rt-f", rounds=3, num_byzantine=1, attack="gap_hide")
    res = FederatedTrainer(cfg).run()
    hist = res["history"]
    assert len(seen) == 3 and all(m == "hide" for _, _, m in seen)
    for true, rep, _ in seen:
        assert rep[0] in (-1.0, 0.0, 1.0)     # hide reports -sign(g^H) * 1
        assert rep[1:] == true[1:]
    for h in hist:
        for key in ("dhat", "gap", "tm_clean", "lambda", "direction", "n_rejected",
                    "honest_lo", "honest_hi", "honest_spread"):
            assert key in h
        assert h["n_rejected"] == 0            # falsified gaps are inside [-1, 1]
        assert h["honest_lo"] <= h["tm_clean"] + 1e-12 <= h["honest_hi"] + 1e-12
        assert h["honest_spread"] == pytest.approx(h["honest_hi"] - h["honest_lo"])
        assert math.isfinite(h["dhat"])
    final = res["final"]
    for key in ("n0_min", "trim_b", "sigma_d", "n_small_groups"):
        assert key in final
    assert final["trim_b"] == 1 and final["n0_min"] >= 1
    assert math.isfinite(final["auc"])


def test_off_period_rounds_request_no_gaps(monkeypatch):
    calls = []
    orig = Client.local_gap

    def counting(self):
        calls.append(1)
        return orig(self)

    monkeypatch.setattr(Client, "local_gap", counting)
    trainer = FederatedTrainer(_cfg("clamp-rt-f", rounds=4, r2_stat_period=2))
    calls.clear()
    hist = trainer.run()["history"]
    assert len(calls) == 4 * 2                 # rounds t = 0 and t = 2, four clients
    assert hist[1]["lambda"] == hist[0]["lambda"] and hist[1]["direction"] == hist[0]["direction"]
    assert hist[1]["dhat"] is None and hist[3]["dhat"] is None
    assert hist[1]["n_accepted"] == 4


# --------------------------------------------------------------------------- #
# Gap EMA (post-processing of released gaps; alpha = r2_gap_ema)
# --------------------------------------------------------------------------- #
def _release(gaps, state, *, alpha=1.0, tau=0.05, eta=0.5, release=True):
    """One trimmed release with noise off (raw gap = trimmed mean of the inputs)."""
    updates = [torch.zeros(3)] * len(gaps)
    _, info = secagg_trimmed(updates, gaps if release else None, state, C=1.0, tau=tau,
                             eta_lambda=eta, b=0, sigma_d=0.0, sigma_u=0.0,
                             release=release, direction_rule="sign", gap_ema=alpha)
    return info


def test_alpha_one_is_identical_to_unsmoothed_release():
    seq = [[0.3, 0.1, -0.2], [-0.4, 0.0, 0.2], [0.5, 0.5, 0.5]]
    default_state, explicit_state = {}, {}
    for gaps in seq:
        updates = [torch.zeros(3)] * len(gaps)
        g_def, a = secagg_trimmed(updates, gaps, default_state, C=1.0, tau=0.05,
                                  eta_lambda=0.5, b=0, sigma_d=0.0, sigma_u=0.0)
        g_one, b = secagg_trimmed(updates, gaps, explicit_state, C=1.0, tau=0.05,
                                  eta_lambda=0.5, b=0, sigma_d=0.0, sigma_u=0.0,
                                  gap_ema=1.0)
        assert torch.equal(g_def, g_one)
        assert a["gap"] == b["gap"] == a["gap_raw"] == b["gap_raw"]
        assert a["dhat"] == b["dhat"] and a["lambda"] == b["lambda"]
        assert a["direction"] == b["direction"]
        assert default_state == explicit_state
        assert default_state["r2_gap_ema"] == pytest.approx(b["gap_raw"], abs=0.0)


def test_ema_recursion_over_releases_with_off_period_round():
    alpha = 0.3
    state = {}
    raws = [0.5, -0.1, 0.2, 0.9]
    expected = None
    for i, r in enumerate(raws):
        info = _release([r, r, r], state, alpha=alpha)
        expected = r if expected is None else (1 - alpha) * expected + alpha * r
        assert state["r2_gap_ema"] == pytest.approx(expected, abs=1e-12)
        assert info["gap_raw"] == pytest.approx(r) and info["gap"] == pytest.approx(expected)
        if i == 1:
            # Off-period round between releases: no release, the ema is unchanged.
            before = dict(state)
            off = _release([9.0, 9.0, 9.0], state, alpha=alpha, release=False)
            assert state == before
            assert off["gap"] is None and off["gap_raw"] is None and off["dhat"] is None


def test_dhat_and_direction_use_the_smoothed_value():
    tau, alpha, eta = 0.1, 0.3, 0.5
    state = {}
    _release([0.5] * 3, state, alpha=alpha, tau=tau)          # ema = 0.5
    info = _release([-0.5] * 3, state, alpha=alpha, tau=tau)  # ema = 0.7*0.5 + 0.3*(-0.5) = 0.2
    assert info["gap_raw"] == pytest.approx(-0.5)
    assert info["gap"] == pytest.approx(0.2)
    assert info["dhat"] == pytest.approx(0.2)
    assert info["direction"] == 1                              # sign of the ema, not the raw gap
    # lambda: lam_2 = max(0, lam_1 + eta (|ema_2| - tau)), with lam_1 from the first release.
    state2 = {}
    first = _release([0.5] * 3, state2, alpha=alpha, tau=tau)
    second = _release([-0.5] * 3, state2, alpha=alpha, tau=tau)
    assert second["lambda"] == pytest.approx(max(0.0, first["lambda"] + eta * (0.2 - tau)))


def test_dead_zone_applies_to_the_smoothed_gap():
    tau = 0.1
    state = {}
    _release([0.4] * 3, state, alpha=0.3, tau=tau)
    info = _release([-0.4] * 3, state, alpha=0.3, tau=tau)    # ema = 0.28 - 0.12 = 0.16 > tau/2
    assert info["direction"] == 1
    state = {}
    _release([0.1] * 3, state, alpha=0.3, tau=tau)
    info = _release([-0.1] * 3, state, alpha=0.3, tau=tau)    # ema = 0.07 - 0.03 = 0.04 <= tau/2
    assert info["direction"] == 0


def test_sums_channel_does_not_read_or_write_the_gap_state():
    from src.federated.aggregation import secagg_lagrangian
    updates = [torch.zeros(3)] * 2
    stats = [{"S0": 30.0, "N0": 100.0, "S1": 60.0, "N1": 100.0}] * 2
    kw = dict(C=1.0, tau=0.05, eta_lambda=0.5, n_cap=1000, sigma_s=0.0, sigma_u=0.0)
    g1, i1 = secagg_lagrangian(updates, stats, {}, **kw)
    st = {"r2_gap_ema": 123.0}
    g2, i2 = secagg_lagrangian(updates, stats, st, **kw)
    assert torch.equal(g1, g2) and i1["gap"] == i2["gap"]
    assert st["r2_gap_ema"] == 123.0
    assert "gap_raw" not in i1


def test_trainer_gap_follows_ema_recursion_and_skips_off_rounds():
    cfg = _cfg("clamp-rt-ema-f", rounds=5, r2_stat_period=2)
    assert cfg.r2_gap_ema == 0.3 and cfg.r2_output == "feasible"
    hist = FederatedTrainer(cfg).run()["history"]
    expected = None
    released = 0
    for h in hist:
        if h["gap_raw"] is None:
            assert h["gap"] is None and h["dhat"] is None
            continue
        released += 1
        expected = h["gap_raw"] if expected is None else 0.7 * expected + 0.3 * h["gap_raw"]
        assert h["gap"] == pytest.approx(expected, abs=1e-12)
        assert h["dhat"] == pytest.approx(abs(expected), abs=1e-12)
    assert released == 3


def test_trainer_alpha_one_reports_raw_gap_as_smoothed():
    hist = FederatedTrainer(_cfg("clamp-rt-f", rounds=3)).run()["history"]
    for h in hist:
        assert h["gap"] == h["gap_raw"]
        assert h["dhat"] == pytest.approx(abs(h["gap_raw"]))


def test_invalid_alpha_is_rejected():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            FederatedTrainer(_cfg("clamp-rt-f", r2_gap_ema=bad))
