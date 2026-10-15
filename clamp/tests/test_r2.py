"""TrustFedGNN-R2 and FedFDP tests (offline, synthetic data only).

Covers the server-side secure-aggregation step, the s-blind statistics, the
signed Lagrangian client loss, the stat_falsify attack, FedFDP wiring, and an
exact check that existing methods are unchanged.
"""
import math
import warnings

import pytest
import torch

from experiments.methods import apply_method
from src.config import ExperimentConfig
from src.federated import aggregate
from src.federated.aggregation import secagg_lagrangian
from src.federated.attacks import falsify_group_stats
from src.federated.client import Client
from src.federated.trainer import FederatedTrainer

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _cfg(method, rounds=2, **kw):
    cfg = ExperimentConfig(dataset="synthetic", num_clients=4, rounds=rounds,
                           local_epochs=2, seed=7)
    apply_method(cfg, method)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _stats(s0, n0, s1, n1):
    return {"S0": float(s0), "N0": float(n0), "S1": float(s1), "N1": float(n1)}


def _secagg(**kw):
    p = dict(C=1.0, tau=0.05, eta_lambda=0.5, n_cap=1000, sigma_s=0.0, sigma_u=0.0,
             generator=torch.Generator().manual_seed(0))
    p.update(kw)
    return p


# --------------------------------------------------------------------------- #
# Aggregation (server side)
# --------------------------------------------------------------------------- #
def test_noise_free_aggregate_is_clipped_mean_when_all_valid():
    torch.manual_seed(0)
    updates = [torch.randn(20) for _ in range(4)]
    updates = [u / max(1.0, float(u.norm())) * 0.9 for u in updates]  # all norms <= C
    stats = [_stats(30, 100, 40, 100)] * 4
    sec = _secagg()
    sec.pop("tau")  # aggregate() takes tau as its own argument
    g, info = aggregate("secagg_lagrangian", updates, [{"n": 10}] * 4, state={},
                        stats=stats, secagg=sec)
    expected = torch.stack(updates).mean(0)
    assert torch.allclose(g, expected, atol=1e-6)
    assert info["n_accepted"] == 4 and info["n_rejected"] == 0
    assert info["weights"] == [0.25] * 4


def test_invalid_submissions_are_rejected_and_excluded_from_every_sum():
    updates = [torch.full((5,), 0.1), torch.full((5,), 5.0),   # #1: norm > C
               torch.full((5,), 0.1), torch.full((5,), 0.1)]
    stats = [
        _stats(30, 100, 60, 100),
        _stats(90, 100, 90, 100),        # norm-invalid, stats fine
        _stats(10, 100, 120, 100),       # S1 > N1: invalid
        _stats(20, 100, 40, 100),
    ]
    state = {}
    g, info = secagg_lagrangian(updates, stats, state, **_secagg())
    assert info["n_accepted"] == 2 and info["n_rejected"] == 2
    assert info["weights"] == [0.5, 0.0, 0.0, 0.5]
    assert torch.allclose(g, torch.full((5,), 0.1), atol=1e-7)
    # Rates come from accepted clients 0 and 3 only: r0 = 50/200, r1 = 100/200.
    assert info["gap"] == pytest.approx(0.25, abs=1e-9)
    assert info["dhat"] == pytest.approx(0.25, abs=1e-9)


def test_all_rejected_round_is_skipped_and_state_is_unchanged():
    updates = [torch.full((4,), 9.0), torch.full((4,), 9.0)]
    stats = [_stats(10, 100, 10, 100), _stats(10, 100, 10, 100)]
    state = {"r2_lambda": 0.3, "r2_dir": -1}
    g, info = secagg_lagrangian(updates, stats, state, **_secagg())
    assert info["n_accepted"] == 0
    assert torch.equal(g, torch.zeros(4))
    assert state == {"r2_lambda": 0.3, "r2_dir": -1}
    assert info["lambda"] == 0.3 and info["direction"] == -1


def test_direction_dead_zone():
    tau = 0.1
    u = [torch.zeros(3)]
    cases = {
        # gap, expected direction (dead zone is |g| <= tau/2 = 0.05)
        0.04: 0, -0.04: 0,
        0.06: 1, -0.2: -1,
    }
    for gap, want in cases.items():
        r0 = 0.3
        r1 = r0 + gap
        s = [_stats(100 * r0, 100, 100 * r1, 100)]
        _, info = secagg_lagrangian(u, s, {}, **_secagg(tau=tau))
        assert info["direction"] == want, (gap, info)
        assert info["dhat"] == pytest.approx(abs(gap), abs=1e-9)


def test_t3_bound_through_secagg_for_random_valid_adversarial_stats():
    """|Dhat - D^H| <= b_0 + b_1 with b_a = f n_cap / (N^H_a + f n_cap), checked
    through secagg_lagrangian (noise off) on random valid Byzantine claims."""
    g = torch.Generator().manual_seed(11)
    n_cap, K_h, f = 200, 3, 2
    for _ in range(100):
        honest = []
        for _ in range(K_h):
            N0, N1 = (int(torch.randint(1, 150, (1,), generator=g)) for _ in range(2))
            honest.append(_stats(float(torch.rand(1, generator=g)) * N0, N0,
                                 float(torch.rand(1, generator=g)) * N1, N1))
        byz = []
        for _ in range(f):
            N0, N1 = (int(torch.randint(0, n_cap + 1, (1,), generator=g)) for _ in range(2))
            byz.append(_stats(float(torch.rand(1, generator=g)) * N0, N0,
                              float(torch.rand(1, generator=g)) * N1, N1))
        zeros = [torch.zeros(3)] * (K_h + f)
        _, h = secagg_lagrangian(zeros[:K_h], honest, {}, **_secagg(n_cap=n_cap))
        _, a = secagg_lagrangian(zeros, honest + byz, {}, **_secagg(n_cap=n_cap))
        assert a["n_accepted"] == K_h + f
        NH = [sum(st[f"N{k}"] for st in honest) for k in (0, 1)]
        b = [f * n_cap / (NH[k] + f * n_cap) for k in (0, 1)]
        assert abs(a["dhat"] - h["dhat"]) <= b[0] + b[1] + 1e-12


def test_stat_falsify_mid_shrinks_gap_without_sign_change():
    g = torch.Generator().manual_seed(21)
    n_cap = 200
    for _ in range(100):
        honest = []
        for _ in range(4):
            N0, N1 = (int(torch.randint(1, 150, (1,), generator=g)) for _ in range(2))
            honest.append(_stats(float(torch.rand(1, generator=g)) * N0, N0,
                                 float(torch.rand(1, generator=g)) * N1, N1))
        stats = honest + [_stats(0, 0, 0, 0), _stats(0, 0, 0, 0)]
        fals = falsify_group_stats(stats, [4, 5], n_cap, mode="mid")
        for k in (4, 5):
            for a in (0, 1):
                assert 0 <= fals[k][f"S{a}"] <= fals[k][f"N{a}"] <= n_cap
        zeros = [torch.zeros(2)] * 6
        _, h = secagg_lagrangian(zeros[:4], honest, {}, **_secagg(n_cap=n_cap))
        _, a = secagg_lagrangian(zeros, fals, {}, **_secagg(n_cap=n_cap))
        assert a["gap"] * h["gap"] >= -1e-12          # no sign change
        assert abs(a["gap"]) <= abs(h["gap"]) + 1e-12  # gap shrinks


def test_t3_bound_holds_for_falsify_group_stats():
    n_cap = 150
    honest = [_stats(20 + i, 80 + i, 30 + 2 * i, 90) for i in range(3)]
    byz = [3, 4]
    stats = honest + [_stats(0, 0, 0, 0), _stats(0, 0, 0, 0)]
    fals = falsify_group_stats(stats, byz, n_cap)
    for k in byz:  # claims are valid
        for a in (0, 1):
            assert 0 <= fals[k][f"S{a}"] <= fals[k][f"N{a}"] <= n_cap
    S = [sum(st[f"S{a}"] for st in fals) for a in (0, 1)]
    N = [sum(st[f"N{a}"] for st in fals) for a in (0, 1)]
    SH = [sum(st[f"S{a}"] for st in honest) for a in (0, 1)]
    NH = [sum(st[f"N{a}"] for st in honest) for a in (0, 1)]
    for a in (0, 1):
        bound = len(byz) * n_cap / (NH[a] + len(byz) * n_cap)
        assert abs(S[a] / N[a] - SH[a] / NH[a]) <= bound + 1e-12
    # The swap pulls the aggregate rates toward each other, so the gap shrinks.
    honest_gap = abs(SH[1] / NH[1] - SH[0] / NH[0])
    fals_gap = abs(S[1] / N[1] - S[0] / N[0])
    assert fals_gap < honest_gap


def test_lambda_is_projected_and_t4_certificate_holds():
    """lambda >= 0 always, and (1/T) sum (Dhat_t - tau) <= lambda_T / (eta T)."""
    g = torch.Generator().manual_seed(3)
    tau, eta, T = 0.05, 0.5, 60
    state = {}
    dhats, lams = [], []
    for _ in range(T):
        d = float(torch.rand(1, generator=g)) * 0.8
        r0 = 0.1
        s = [_stats(100 * r0, 100, 100 * (r0 + d), 100)]
        _, info = secagg_lagrangian([torch.zeros(2)], s, state, **_secagg(tau=tau, eta_lambda=eta))
        assert info["lambda"] >= 0.0
        dhats.append(info["dhat"]); lams.append(info["lambda"])
    lhs = sum(d - tau for d in dhats) / T
    assert lhs <= lams[-1] / (eta * T) + 1e-9


def test_statistics_noise_is_drawn_and_gap_stays_finite():
    gen = torch.Generator().manual_seed(5)
    updates = [torch.full((3,), 0.1)] * 3
    stats = [_stats(40, 100, 60, 100)] * 3
    _, info = secagg_lagrangian(updates, stats, {},
                                **_secagg(sigma_s=5.0, sigma_u=0.1, generator=gen))
    assert math.isfinite(info["dhat"]) and 0.0 <= info["dhat"] <= 1.0


# --------------------------------------------------------------------------- #
# Client side
# --------------------------------------------------------------------------- #
def _gap(client):
    client.model.eval()
    with torch.no_grad():
        d = client.data
        pred = client.model(d.x, d.edge_index, d.sensitive_attr)[d.train_mask]
        s = d.sensitive_attr[d.train_mask]
        return float(pred[s == 1].mean() - pred[s == 0].mean())


def test_signed_lagrangian_loss_pushes_rates_together():
    """With d = sign(gap), the signed term lambda * d * (m1 - m0) moves the gap
    against d relative to plain weighted BCE (lambda = 0). Same start, dropout
    off, so the two runs differ only in that term. The linear term has no
    stopping point at zero, so this checks direction, not convergence."""
    cfg = _cfg("trustfedgnn-r2", local_epochs=10)
    cfg.local_lr = 0.02
    cfg.dropout = 0.0
    trainer = FederatedTrainer(cfg)
    client = trainer.clients[0]
    client.set_flat(trainer.global_flat)
    gap0 = _gap(client)
    assert abs(gap0) > 1e-3
    start = trainer.global_flat.clone()
    d = 1 if gap0 > 0 else -1

    client.lagrange = (0.0, 0)
    client.train()
    gap_base = _gap(client)

    client.set_flat(start)
    client.lagrange = (0.5, d)
    client.train()
    gap_lam = _gap(client)
    assert d * (gap_lam - gap_base) < 0


def test_group_stats_are_s_blind_and_flip_moves_only_membership():
    cfg = _cfg("trustfedgnn-r2")
    trainer = FederatedTrainer(cfg)
    client = trainer.clients[0]
    client.r2_n_cap = 10 ** 6
    client.set_flat(trainer.global_flat)
    d = client.data
    v = int(d.val_mask.nonzero()[0])
    s_old = int(d.sensitive_attr[v])

    client.model.eval()
    with torch.no_grad():
        p_v = float(client.model(d.x, d.edge_index, None)[v])
    base = client.group_stats()

    d.sensitive_attr[v] = 1 - s_old
    try:
        flipped = client.group_stats()
    finally:
        d.sensitive_attr[v] = s_old

    assert flipped["S0"] + flipped["S1"] == pytest.approx(base["S0"] + base["S1"], abs=1e-4)
    assert flipped["N0"] + flipped["N1"] == base["N0"] + base["N1"]
    sign = 1.0 if s_old == 0 else -1.0       # node v leaves group s_old, joins the other
    if s_old == 0:
        assert flipped["S0"] == pytest.approx(base["S0"] - p_v, abs=1e-4)
        assert flipped["S1"] == pytest.approx(base["S1"] + p_v, abs=1e-4)
        assert flipped["N0"] == base["N0"] - 1 and flipped["N1"] == base["N1"] + 1
    else:
        assert flipped["S0"] == pytest.approx(base["S0"] + p_v, abs=1e-4)
        assert flipped["S1"] == pytest.approx(base["S1"] - p_v, abs=1e-4)
        assert flipped["N0"] == base["N0"] + 1 and flipped["N1"] == base["N1"] - 1


def test_group_stats_cut_to_first_n_cap_nodes():
    cfg = _cfg("trustfedgnn-r2")
    trainer = FederatedTrainer(cfg)
    client = trainer.clients[0]
    client.r2_n_cap = 3
    client.set_flat(trainer.global_flat)
    st = client.group_stats()
    assert st["N0"] <= 3 and st["N1"] <= 3
    assert all(0 <= st[f"S{a}"] <= st[f"N{a}"] for a in (0, 1))


def test_r2_n_cap_is_derived_from_partition():
    trainer = FederatedTrainer(_cfg("trustfedgnn-r2"))
    largest = max(int(d.val_mask.sum()) for d in trainer.clients_data)
    assert trainer._r2_n_cap == math.ceil(largest / 50) * 50
    assert all(c.r2_n_cap == trainer._r2_n_cap for c in trainer.clients)


def test_sigma_s_is_two_times_calibrated_multiplier():
    from src.trust.privacy import calibrate_noise_multiplier
    cfg = _cfg("trustfedgnn-r2")
    trainer = FederatedTrainer(cfg)
    z = calibrate_noise_multiplier(cfg.dp_epsilon, steps=cfg.rounds, delta=cfg.dp_delta)
    assert trainer._secagg_sigma_s == pytest.approx(2.0 * z)
    cfg_off = _cfg("trustfedgnn-r2", r2_stat_dp=False)
    assert FederatedTrainer(cfg_off)._secagg_sigma_s == 0.0


# --------------------------------------------------------------------------- #
# Attacks, FedFDP, end-to-end
# --------------------------------------------------------------------------- #
def test_stat_falsify_does_not_touch_updates():
    from src.federated.attacks import poison_updates
    ups = [torch.randn(6) for _ in range(3)]
    metas = [{"perf": 0.9, "dpd": 0.1} for _ in range(3)]
    out_u, out_m = poison_updates("stat_falsify", ups, metas, [2], 10.0)
    assert all(torch.equal(a, b) for a, b in zip(ups, out_u))
    assert out_m == metas


def test_end_to_end_trustfedgnn_r2_records_round_diagnostics():
    cfg = _cfg("trustfedgnn-r2", rounds=3, num_byzantine=1, attack="stat_falsify")
    res = FederatedTrainer(cfg).run()
    hist = res["history"]
    assert len(hist) == 3
    for h in hist:
        for key in ("lambda", "dhat", "gap", "n_accepted", "n_rejected", "direction"):
            assert key in h
        assert math.isfinite(h["dhat"]) and h["lambda"] >= 0.0
        assert h["n_accepted"] == 4 and h["n_rejected"] == 0  # falsified claims are valid
    assert math.isfinite(res["final"]["auc"])


def test_end_to_end_fedfdp_runs_and_tracks_fbar():
    cfg = _cfg("fedfdp", rounds=3)
    res = FederatedTrainer(cfg).run()
    hist = res["history"]
    assert len(hist) == 3
    assert all("fedfdp_fbar" in h and math.isfinite(h["fedfdp_fbar"]) for h in hist)
    assert math.isfinite(res["final"]["auc"])


def test_fedfdp_local_step_uses_node_factor_and_clip():
    cfg = _cfg("fedfdp", rounds=2)
    trainer = FederatedTrainer(cfg)
    client = trainer.clients[0]
    client.set_flat(trainer.global_flat)
    client.fedfdp_fbar = 0.0
    client.train()
    rep = client.meta()["loss"]
    assert math.isfinite(rep)
    assert client._fedfdp_clip == pytest.approx(max(1e-3, rep))


def test_fedavg_is_bit_identical_to_pre_r2_code():
    """Golden values from the pre-R2 code (same config, same seed)."""
    cfg = ExperimentConfig(dataset="synthetic", num_clients=4, rounds=2, local_epochs=2, seed=7)
    apply_method(cfg, "fedavg-gcn")
    cfg.exp_name = "fedavg-gcn"
    trainer = FederatedTrainer(cfg)
    res = trainer.run()
    assert float(trainer.global_flat.double().sum()) == 0.44653553306125104
    assert res["history"][-1]["g_auc"] == 0.9739670408406974


# --------------------------------------------------------------------------- #
# CLAMP registrations, off-period rounds, trainer-level checks
# --------------------------------------------------------------------------- #
def test_clamp_registrations():
    expect = {"clamp": "gat", "clamp-gcn": "gcn", "clamp-clipmean": "gat"}
    for name, model in expect.items():
        cfg = ExperimentConfig()
        apply_method(cfg, name)
        assert cfg.model == model
        assert cfg.aggregator == "secagg_lagrangian" and cfg.fair_mode == "lagrangian"
        assert cfg.dp_enabled is False
    cfg = ExperimentConfig()
    apply_method(cfg, "clamp-clipmean")
    assert cfg.r2_lambda_step == 0.0


def test_off_period_rounds_request_no_stats(monkeypatch):
    from src.federated.client import Client
    calls = []
    orig = Client.group_stats

    def counting(self):
        calls.append(1)
        return orig(self)

    monkeypatch.setattr(Client, "group_stats", counting)
    trainer = FederatedTrainer(_cfg("trustfedgnn-r2", rounds=4, r2_stat_period=2))
    calls.clear()
    hist = trainer.run()["history"]
    # Releases in rounds t = 0 and t = 2 only, for each of the 4 clients.
    assert len(calls) == 4 * 2
    assert hist[1]["lambda"] == hist[0]["lambda"] and hist[1]["direction"] == hist[0]["direction"]
    assert hist[1]["dhat"] is None and hist[3]["dhat"] is None
    assert hist[1]["n_accepted"] == 4


def test_sigma_s_uses_ceil_rounds_over_period():
    from src.trust.privacy import calibrate_noise_multiplier
    cfg = _cfg("trustfedgnn-r2", rounds=10, r2_stat_period=3)
    trainer = FederatedTrainer(cfg)
    z = calibrate_noise_multiplier(cfg.dp_epsilon, steps=4, delta=cfg.dp_delta)  # ceil(10/3)
    assert trainer._secagg_sigma_s == pytest.approx(2.0 * z)


def test_group_stats_uncapped_when_n_cap_is_zero():
    trainer = FederatedTrainer(_cfg("trustfedgnn-r2"))
    c = trainer.clients[0]
    c.r2_n_cap = 0
    c.set_flat(trainer.global_flat)
    st = c.group_stats()
    assert st["N0"] + st["N1"] == int(c.data.val_mask.sum())


def test_stub_clients_submit_v_and_global_moves_by_minus_v(monkeypatch):
    cfg = _cfg("trustfedgnn-r2")
    trainer = FederatedTrainer(cfg)
    old = trainer.global_flat.clone()
    v = torch.randn(old.numel())
    v = v / v.norm() * 0.5 * cfg.r2_clip_norm            # norm < C
    monkeypatch.setattr(Client, "get_flat", lambda self: old - v)
    trainer._round(0)
    assert torch.allclose(trainer.global_flat, old - v, atol=1e-6)


def test_all_rejected_round_leaves_global_lambda_and_direction_unchanged(monkeypatch):
    import src.federated.trainer as T
    cfg = _cfg("trustfedgnn-r2")
    trainer = FederatedTrainer(cfg)
    trainer._agg_state.update({"r2_lambda": 0.3, "r2_dir": -1})
    old = trainer.global_flat.clone()
    v = torch.randn(old.numel())
    v = v / v.norm() * 5.0 * cfg.r2_clip_norm            # norm > C for every client
    monkeypatch.setattr(Client, "get_flat", lambda self: old - v)
    monkeypatch.setattr(T, "_clip_to_norm", lambda u, C: u)  # non-compliant: skip step 3
    rec = trainer._round(0)
    assert torch.equal(trainer.global_flat, old)
    assert trainer._agg_state["r2_lambda"] == 0.3 and trainer._agg_state["r2_dir"] == -1
    assert rec["n_accepted"] == 0 and rec["n_rejected"] == cfg.num_clients


def test_fedfdp_meta_is_cached_and_does_not_move_clip():
    trainer = FederatedTrainer(_cfg("fedfdp", rounds=2))
    client = trainer.clients[0]
    client.set_flat(trainer.global_flat)
    client.train()
    clip = client._fedfdp_clip
    a = client.meta()["loss"]
    b = client.meta()["loss"]
    assert a == b and client._fedfdp_clip == clip


# --------------------------------------------------------------------------- #
# Design options: continuous direction and feasible-iterate output (opt-in)
# --------------------------------------------------------------------------- #
def test_prop_direction_is_proportional_and_saturates():
    tau = 0.1
    cases = {0.05: 0.5, -0.05: -0.5, 0.3: 1.0, -0.3: -1.0, 0.0: 0.0}
    for gap, want in cases.items():
        r0 = 0.3
        s = [_stats(100 * r0, 100, 100 * (r0 + gap), 100)]
        _, info = secagg_lagrangian([torch.zeros(3)], s, {},
                                    **_secagg(tau=tau, direction_rule="prop"))
        assert info["direction"] == pytest.approx(want, abs=1e-9), gap


def test_unknown_direction_rule_is_rejected():
    s = [_stats(30, 100, 60, 100)]
    with pytest.raises(ValueError):
        secagg_lagrangian([torch.zeros(3)], s, {}, **_secagg(direction_rule="bogus"))


def test_invalid_design_options_are_rejected_by_the_trainer():
    with pytest.raises(ValueError):
        FederatedTrainer(_cfg("trustfedgnn-r2", r2_direction="bogus"))
    with pytest.raises(ValueError):
        FederatedTrainer(_cfg("trustfedgnn-r2", r2_output="bogus"))


def _scripted_secagg(dhats):
    """Stand-in for secagg_lagrangian: a fixed Dhat per release, and a constant
    aggregate 0.1 per round so every round's model is distinct."""
    calls = {"i": 0}

    def stub(updates, stats, state, *, C, tau, eta_lambda, n_cap, sigma_s, sigma_u,
             generator=None, release=True, direction_rule="sign"):
        d = dhats[calls["i"]]
        calls["i"] += 1
        K = len(updates)
        g = torch.full_like(updates[0].flatten(), 0.1).view_as(updates[0])
        info = {"weights": [1.0 / K] * K, "n_accepted": K, "n_rejected": 0,
                "lambda": 0.0, "lambda_in": 0.0, "direction": 0, "dhat": d,
                "gap": d, "u_norm": 1.0}
        return g, info
    return stub


def test_feasible_output_selects_last_feasible_round(monkeypatch):
    import src.federated.aggregation as agg_mod
    cfg = _cfg("trustfedgnn-r2", rounds=4, r2_output="feasible")
    cfg.fairness_budget = 0.03
    monkeypatch.setattr(agg_mod, "secagg_lagrangian", _scripted_secagg([0.5, 0.01, 0.4, 0.02]))
    trainer = FederatedTrainer(cfg)
    g0 = trainer.global_flat.clone()
    res = trainer.run()
    # Feasible rounds are 2 and 4. Round 4 wins; its statistics were computed on theta^3.
    assert res["final"]["selected_round"] == 4
    assert torch.allclose(trainer._feasible_flat, g0 - 0.3, atol=1e-6)
    assert len(res["history"]) == 4
    last = trainer.global_flat.clone()
    trainer.global_flat = trainer._feasible_flat
    expected = trainer.evaluate_global()
    trainer.global_flat = last
    assert res["final"]["auc"] == pytest.approx(expected["auc"])


def test_feasible_output_falls_back_to_last_model_when_none_feasible(monkeypatch):
    import src.federated.aggregation as agg_mod
    cfg = _cfg("trustfedgnn-r2", rounds=3, r2_output="feasible")
    cfg.fairness_budget = 0.03
    monkeypatch.setattr(agg_mod, "secagg_lagrangian", _scripted_secagg([0.5, 0.4, 0.3]))
    trainer = FederatedTrainer(cfg)
    res = trainer.run()
    assert res["final"]["selected_round"] is None
    assert res["final"]["auc"] == pytest.approx(trainer.evaluate_global()["auc"])


def test_default_output_has_no_selection_key():
    trainer = FederatedTrainer(_cfg("trustfedgnn-r2", rounds=1))
    assert "selected_round" not in trainer.run()["final"]


def test_clamp_design_variants_are_registered():
    expect = {"clamp-p": ("prop", "last"), "clamp-f": ("sign", "feasible"),
              "clamp-pf": ("prop", "feasible")}
    for name, (direction, output) in expect.items():
        cfg = ExperimentConfig()
        apply_method(cfg, name)
        assert cfg.r2_direction == direction and cfg.r2_output == output
        assert cfg.aggregator == "secagg_lagrangian" and cfg.model == "gat"


def _stub_eval(trainer):
    """Cheap stand-in for evaluate_global: its AUC is the sum of the evaluated model,
    so the test can tell which model was scored."""
    trainer.evaluate_global = lambda: {"auc": float(trainer.global_flat.double().sum()),
                                       "dpd": 0.0, "eod": 0.0}


def test_warmup_round_1_only_feasible_falls_back_to_last_model(monkeypatch):
    import src.federated.aggregation as agg_mod
    cfg = _cfg("trustfedgnn-r2", rounds=30, r2_output="feasible")
    cfg.fairness_budget = 0.03
    dhats = [0.01] + [0.5] * 29          # only round 1 is feasible
    monkeypatch.setattr(agg_mod, "secagg_lagrangian", _scripted_secagg(dhats))
    trainer = FederatedTrainer(cfg)
    _stub_eval(trainer)
    res = trainer.run()
    assert res["final"]["selected_round"] is None
    assert res["final"]["auc"] == pytest.approx(float(trainer.global_flat.double().sum()))


def test_warmup_selects_the_feasible_round_after_warmup(monkeypatch):
    import src.federated.aggregation as agg_mod
    cfg = _cfg("trustfedgnn-r2", rounds=30, r2_output="feasible")
    cfg.fairness_budget = 0.03
    dhats = [0.01] + [0.5] * 23 + [0.01] + [0.5] * 5   # feasible at rounds 1 and 25
    monkeypatch.setattr(agg_mod, "secagg_lagrangian", _scripted_secagg(dhats))
    trainer = FederatedTrainer(cfg)
    g0 = trainer.global_flat.clone()
    _stub_eval(trainer)
    res = trainer.run()
    assert res["final"]["selected_round"] == 25
    # theta^24 = g0 - 24 * 0.1 is the model the round-25 statistics were computed on.
    assert res["final"]["auc"] == pytest.approx(float((g0 - 2.4).double().sum()), rel=1e-5)


def test_public_n_cap_is_invariant_to_flipping_s():
    from src.federated.trainer import public_n_cap
    trainer = FederatedTrainer(_cfg("trustfedgnn-r2"))
    before = public_n_cap(trainer.clients_data)
    for d in trainer.clients_data:
        v = int(d.val_mask.nonzero()[0])
        d.sensitive_attr[v] = 1 - d.sensitive_attr[v]
    assert public_n_cap(trainer.clients_data) == before
    assert before % 50 == 0 and before >= max(int(d.val_mask.sum()) for d in trainer.clients_data)
