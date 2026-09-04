"""Server-side aggregation rules.

All rules operate on a list of client *pseudo-gradients* ``g_k = theta_global -
theta_local_k`` (flat tensors). The server applies ``theta_new = theta_global -
g_agg``; with weights summing to 1 this reduces to a weighted average of local
models, so FedAvg, BFWA and the robust rules share one interface.

Rules
    fedavg        weighted by client sample count.
    bfwa          Bi-objective Frank-Wolfe: maximise weighted utility s.t. a
                  hard bound tau on weighted demographic-parity gap (ours).
    krum          Blanchard et al. 2017 -- pick the update closest to its
                  n-f-2 nearest neighbours.
    multikrum     average the m best Krum-scored updates.
    median        coordinate-wise median (Yin et al. 2018).
    trimmed_mean  coordinate-wise beta-trimmed mean (Yin et al. 2018).
    robust_bfwa   distance-screen Byzantine updates, then BFWA on survivors
                  with the fairness constraint (ours; robust + fair).
    fairgfl       weight ~ 1/(1+heterogeneity); heterogeneity proxied by a
                  client's sample-count deviation from the mean (Khan-style
                  overlap-ratio reweighting; see docs/BASELINES_AND_SOURCES.md).
    fedgraphfair  minimax/DRO dual-ascent reweighting toward high-loss clients,
                  simplex-projected, with lambda persisted across rounds.
    popets_fairfed FHE-friendly FairFed: degree-2 polynomial in place of
                  exp(-beta|.|) so the weighting stays homomorphically
                  computable (the crypto itself is not reimplemented).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


# --------------------------------------------------------------------------- #
# Frank-Wolfe fairness-constrained weights
# --------------------------------------------------------------------------- #
def _scale(v: torch.Tensor) -> float:
    """Positive scale of a K-vector: its range, falling back to its magnitude
    and finally to 1.0 (K=1 or all-equal entries)."""
    if v.numel() == 0:
        return 1.0
    rng = float(v.max() - v.min())
    if rng > 1e-12:
        return rng
    mag = float(v.abs().max())
    return mag if mag > 1e-12 else 1.0


def bfwa_weights(perf: torch.Tensor, dpd: torch.Tensor, tau: float,
                 iters: int = 20, dual_step: float = 0.1,
                 w_min_factor: float = 0.2, mu_init: float = 0.0,
                 info: Dict = None) -> torch.Tensor:
    """Solve  max_w  w.perf  s.t.  w.dpd <= tau,  w in simplex  via Frank-Wolfe
    with a dual variable enforcing the fairness budget.

    Args:
        mu_init: starting value of the dual multiplier. The caller passes the
            multiplier carried over from the previous communication round when
            ``bfwa_persist_dual`` is on (see :func:`aggregate`); ``0.0``
            restarts the dual ascent every call.
        info: optional dict filled in-place with ``bfwa_mu`` (the multiplier
            after this call, to be persisted), the constraint residual
            ``w.dpd - tau`` and the feasibility flag, both BEFORE
            (``*_preclamp``) and AFTER the weight-floor clamp.

    Two properties this loop must have, and did not before (both are pinned by
    ``tests/test_revision_invariants.py``):

    * **The Frank-Wolfe averaging must actually average.** ``gamma = 2/(t+2)``
      evaluated at ``t = 0`` is ``1.0``: the first step *discards* the uniform
      iterate and jumps straight onto the vertex ``argmin(-perf + mu*dpd)``.
      Because the objective is linear in ``w`` its gradient does not depend on
      ``w``, so every later step re-selects that same vertex until ``mu`` grows
      large enough to flip it -- which, at the shipped ``iters=20``, never
      happened, and ``tau`` had literally no effect on the returned weights.
      The classic schedule is therefore run from ``t = 1`` (``gamma_1 = 2/3``),
      so the uniform initialisation is mixed into the iterate rather than
      erased.
    * **The dual must be able to reach the binding regime.** The vertex flips
      only once ``mu`` exceeds (utility gap)/(disparity gap), a ratio in the
      units of ``perf`` per unit of ``dpd``. With raw units and the shipped
      ``dual_step=0.1`` that takes hundreds of iterations. The Lagrangian is
      therefore evaluated on range-normalised terms, which makes ``mu``
      dimensionless (order 1 at the flip point) and gives ``dual_step`` the
      same meaning across datasets, scales and rounds -- necessary for the
      multiplier to be persisted across rounds at all.
    """
    K = perf.numel()
    dev = perf.device
    w = torch.full((K,), 1.0 / K, device=dev)
    mu = float(mu_init)
    perf_scale = _scale(perf)                       # dimensionless Lagrangian
    dpd_scale = _scale(dpd)
    for t in range(1, iters + 1):                   # t=1 -> gamma=2/3, not 1.0
        violation = float(torch.dot(w, dpd) - tau)
        grad = -perf / perf_scale + mu * dpd / dpd_scale   # d/dw of Lagrangian
        s = torch.zeros(K, device=dev)
        s[int(torch.argmin(grad))] = 1.0            # LMO on simplex
        gamma = 2.0 / (t + 2.0)                     # decaying step (spec-compliant)
        w = (1 - gamma) * w + gamma * s
        mu = max(0.0, mu + dual_step * violation / dpd_scale)
    resid_pre = float(torch.dot(w, dpd)) - tau
    # weight floor + renormalise (avoid a single client dominating)
    w = torch.clamp(w, min=w_min_factor / K)
    w = w / w.sum()
    resid_post = float(torch.dot(w, dpd)) - tau
    if info is not None:
        # The floor is part of the shipped rule, so the headline residual is the
        # post-clamp one -- it is what the aggregated model actually incurs. The
        # pre-clamp value is reported alongside it so the floor's cost in
        # feasibility is visible rather than hidden.
        info["bfwa_mu"] = mu
        info["constraint_residual_preclamp"] = resid_pre
        info["feasible_preclamp"] = bool(resid_pre <= 0.0)
        info["constraint_residual"] = resid_post
        info["feasible"] = bool(resid_post <= 0.0)
    return w


# --------------------------------------------------------------------------- #
# Robust primitives
# --------------------------------------------------------------------------- #
def _pairwise_sq_dists(mat: torch.Tensor) -> torch.Tensor:
    return torch.cdist(mat, mat) ** 2


def krum_scores(updates: List[torch.Tensor], f: int) -> torch.Tensor:
    K = len(updates)
    mat = torch.stack([u.flatten() for u in updates])
    d = _pairwise_sq_dists(mat)
    m = max(1, K - f - 2)                            # neighbours to sum
    scores = torch.empty(K, device=mat.device)
    for i in range(K):
        di = torch.cat([d[i, :i], d[i, i + 1:]])
        scores[i] = torch.sort(di).values[:m].sum()
    return scores


# --------------------------------------------------------------------------- #
# Null players
# --------------------------------------------------------------------------- #
def _zero_null_players(w: torch.Tensor, stack: torch.Tensor) -> torch.Tensor:
    """Force weight 0 on clients whose update is exactly zero, then renormalise.

    Null-player axiom: a client that submitted nothing must receive no credit.
    Any *fallback* that spreads weight uniformly (or by sample size) over all K
    clients silently breaks it, so the mask is applied to every weight vector
    the FU-Shapley branch can produce, not just to the ReLU-gated one.
    """
    null = stack.abs().amax(dim=1) == 0
    if not bool(null.any()):
        return w
    w = w.clone()
    w[null] = 0.0
    total = float(w.sum())
    if total > 0:
        return w / total
    active = (~null).to(w.dtype)
    if float(active.sum()) > 0:
        return active / active.sum()
    return torch.full_like(w, 1.0 / w.numel())   # everyone null: aggregate is 0


# --------------------------------------------------------------------------- #
# Cross-round dual state for BFWA
# --------------------------------------------------------------------------- #
def _read_mu(state: Dict, key: str, persist: bool) -> float:
    """Dual multiplier carried in from the previous round (0.0 if none)."""
    if not persist or state is None:
        return 0.0
    return float(state.get(key, 0.0))


def _write_mu(state: Dict, key: str, mu: float, persist: bool) -> None:
    if persist and state is not None:
        state[key] = float(mu)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def aggregate(method: str, updates: List[torch.Tensor], meta: List[dict],
              *, tau: float = 0.05, fw_iters: int = 20, dual_step: float = 0.1,
              trimmed_beta: float = 0.1, krum_f: int = 1,
              q_ffl: float = 2.0, fairfed_beta: float = 1.0,
              state: Dict = None,
              g_target: torch.Tensor = None, g_task: torch.Tensor = None,
              g_fair: torch.Tensor = None, fu_alpha: float = 0.1,
              fu_beta_ema: float = 0.9, fu_normalize: str = "target_norm",
              fu_score: str = "dot", fu_warmup: bool = False,
              fu_grad_clip: float = 0.0, fu_warmup_agg: str = "fedavg",
              bfwa_persist_dual: bool = True,
              ) -> Tuple[torch.Tensor, Dict]:
    """Aggregate client pseudo-gradients under ``method``.

    ``bfwa_persist_dual`` (default True, matching ``cfg.bfwa_persist_dual``)
    carries the BFWA dual multiplier across communication rounds through
    ``state`` -- the same pattern as ``state['fedgraphfair_lambda']`` and
    ``state['fu_phi_ema']``. ``bfwa`` and ``robust_bfwa`` keep separate keys so
    the two rules never share a multiplier. False restores the pre-fix
    behaviour (the dual restarts from 0 every round, so ``tau`` never binds);
    it exists to keep that regression reproducible, not for reporting.
    """
    K = len(updates)
    stack = torch.stack([u.flatten() for u in updates])
    # Every scalar summary below is built from python floats, so it must be
    # placed on the same device as the updates or GPU runs die in the first
    # weighted sum.
    dev = stack.device
    _t = lambda vals: torch.tensor(vals, dtype=torch.float32, device=dev)
    n = _t([m.get("n", 1) for m in meta])
    perf = _t([m.get("perf", 0.5) for m in meta])
    dpd = _t([m.get("dpd", 0.0) for m in meta])
    loss = _t([m.get("loss", 1.0 - m.get("perf", 0.5)) for m in meta])
    info: Dict = {"method": method}

    if method == "fedavg":
        w = n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "fairfed":
        # FairFed (Ezzeldin et al., AAAI 2023): start from data-size weights and
        # down-weight clients whose local fairness gap exceeds the mean gap.
        mean_gap = dpd.mean()
        w = n / n.sum()
        w = w - fairfed_beta * (dpd - mean_gap)
        w = torch.clamp(w, min=0.0)
        w = w / w.sum() if w.sum() > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "qffl":
        # q-FedAvg (Li et al., ICLR 2020): up-weight high-loss clients for
        # client-level (performance) fairness.
        w = (loss.clamp(min=1e-6)) ** q_ffl
        w = w / w.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "f2gnn":
        # F2GNN (Meng et al.): softmax aggregation combining a model-fairness
        # weight (lower DPD -> higher weight) and a data-balance weight (group
        # balance per client), temperature-scaled.
        gbal = _t([1.0 - abs(2.0 * m.get("group1_rate", 0.5) - 1.0) for m in meta])
        gamma_f = torch.softmax(-dpd / 0.1, dim=0)
        gamma_e = torch.softmax(gbal / 0.1, dim=0)
        w = torch.softmax((0.5 * gamma_e + gamma_f) / 0.1, dim=0)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "bfwa":
        mu0 = _read_mu(state, "bfwa_mu", bfwa_persist_dual)
        fw_info: Dict = {}
        w = bfwa_weights(perf, dpd, tau, fw_iters, dual_step,
                         mu_init=mu0, info=fw_info)
        _write_mu(state, "bfwa_mu", fw_info["bfwa_mu"], bfwa_persist_dual)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()
        info.update(fw_info)
        info["tau"] = tau

    elif method == "median":
        agg = stack.median(dim=0).values

    elif method == "trimmed_mean":
        # trim at least krum_f (assumed Byzantine count) from each side
        k = max(int(trimmed_beta * K), krum_f)
        k = min(k, (K - 1) // 2)
        srt = stack.sort(dim=0).values
        agg = srt[k:K - k].mean(0) if K - 2 * k > 0 else srt.mean(0)
        info["trimmed"] = k

    elif method == "krum":
        scores = krum_scores(updates, krum_f)
        sel = int(torch.argmin(scores))
        agg = stack[sel]
        w_full = torch.zeros(K, device=stack.device)
        w_full[sel] = 1.0
        info["selected"] = sel
        info["weights"] = w_full.tolist()

    elif method == "multikrum":
        scores = krum_scores(updates, krum_f)
        m = max(1, K - krum_f)
        sel = torch.argsort(scores)[:m].tolist()
        agg = stack[sel].mean(0)
        w_full = torch.zeros(K, device=stack.device)
        w_full[sel] = 1.0 / len(sel)
        info["selected"] = sel
        info["weights"] = w_full.tolist()

    elif method == "fairgfl":
        # FairGFL (Khan-family overlap-aware reweighting): weight ~
        # 1/(1+overlap_ratio). We proxy the paper's node/edge overlap ratio
        # (a multi-graph-federated quantity we don't have) with each client's
        # normalised deviation from the mean sample count -- a client whose
        # local data looks atypical is analogous to one with an atypical
        # overlap profile. See docs/BASELINES_AND_SOURCES.md.
        het = (n - n.mean()).abs() / (n.mean() + 1e-8)
        w = 1.0 / (1.0 + het)
        w = w / w.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "fedgraphfair":
        # FedGraph-Fair (Khan, Information Sciences 2026): the minimax/DRO
        # core -- a simplex-projected dual weight lambda dual-ascended toward
        # clients whose loss exceeds an adaptive cap kappa_max, persisted
        # across rounds via `state`. The paper's personalised-model/graph-
        # mixing layer is not reproduced (single global model here); see
        # docs/BASELINES_AND_SOURCES.md.
        lam = state.get("fedgraphfair_lambda") if state is not None else None
        if lam is None or lam.numel() != K:
            lam = torch.full((K,), 1.0 / K, device=dev)
        lam = lam.to(dev)
        kappa_max = float(loss.mean())
        lam = lam + dual_step * (loss - kappa_max)
        lam = torch.clamp(lam, min=0.0)
        lam = lam / lam.sum() if lam.sum() > 0 else torch.full((K,), 1.0 / K, device=dev)
        if state is not None:
            state["fedgraphfair_lambda"] = lam
        agg = (lam[:, None] * stack).sum(0)
        info["weights"] = lam.tolist()
        info["kappa_max"] = kappa_max

    elif method == "popets_fairfed":
        # PoPETs'25 (Bendoukha et al.): FairFed's weighting made FHE-friendly
        # by replacing exp(-beta|F_i-F_g|) with a degree-2 polynomial (a sign/
        # abs-free surrogate a threshold-CKKS scheme can evaluate). We
        # reproduce only this statistical weighting core; the paper's actual
        # contribution -- a multi-key homomorphic-encryption aggregation
        # protocol -- is systems/crypto infrastructure with no effect on the
        # cleartext numeric result and is not reimplemented (see
        # docs/BASELINES_AND_SOURCES.md).
        F_g = float((n * dpd).sum() / n.sum())
        poly = -fairfed_beta * (dpd - F_g) ** 2 + 1.0
        poly = torch.clamp(poly, min=0.0)
        w_hat = (n / n.sum()) * poly
        w = w_hat / w_hat.sum() if w_hat.sum() > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()
        info["F_g"] = F_g

    elif method == "robust_bfwa":
        # Stage 1: screen out the krum_f updates farthest from the geometric
        # median direction (Byzantine screening).
        med = stack.median(dim=0).values
        dist = ((stack - med) ** 2).sum(1)
        keep = torch.argsort(dist)[: max(1, K - krum_f)].tolist()
        # Stage 2: fairness-constrained Frank-Wolfe among survivors. The dual
        # is keyed separately from plain BFWA: the two rules solve different
        # subproblems (all clients vs survivors) and must not share a
        # multiplier.
        mu0 = _read_mu(state, "robust_bfwa_mu", bfwa_persist_dual)
        fw_info: Dict = {}
        w_sub = bfwa_weights(perf[keep], dpd[keep], tau, fw_iters, dual_step,
                             mu_init=mu0, info=fw_info)
        _write_mu(state, "robust_bfwa_mu", fw_info["bfwa_mu"], bfwa_persist_dual)
        agg = (w_sub[:, None] * stack[keep]).sum(0)
        w_full = torch.zeros(K, device=stack.device)
        w_full[keep] = w_sub
        info["kept"] = keep
        info["survivor_weights"] = w_sub.tolist()
        info["weights"] = w_full.tolist()
        info.update(fw_info)
        info["tau"] = tau

    elif method == "cgsv":
        # CGSV (Xu et al., NeurIPS 2021): cosine-gradient Shapley value. No
        # server validation set -- the reference is the mean client gradient.
        mean_g = stack.mean(0)
        denom = mean_g.norm() + 1e-12
        cos = torch.stack([torch.dot(u, mean_g) / (u.norm() * denom + 1e-12) for u in stack])
        w = torch.relu(cos)
        w = w / w.sum() if float(w.sum()) > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method in ("fu_shapley", "robust_fu_shapley"):
        # FairShare-GNN FU-Shapley: score each client against the server target
        # gradient, EMA-smooth, ReLU-gate onto the simplex. See
        # src/trust/incentive.py and implementation_plan_and_ac_review.md PART B.
        from ..trust.incentive import compute_fu_weights, decompose
        if g_target is None:
            # no server validation nodes this round -> fall back to FedAvg
            # (over the non-null clients: see _zero_null_players)
            w = _zero_null_players(n / n.sum(), stack)
            agg = (w[:, None] * stack).sum(0)
            info["weights"] = w.tolist(); info["fu_fallback"] = "no_target"
        else:
            grads = [u for u in stack]
            phi_ema = state.get("fu_phi_ema") if state is not None else None
            fu_info: Dict = {}
            w, phi_raw, phi_ema_new = compute_fu_weights(
                grads, g_target, phi_ema=phi_ema, beta_ema=fu_beta_ema,
                normalize=fu_normalize, score=fu_score,
                grad_clip=fu_grad_clip, info=fu_info)
            if state is not None:
                state["fu_phi_ema"] = phi_ema_new            # thread EMA across rounds
            # SPEC 4.0(d): the status comes from compute_fu_weights itself now.
            # It must not be re-derived from phi_ema_new here: the D2 guard holds
            # the previous (finite) EMA when a score is non-finite, so a NaN
            # round is no longer visible downstream of the guard.
            info["phi_nan_frac"] = fu_info["phi_nan_frac"]
            info["n_clipped"] = fu_info["n_clipped"]
            info["g_norm_median"] = fu_info["g_norm_median"]
            info["g_norm_max"] = fu_info["g_norm_max"]
            info["phi_norm"] = fu_info["phi_norm"]
            info["n_null"] = fu_info["n_null"]
            if fu_info["fu_status"] != "ok":
                info["fu_fallback"] = fu_info["fu_status"]
            robust_median_fallback = False
            keep = None
            if method == "robust_fu_shapley":
                # F4: median-screen the krum_f farthest updates, then re-gate.
                med = stack.median(dim=0).values
                dist = ((stack - med) ** 2).sum(1)
                keep = torch.argsort(dist)[: max(1, K - krum_f)]
                mask = torch.zeros(K, device=dev); mask[keep] = 1.0
                w = w * mask
                if float(w.sum()) > 0:
                    w = w / w.sum()
                else:
                    # DEGENERATE: the screen kept a survivor set, but the FU gate
                    # scored every survivor <= 0. The old fallback here was
                    # `w = n / n.sum()` -- sample-size weights over ALL K clients
                    # -- which hands weight straight back to the very updates the
                    # distance screen had just flagged as Byzantine, so the
                    # robustness of the rule evaporated exactly in the round it
                    # was needed. Fall back to the coordinate-wise median of the
                    # SURVIVORS instead: it is the standard robust estimator, it
                    # needs no contribution score (which is what just failed),
                    # and a screened client contributes nothing to it.
                    # (Uniform-over-survivors was the alternative; the median is
                    # strictly more robust to an outlier that slipped past the
                    # screen and costs nothing, since no weight vector is needed
                    # downstream.)
                    robust_median_fallback = True
                    info["fu_fallback"] = "robust_screen_degenerate_median"
                info["kept"] = keep.tolist()
            warmup_median = False
            if fu_warmup:
                # Warm-up: keep feeding the EMA (done above) but do not let phi
                # set the weights yet (F9).
                #
                # F25: with fu_warmup_agg="fedavg" this window is an attack
                # surface -- the attacker holds ~1/K for the whole window, and
                # under sign_flip a -10x gradient kills the model before the
                # gate ever engages. "median" needs no score history, so it
                # covers the same window without one. Which value to ship is
                # decided by the D11 warm-up ablation, not asserted here.
                if fu_warmup_agg == "median":
                    warmup_median = True
                else:
                    w = _zero_null_players(n / n.sum(), stack)
                # During warm-up the scores are deliberately not trusted, so an
                # all-zero gate is expected rather than degenerate: the robust
                # median fallback (and its flag) do not apply.
                if robust_median_fallback and not warmup_median:
                    robust_median_fallback = False
                    info.pop("fu_fallback", None)
                info["fu_warmup"] = True
                info["fu_warmup_agg"] = fu_warmup_agg
            if warmup_median:
                # Coordinate-wise median is not expressible as client weights,
                # so aggregate directly and report no weight vector.
                agg = stack.median(dim=0).values
                info["weights"] = None
            elif robust_median_fallback:
                # Median of the SURVIVORS only -- screened clients stay out.
                agg = stack[keep].median(dim=0).values
                info["weights"] = None
            else:
                w = _zero_null_players(w, stack)
                agg = (w[:, None] * stack).sum(0)
                info["weights"] = w.tolist()
            info["phi_raw"] = phi_raw.tolist()
            info["phi_ema"] = phi_ema_new.tolist()
            if g_task is not None and g_fair is not None:
                # The scores were computed on the CLIPPED gradients (SPEC
                # 4.0(b)); decomposing the raw stack instead broke the
                # explainability identity phi_util + phi_fair == phi_raw in
                # every round where the clip bound.
                scored = fu_info.get("grads_scored", grads)
                phi_util, phi_fair = decompose(scored, g_task, g_fair, fu_alpha,
                                               score=fu_score)
                info["phi_util"] = phi_util.tolist()
                info["phi_fair"] = phi_fair.tolist()

    else:
        raise ValueError(f"Unknown aggregator '{method}'")

    return agg.view_as(updates[0]), info


ROBUST_METHODS = {"krum", "multikrum", "median", "trimmed_mean", "robust_bfwa",
                  "robust_fu_shapley"}
FAIR_METHODS = {"bfwa", "fairfed", "qffl", "f2gnn", "fairgfl", "fedgraphfair",
                "popets_fairfed", "fu_shapley", "robust_fu_shapley"}
ALL_METHODS = {"fedavg", "cgsv"} | FAIR_METHODS | ROBUST_METHODS
