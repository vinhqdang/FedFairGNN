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
def bfwa_weights(perf: torch.Tensor, dpd: torch.Tensor, tau: float,
                 iters: int = 20, dual_step: float = 0.1,
                 w_min_factor: float = 0.2) -> torch.Tensor:
    """Solve  max_w  w.perf  s.t.  w.dpd <= tau,  w in simplex  via Frank-Wolfe
    with a dual variable enforcing the fairness budget."""
    K = perf.numel()
    w = torch.full((K,), 1.0 / K)
    mu = 0.0
    for t in range(iters):
        violation = float(torch.dot(w, dpd) - tau)
        grad = -perf + mu * dpd                     # d/dw of Lagrangian
        s = torch.zeros(K)
        s[int(torch.argmin(grad))] = 1.0            # LMO on simplex
        gamma = 2.0 / (t + 2.0)                     # decaying step (spec-compliant)
        w = (1 - gamma) * w + gamma * s
        mu = max(0.0, mu + dual_step * violation)
    # weight floor + renormalise (avoid a single client dominating)
    w = torch.clamp(w, min=w_min_factor / K)
    return w / w.sum()


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
    scores = torch.empty(K)
    for i in range(K):
        di = torch.cat([d[i, :i], d[i, i + 1:]])
        scores[i] = torch.sort(di).values[:m].sum()
    return scores


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def aggregate(method: str, updates: List[torch.Tensor], meta: List[dict],
              *, tau: float = 0.05, fw_iters: int = 20, dual_step: float = 0.1,
              trimmed_beta: float = 0.1, krum_f: int = 1,
              q_ffl: float = 2.0, fairfed_beta: float = 1.0,
              state: Dict = None) -> Tuple[torch.Tensor, Dict]:
    K = len(updates)
    n = torch.tensor([m.get("n", 1) for m in meta], dtype=torch.float32)
    perf = torch.tensor([m.get("perf", 0.5) for m in meta], dtype=torch.float32)
    dpd = torch.tensor([m.get("dpd", 0.0) for m in meta], dtype=torch.float32)
    loss = torch.tensor([m.get("loss", 1.0 - m.get("perf", 0.5)) for m in meta], dtype=torch.float32)
    stack = torch.stack([u.flatten() for u in updates])
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
        gbal = torch.tensor([1.0 - abs(2.0 * m.get("group1_rate", 0.5) - 1.0) for m in meta])
        gamma_f = torch.softmax(-dpd / 0.1, dim=0)
        gamma_e = torch.softmax(gbal / 0.1, dim=0)
        w = torch.softmax((0.5 * gamma_e + gamma_f) / 0.1, dim=0)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "bfwa":
        w = bfwa_weights(perf, dpd, tau, fw_iters, dual_step)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

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
        info["selected"] = sel

    elif method == "multikrum":
        scores = krum_scores(updates, krum_f)
        m = max(1, K - krum_f)
        sel = torch.argsort(scores)[:m].tolist()
        agg = stack[sel].mean(0)
        info["selected"] = sel

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
            lam = torch.full((K,), 1.0 / K)
        kappa_max = float(loss.mean())
        lam = lam + dual_step * (loss - kappa_max)
        lam = torch.clamp(lam, min=0.0)
        lam = lam / lam.sum() if lam.sum() > 0 else torch.full((K,), 1.0 / K)
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
        # Stage 2: fairness-constrained Frank-Wolfe among survivors.
        w_sub = bfwa_weights(perf[keep], dpd[keep], tau, fw_iters, dual_step)
        agg = (w_sub[:, None] * stack[keep]).sum(0)
        info["kept"] = keep
        info["weights"] = w_sub.tolist()

    else:
        raise ValueError(f"Unknown aggregator '{method}'")

    return agg.view_as(updates[0]), info


ROBUST_METHODS = {"krum", "multikrum", "median", "trimmed_mean", "robust_bfwa"}
FAIR_METHODS = {"bfwa", "fairfed", "qffl", "f2gnn", "fairgfl", "fedgraphfair", "popets_fairfed"}
ALL_METHODS = {"fedavg"} | FAIR_METHODS | ROBUST_METHODS
