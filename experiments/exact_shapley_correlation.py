"""FS-WI-6: does the O(K*P) gradient FU-Shapley approximate the exact Shapley?

For K=4 the exact Shapley value is tractable (2^4 = 16 coalitions). At a given
round we (1) extract the K client pseudo-gradients, (2) compute the gradient
FU-Shapley phi_k = <g_k, g_target>, and (3) compute the exact Shapley value of
the cooperative game V(S) = utility(S) - alpha*disparity(S) evaluated on the
pooled validation set. We then correlate the two. This is the *primary* evidence
that replaces the (deliberately downgraded) Taylor bound -- see plan finding F8.

    python -m experiments.exact_shapley_correlation --dataset german --rounds 8

Gate (plan): Pearson >= 0.85 on the majority of probed rounds.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from itertools import combinations

from src.trust.incentive import get_server_target_gradients_pooled, compute_fu_weights
from experiments.fairshare_common import (
    make_trainer, client_pseudo_grads, value_of_coalition, pearson_spearman,
)


def exact_shapley(trainer, grads, alpha, game: str = "loss") -> list:
    K = len(grads)
    phi = [0.0] * K
    # precompute V for every subset (memoised by frozenset)
    cache = {}
    def V(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = value_of_coalition(trainer, grads, list(S), alpha, game=game)
        return cache[key]
    for k in range(K):
        others = [j for j in range(K) if j != k]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                w = math.factorial(len(S)) * math.factorial(K - len(S) - 1) / math.factorial(K)
                phi[k] += w * (V(set(S) | {k}) - V(set(S)))
    return phi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=8)      # (legacy; ignored if --probe_at set)
    p.add_argument("--probe_rounds", type=int, default=3)  # (legacy)
    p.add_argument("--probe_at", nargs="+", type=int, default=[8, 12, 16, 20, 24],
                   help="absolute mid-training rounds to probe (where gradients still "
                        "carry signal -- NOT at convergence where g->0 makes phi noise)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--game", default="loss", choices=["loss", "auc"],
                   help="cooperative-game value function. loss = -(wBCE+alpha*softDPD), "
                        "consistent with the gradient phi (plan B1/D9). auc = legacy "
                        "AUC-alpha*DPD used by the original Phase-3 run.")
    p.add_argument("--normalize", default="target_norm",
                   choices=["none", "target_norm"],
                   help="must match the aggregator's setting when pooling across "
                        "rounds, otherwise POOLED mixes per-round scales (F12)")
    p.add_argument("--out", default="results/fairshare")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from src.federated.client import load_flat_state
    probe_at = sorted(args.probe_at)
    trainer = make_trainer(dataset=args.dataset, seed=args.seed, num_clients=4,
                           rounds=max(probe_at) + 2, method="fairshare", alpha=args.alpha)

    rows = []
    pooled_g, pooled_e = [], []
    perround_g, perround_e = [], []
    absr = 0
    for r in probe_at:
        # Advance to absolute round r under the real FU-Shapley dynamics (absolute
        # counter -> warm-up gating stays correct), then probe MID-TRAINING: the
        # first-order gradient/exact-SV agreement needs gradients with signal, so
        # we probe before convergence (where g->0 turns phi into pure noise) and
        # after the earliest nonlinear rounds. Per-round n=K=4 is too small to be
        # stable, so the headline number is the POOLED correlation below.
        while absr < r:
            trainer._round(absr); absr += 1
        grads = client_pseudo_grads(trainer)
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        tg = get_server_target_gradients_pooled(
            trainer.ref_model, trainer.clients_data, trainer.device, args.alpha,
            fair_surrogate=trainer.cfg.fu_fair_surrogate)
        if tg is None:
            continue
        # F12: normalise exactly like the aggregator, otherwise the POOLED
        # correlation below mixes incomparable per-round scales.
        g_target_cpu = tg[0].cpu()
        _, phi_raw_t, _ = compute_fu_weights(grads, g_target_cpu, normalize="none")
        if args.normalize == "target_norm":
            phi_grad = (phi_raw_t / (g_target_cpu.norm() + 1e-8)).tolist()
        else:
            phi_grad = phi_raw_t.tolist()
        phi_exact = exact_shapley(trainer, grads, args.alpha, game=args.game)
        pr, sr = pearson_spearman(phi_grad, phi_exact)
        pooled_g += phi_grad; pooled_e += phi_exact
        perround_g.append(phi_grad); perround_e.append(phi_exact)
        rows.append({"round": r, "pearson": round(pr, 4), "spearman": round(sr, 4),
                     "phi_grad": [round(x, 4) for x in phi_grad],
                     "phi_exact": [round(x, 5) for x in phi_exact]})
        print(f"round {r:3d}: pearson={pr:.3f} spearman={sr:.3f}")

    import numpy as _np
    ppool, spool = pearson_spearman(pooled_g, pooled_e)
    # Time-averaged per-client correlation: this is the METHOD-CONSISTENT test --
    # the aggregator scores clients with the EMA-smoothed phi, not a single round.
    # Per-round n=K=4 is noisy and occasionally sign-flips (precisely why the
    # method smooths), but the accumulated contribution ranks clients correctly.
    gbar = _np.array(perround_g).mean(0).tolist() if perround_g else []
    ebar = _np.array(perround_e).mean(0).tolist() if perround_e else []
    pta, sta = pearson_spearman(gbar, ebar) if gbar else (float("nan"), float("nan"))
    perr = [rw["pearson"] for rw in rows]
    print(f"POOLED (n={len(pooled_g)}): pearson={ppool:.3f} spearman={spool:.3f}")
    print(f"TIME-AVG per-client (EMA-consistent): pearson={pta:.3f} spearman={sta:.3f}")
    print(f"per-round: median pearson={_np.median(perr):.3f}  frac>0.7={_np.mean([p>0.7 for p in perr]):.2f}")
    rows.append({"round": "POOLED", "pearson": round(ppool, 4), "spearman": round(spool, 4),
                 "phi_grad": "", "phi_exact": ""})
    rows.append({"round": "TIME_AVG", "pearson": round(pta, 4), "spearman": round(sta, 4),
                 "phi_grad": [round(x, 5) for x in gbar], "phi_exact": [round(x, 5) for x in ebar]})

    # --- decision-relevant agreement (plan F14): the ReLU gate acts on the SIGN
    # of phi and the simplex on its rank, so value-correlation is not the
    # quantity the mechanism actually uses. Report both.
    sign_agree = float(_np.mean([(g > 0) == (e > 0)
                                 for g, e in zip(pooled_g, pooled_e)])) if pooled_g else float("nan")
    bottom_hit = float(_np.mean([int(_np.argmin(g) == _np.argmin(e))
                                 for g, e in zip(perround_g, perround_e)])) if perround_g else float("nan")
    def _simplex(v):
        r = _np.maximum(_np.asarray(v, float), 0.0)
        return r / r.sum() if r.sum() > 0 else _np.full(len(v), 1.0 / len(v))
    wl1 = float(_np.mean([_np.abs(_simplex(g) - _simplex(e)).sum()
                          for g, e in zip(perround_g, perround_e)])) if perround_g else float("nan")
    print(f"SIGN-AGREE={sign_agree:.3f}  BOTTOM1-HIT={bottom_hit:.3f}  ||w(phi)-w(SV)||_1={wl1:.3f}")
    rows.append({"round": "SIGN_AGREE", "pearson": round(sign_agree, 4),
                 "spearman": round(bottom_hit, 4),
                 "phi_grad": f"bottom1_hit={bottom_hit:.4f}", "phi_exact": f"w_l1={wl1:.4f}"})

    path = os.path.join(args.out,
                        f"exact_sv_corr__{args.dataset}__s{args.seed}__{args.game}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "pearson", "spearman", "phi_grad", "phi_exact"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}  (POOLED pearson={ppool:.3f} spearman={spool:.3f})")


if __name__ == "__main__":
    main()
