"""FS-WI-2: incentive/robustness audit -- does FU-Shapley resist attacks that
capture BFWA, and where does it honestly fail?

Matrix: aggregator x attack x seed on one dataset. For each run we log final
utility/fairness and the fraction of rounds in which a Byzantine client received
non-negligible aggregation weight (attack success proxy).

Expected (plan):
  * fairness_poison / sign_flip: fu_shapley keeps attacker weight ~0, BFWA gets
    captured.
  * scaling / alie: fu_shapley does NOT fully block (||g_k|| inflates a
    positive-aligned score past the ReLU gate, F4) -> robust_fu_shapley does.

    python -m experiments.incentive_audit --dataset german --rounds 20 --seeds 1
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from experiments.fairshare_common import make_trainer


def attacker_weight_fraction(history, byzantine_ids, thresh=1e-3) -> float:
    if not byzantine_ids:
        return 0.0
    hits = 0; tot = 0
    for h in history:
        w = h.get("agg_weights")
        if not w:
            continue
        tot += 1
        if any(w[b] > thresh for b in byzantine_ids if b < len(w)):
            hits += 1
    return round(hits / tot, 4) if tot else 0.0


def attacker_weight_stats(history, byzantine_ids, num_clients) -> dict:
    """Magnitude-level view of what the gate actually did (plan F16).

    ``attacker_weight_fraction`` only counts rounds where the attacker cleared a
    1e-3 threshold, which conflates "gated out" with "kept at a small weight" --
    exactly the ambiguity that made the Phase-3 report state the mechanism
    backwards. These are the quantities that decide the outcome:
      mass  -- mean over rounds of the attacker's total weight (fair share=f/K)
      max   -- worst round
      supp  -- share of rounds the attacker was *fully* zeroed by the ReLU gate
    """
    ws, zero, fb, wu, tot = [], 0, 0, 0, 0
    n_nan_rounds, n_degen = 0, 0
    # Specificity control (Phase 1 checkpoint D2). "atk_w_mass = 0 under sign_flip" is only
    # evidence that phi IDENTIFIES the attacker if the gate does not also zero
    # honest clients. A gate that zeroes somebody every round scores 0.0 on the
    # attacker by reflex, not by detection -- the same number, a different claim.
    ben_zero_rounds, ben_min = 0, []
    gnorm_med, gnorm_max = [], []
    for h in history:
        w = h.get("agg_weights")
        if h.get("fu_fallback"):
            fb += 1
        # Which flavour of fallback: NaN-poisoned phi vs. all-nonpositive phi.
        if str(h.get("fu_fallback", "")).startswith("degenerate"):
            n_degen += 1
        if (h.get("phi_nan_frac") or 0) > 0:
            n_nan_rounds += 1
        if h.get("fu_warmup"):
            wu += 1
        for key, sink in (("g_norm_median", gnorm_med), ("g_norm_max", gnorm_max)):
            v = h.get(key)
            if v is not None and v == v:
                sink.append(float(v))
        if not w:
            continue
        tot += 1
        benign = [w[i] for i in range(len(w)) if i not in byzantine_ids]
        if benign:
            ben_min.append(min(benign))
            if min(benign) <= 1e-12:
                ben_zero_rounds += 1
        if not byzantine_ids:
            continue
        s = sum(w[b] for b in byzantine_ids if b < len(w))
        ws.append(s)
        if s <= 1e-12:
            zero += 1

    def _med(xs):
        return round(sorted(xs)[len(xs) // 2], 4) if xs else float("nan")

    n = len(ws)
    common = {
        # Defined with or without an attacker on purpose: the attack="none" row
        # is the one that makes it interpretable, and that row has no attacker.
        "benign_zeroed_frac": round(ben_zero_rounds / tot, 4) if tot else float("nan"),
        "benign_w_min_med": _med(ben_min),
        "g_norm_median_med": _med(gnorm_med),
        "g_norm_max_med": _med(gnorm_max),
        "fallback_frac": round(fb / max(1, len(history)), 4),
        "warmup_frac": round(wu / max(1, len(history)), 4),
        "nan_round_frac": round(n_nan_rounds / max(1, len(history)), 4),
        "degenerate_frac": round(n_degen / max(1, len(history)), 4),
    }
    if not byzantine_ids:
        return {"atk_w_mass": 0.0, "atk_w_max": 0.0, "atk_w_zeroed_frac": 1.0,
                "fair_share": 0.0, **common}
    return {"atk_w_mass": round(sum(ws) / n, 4) if n else float("nan"),
            "atk_w_max": round(max(ws), 4) if n else float("nan"),
            "atk_w_zeroed_frac": round(zero / n, 4) if n else float("nan"),
            "fair_share": round(len(byzantine_ids) / max(1, num_clients), 4),
            **common}


def dump_trajectory(path, history):
    """Per-round weights / phi so the mechanism can be plotted and audited."""
    keys = ["round", "agg_weights", "phi_raw", "phi_ema", "phi_util", "phi_fair",
            "fu_warmup", "fu_fallback", "phi_nan_frac", "n_clipped", "kept",
            "g_norm_median", "g_norm_max", "phi_norm", "g_auc", "g_dpd", "g_diverged"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for h in history:
            w.writerow({k: h.get(k) for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--num_byzantine", type=int, default=1)
    p.add_argument("--aggregators", nargs="+",
                   default=["bfwa", "fu_shapley", "robust_fu_shapley", "cgsv"])
    p.add_argument("--attacks", nargs="+",
                   default=["none", "fairness_poison", "sign_flip", "scaling"])
    p.add_argument("--out", default="results/fairshare")
    # --- knobs the Phase-1 mechanism arms need to vary -----------------------
    # A1/F10: 'pooled' builds g_target from every client's val nodes INCLUDING
    # the attacker's, which assumes away the threat being measured. Exposed here
    # so the canonical arm can run 'server_holdout' and the delta is measurable
    # rather than asserted.
    p.add_argument("--fu_val_source", default=None, choices=["pooled", "server_holdout"])
    p.add_argument("--fu_grad_clip", type=float, default=None)
    p.add_argument("--fu_alpha", type=float, default=None)
    p.add_argument("--fu_holdout_size", type=int, default=None)
    # Output files are named per-dataset, so two arms sharing --out silently
    # overwrite each other. That is exactly how the RUN-1-A artifact was lost;
    # --tag makes each arm's CSV a distinct file.
    p.add_argument("--tag", default="")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    overrides = {}
    if args.fu_val_source is not None:
        overrides["fu_val_source"] = args.fu_val_source
    if args.fu_grad_clip is not None:
        overrides["fu_grad_clip"] = args.fu_grad_clip
    if args.fu_alpha is not None:
        overrides["fu_alpha"] = args.fu_alpha
    if args.fu_holdout_size is not None:
        overrides["fu_holdout_size"] = args.fu_holdout_size

    rows = []
    for agg in args.aggregators:
        method = {"bfwa": "fedfairgnn-nodp", "fu_shapley": "fairshare",
                  "robust_fu_shapley": "fairshare-robust", "cgsv": "cgsv",
                  "gtg_shapley": "gtg-shapley"}.get(agg, "fairshare")
        for attack in args.attacks:
            for seed in range(args.seeds):
                nb = 0 if attack == "none" else args.num_byzantine
                tr = make_trainer(dataset=args.dataset, seed=seed,
                                  num_clients=args.num_clients, rounds=args.rounds,
                                  method=method, attack=attack, num_byzantine=nb,
                                  **overrides)
                res = tr.run(verbose=False)
                f = res["final"]
                frac = attacker_weight_fraction(res["history"], res["byzantine_ids"])
                st = attacker_weight_stats(res["history"], res["byzantine_ids"],
                                           args.num_clients)
                rows.append({"aggregator": agg, "attack": attack, "seed": seed,
                             "val_source": tr.cfg.fu_val_source,
                             "grad_clip": tr.cfg.fu_grad_clip,
                             "alpha": tr.cfg.fu_alpha,
                             "auc": round(f.get("auc", float("nan")), 4),
                             "dpd": round(f.get("dpd", float("nan")), 4),
                             "eod": round(f.get("eod", float("nan")), 4),
                             "attacker_weight_frac": frac, **st})
                dump_trajectory(os.path.join(
                    args.out,
                    f"audit_traj__{args.dataset}{args.tag}__{agg}__{attack}__s{seed}.csv"),
                    res["history"])
                print(f"{agg:18s} {attack:16s} s{seed} "
                      f"AUC={f.get('auc',0):.3f} DPD={f.get('dpd',0):.3f} "
                      f"atk_w%={frac} mass={st['atk_w_mass']} (fair={st['fair_share']}) "
                      f"zeroed={st['atk_w_zeroed_frac']}")

    path = os.path.join(args.out, f"incentive_audit__{args.dataset}{args.tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
