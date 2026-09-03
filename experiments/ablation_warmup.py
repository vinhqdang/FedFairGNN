"""D11 ablation -- the warm-up row: fu_warmup_rounds x fu_warmup_agg.

    python -m experiments.ablation_warmup --dataset german --seeds 3 --rounds 30

Warm-up is NOT part of the proposed method: it appears nowhere in
``0_incentive_mechanism_proposal.md``. It is an implementation default
(``fu_warmup_rounds=5``) that was never justified, and ``1_5_phase3.md`` already
schedules it as one row of the D11 hyper-parameter ablation, alongside every other
constant. This script runs that row -- nothing more.

Why it matters right now (F25): with ``fu_warmup_agg="fedavg"`` the attacker
holds ~1/K for the whole window, and under ``sign_flip`` a -10x gradient kills
the model at round 2 -- before the ReLU gate ever engages. The gate then works
perfectly from round 6 on a model that is already NaN.

Read the output as an ablation, not as a search for the greenest cell:

  * ``attack="none"`` is not optional. A warm-up rule that buys robustness by
    costing clean AUC has a price, and the price belongs in the table.
  * ``rounds`` must be well above ``max(warmup_rounds)`` or the ablation
    compares "how much of the run was warm-up" rather than the rule itself.
    At the old audit setting (10 rounds, 5 warm-up) half the run was warm-up.
  * ``fu_warmup_agg="median"`` is itself a Byzantine defence, so a win here
    does not attribute to FU-Shapley. The ``median`` aggregator run alone is
    the reference arm that separates the two -- see D12.
"""
from __future__ import annotations

import argparse
import csv
import os

from experiments.fairshare_common import make_trainer
from experiments.incentive_audit import attacker_weight_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="german")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--num_byzantine", type=int, default=1)
    p.add_argument("--warmup_rounds", nargs="+", type=int, default=[0, 3, 5, 10])
    p.add_argument("--warmup_aggs", nargs="+", default=["fedavg", "median"])
    p.add_argument("--attacks", nargs="+",
                   default=["none", "sign_flip", "fairness_poison"])
    p.add_argument("--method", default="fairshare")
    p.add_argument("--out", default="results/fairshare")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for wr in args.warmup_rounds:
        for wa in args.warmup_aggs:
            # warmup_rounds=0 disables the window, so the aggregator choice
            # inside it is vacuous -- run it once, not once per agg.
            if wr == 0 and wa != args.warmup_aggs[0]:
                continue
            for attack in args.attacks:
                for seed in range(args.seeds):
                    nb = 0 if attack == "none" else args.num_byzantine
                    tr = make_trainer(
                        dataset=args.dataset, seed=seed,
                        num_clients=args.num_clients, rounds=args.rounds,
                        method=args.method, attack=attack, num_byzantine=nb,
                        fu_warmup_rounds=wr, fu_warmup_agg=wa)
                    res = tr.run(verbose=False)
                    f = res["final"]
                    st = attacker_weight_stats(res["history"], res["byzantine_ids"],
                                               args.num_clients)
                    row = {
                        "warmup_rounds": wr,
                        "warmup_agg": ("n/a" if wr == 0 else wa),
                        "attack": attack, "seed": seed,
                        "auc": round(f.get("auc", float("nan")), 4),
                        "dpd": round(f.get("dpd", float("nan")), 4),
                        "diverged": f.get("diverged", 0.0),
                        "atk_w_mass": st["atk_w_mass"],
                        "fair_share": st["fair_share"],
                        "nan_round_frac": st["nan_round_frac"],
                    }
                    rows.append(row)
                    print(f"wr={wr:<3}agg={row['warmup_agg']:<8}{attack:<17}s{seed} "
                          f"AUC={row['auc']} DPD={row['dpd']} "
                          f"div={row['diverged']} nan={row['nan_round_frac']} "
                          f"atk_mass={row['atk_w_mass']}", flush=True)

    path = os.path.join(args.out, f"ablation_warmup__{args.dataset}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
