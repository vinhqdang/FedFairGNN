"""Isolate the aggregation rule: identical TrustFedGNN backbone, identical FTGD/DP,
only the server aggregator differs. This is the control the confound objection asks
for -- the Pokec-z decomposition compares a GAT backbone against a GCN one, so it
cannot separate 'our aggregator' from 'our backbone'.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.abspath("."))
from scipy.stats import wilcoxon

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

SEEDS = list(range(42, 52))
ARMS = {
    "M1_fu_shapley": dict(aggregator="fu_shapley"),
    "A0_fedavg": dict(aggregator="fedavg"),
}


def run_aggregator_control(out_file: str = "results/revision/aggregator_control.json",
                           dataset: str = "german",
                           device: str = "cpu"):
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    raw_per_seed = {}
    arms_summary = {}

    for name, over in ARMS.items():
        per = []
        for s in SEEDS:
            cfg = ExperimentConfig.canonical(
                dataset=dataset,
                seed=s,
                num_clients=5,
                rounds=20,
                local_epochs=3,
                dirichlet_alpha=0.3,
                device=device,
                model="trustfedgnn",
                dp_enabled=True,
                dp_mode="ftgd",
                **over,
            )
            r = FederatedTrainer(cfg).run(verbose=False)
            f = r.get("final", r)
            per.append({
                "seed": s,
                "auc": float(f["auc"]),
                "dpd_hard": float(f["dpd_hard"]),
                "eod": float(f["eod"]),
            })
        raw_per_seed[name] = per
        arms_summary[name] = {
            "auc": {"mean": float(st.mean(x["auc"] for x in per)), "std": float(st.stdev(x["auc"] for x in per))},
            "dpd_hard": {"mean": float(st.mean(x["dpd_hard"] for x in per)), "std": float(st.stdev(x["dpd_hard"] for x in per))},
            "eod": {"mean": float(st.mean(x["eod"] for x in per)), "std": float(st.stdev(x["eod"] for x in per))},
        }
        print(f"{name:<20} AUC={arms_summary[name]['auc']['mean']:.4f}±{arms_summary[name]['auc']['std']:.4f}"
              f"  DPD={arms_summary[name]['dpd_hard']['mean']:.4f}")

    a = raw_per_seed["M1_fu_shapley"]
    b = raw_per_seed["A0_fedavg"]
    paired = {}
    for k in ("auc", "dpd_hard", "eod"):
        d = [x[k] - y[k] for x, y in zip(a, b)]
        w_res = wilcoxon(d)
        paired[k] = {
            "mean_delta": float(st.mean(d)),
            "wilcoxon_p": float(w_res.pvalue),
            "wins_m1": int(sum(1 for v in d if v > 0)),
            "wins_a0": int(sum(1 for v in d if v < 0)),
            "ties": int(sum(1 for v in d if v == 0)),
            "per_seed_deltas": [float(v) for v in d],
        }
        print(f"  delta {k:<9} = {paired[k]['mean_delta']:+.4f}   wilcoxon p={paired[k]['wilcoxon_p']:.4f}   wins={paired[k]['wins_m1']}/10")

    manifest = build_manifest(
        experiment="aggregator_isolation_control",
        args={
            "dataset": dataset,
            "num_clients": 5,
            "rounds": 20,
            "local_epochs": 3,
            "seeds": SEEDS,
            "note": "identical trustfedgnn backbone and FTGD/DP in both arms; only cfg.aggregator differs",
        },
    )

    payload = {
        "manifest": manifest,
        "arms": arms_summary,
        "paired_M1_minus_A0": paired,
        "per_seed_raw": raw_per_seed,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote aggregator control results to {out_file}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", default="results/revision/aggregator_control.json")
    parser.add_argument("--dataset", default="german")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_aggregator_control(args.out_file, args.dataset, args.device)
