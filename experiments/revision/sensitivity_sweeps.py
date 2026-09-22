"""Dose-response sweeps for the two free knobs the method actually has.

Both sweeps vary exactly one field of ExperimentConfig.canonical() and change
nothing else, so the contrast is attributable to that field alone.

  alpha    fu_alpha, the weight on the server-side fairness gradient in
           g_target = g_task + alpha * g_fair. This is a PRE-REGISTERED test:
           the criterion is written into docs/04 before the run, and a null
           result cuts the claim rather than being reinterpreted. If alpha is
           the fairness dial we say it is, disparity must respond to it
           monotonically; if it does not, we do not have a dial.

  holdout  fu_holdout_size, the number of nodes carved out for the server
           before partitioning. The whole aggregation rule depends on the
           server holding an untainted validation set, which is the assumption
           a referee is most likely to attack, so the paper needs a measured
           answer to "what happens when it is small".

Usage:
  python experiments/revision/sensitivity_sweeps.py --job alpha   --out-json <path>
  python experiments/revision/sensitivity_sweeps.py --job holdout --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.config import ExperimentConfig            # noqa: E402
from src.federated.trainer import FederatedTrainer  # noqa: E402
from src.utils.provenance import build_manifest     # noqa: E402

SEEDS = tuple(range(42, 52))

# Explicit grids. A value outside these raises rather than being routed to a
# default -- invariant R10.
GRIDS = {
    "alpha":   {"field": "fu_alpha",        "values": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]},
    "holdout": {"field": "fu_holdout_size", "values": [50, 100, 200, 400]},
}
METRICS = ("auc", "dpd_hard", "eod", "omega_w")


def _cfg(job: str, value, seed: int, dataset: str = "german"):
    if job not in GRIDS:
        raise KeyError(f"unknown job {job!r}; known: {sorted(GRIDS)}")
    over = {GRIDS[job]["field"]: value, "dataset": dataset}
    if job == "holdout":
        # the holdout only exists when the target is built from it
        over["fu_val_source"] = "server_holdout"
    return ExperimentConfig.canonical(seed=seed, **over)


def run(job: str, out_json: str, seeds=SEEDS, dataset: str = "german"):
    grid = GRIDS[job]
    results = {}
    total = len(grid["values"]) * len(seeds)
    done = 0
    for value in grid["values"]:
        per_seed = []
        for s in seeds:
            tr = FederatedTrainer(_cfg(job, value, s, dataset=dataset))
            fin = tr.run(verbose=False)["final"]
            row = {"seed": s}
            row.update({m: float(fin.get(m, float("nan"))) for m in METRICS})
            row["diverged"] = float(fin.get("diverged", 0.0))
            per_seed.append(row)
            done += 1
            print(f"[{done}/{total}] {grid['field']}={value} seed={s} "
                  f"auc={row['auc']:.4f} dpd={row['dpd_hard']:.4f} eod={row['eod']:.4f}",
                  flush=True)

        def _m(k):
            v = [r[k] for r in per_seed if np.isfinite(r[k])]
            return (float(np.mean(v)) if v else None,
                    float(np.std(v)) if len(v) > 1 else 0.0, len(v))

        entry = {"per_seed": per_seed,
                 "n_diverged": sum(1 for r in per_seed if r["diverged"])}
        for m in METRICS:
            mean, std, n = _m(m)
            entry[f"{m}_mean"], entry[f"{m}_std"], entry[f"{m}_n_valid"] = mean, std, n
        results[str(value)] = entry
        _dump(out_json, job, results, seeds, dataset=dataset)

    _dump(out_json, job, results, seeds, dataset=dataset)
    return results


def _dump(path, job, results, seeds, dataset: str = "german"):
    payload = {
        "manifest": build_manifest(
            experiment=f"sensitivity_sweep_{job}",
            args={"job": job, "field": GRIDS[job]["field"],
                  "values": GRIDS[job]["values"], "seeds": list(seeds),
                  "dataset": dataset,
                  "base": f"ExperimentConfig.canonical() -- {dataset}, K=5, R=20",
                  "note": "exactly one field varies; everything else is canonical"}),
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(GRIDS))
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--dataset", choices=["german", "credit", "bail"], default="german")
    a = ap.parse_args()
    run(a.job, a.out_json, tuple(a.seeds), dataset=a.dataset)


if __name__ == "__main__":
    main()
