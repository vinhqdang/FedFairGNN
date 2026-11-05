"""Experiment runner for CLAMP-RT (trimmed gap channel; docs/clamp_rt_design.md).

Same job format and resume behaviour as run_clamp_suite; results go to
results/clamp_rt/<group>. Baseline runs are not repeated: the paper reuses
results/clamp/e1 and results/clamp/e3 for every non-CLAMP-RT method.

Usage:
    python -m experiments.run_clamp_rt_suite --group pilot --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

from experiments import run_clamp_suite as base
from experiments.run_experiment import build_config, run_id_for
from src.utils.logging_utils import ResultLogger

OUT_ROOT = os.path.join("results", "clamp_rt")
COMMON = dict(base.COMMON)
MAIN = "clamp-rt-f"
ALL8 = ["german", "bail", "credit", "pokec_z", "pokec_n", "nba", "facebook", "aminer_s"]
BIG = ["bail", "credit", "pokec_z", "pokec_n"]
GAP_ATTACKS = ["gap_hide", "gap_inflate", "gap_inlier", "gap_zero"]

GROUPS = {
    "pilot": dict(seeds=range(2), datasets=["credit", "pokec_z", "bail"]),
    "pilot2": dict(seeds=range(2), datasets=["credit", "pokec_z", "bail"]),
    "e1": dict(seeds=range(10), datasets=ALL8),
    "e3": dict(seeds=range(3), datasets=["bail", "pokec_z"]),
    "e4": dict(seeds=range(3), datasets=BIG),
    "e5": dict(seeds=range(3), datasets=["credit", "pokec_z", "pokec_n"]),
    "e6": dict(seeds=range(3), datasets=["credit", "pokec_z"]),
}


def period(dataset: str) -> int:
    return base._clamp_period(dataset)


def _job(group, method, dataset, seed, overrides) -> Dict:
    ov = dict(COMMON)
    ov.update(overrides)
    if method.startswith("clamp") and "r2_stat_period" not in ov:
        ov["r2_stat_period"] = period(dataset)
    diff = {k: v for k, v in sorted(ov.items()) if COMMON.get(k, object()) != v}
    tag = "__".join(f"{k}-{v}" for k, v in diff.items())
    return dict(group=group, method=method, dataset=dataset, seed=seed,
                overrides=ov, tag=tag, out_dir=os.path.join(OUT_ROOT, group))


def expand(group: str, datasets: Optional[List[str]] = None,
           seeds: Optional[List[int]] = None) -> List[Dict]:
    spec = GROUPS[group]
    ds = datasets or spec["datasets"]
    sd = list(seeds) if seeds is not None else list(spec["seeds"])
    jobs: List[Dict] = []

    def add(m, d, s, ov):
        jobs.append(_job(group, m, d, s, ov))

    for d in ds:
        for s in sd:
            if group == "pilot":
                for P in (1, 10):
                    add(MAIN, d, s, {"r2_stat_period": P})
                    add(MAIN, d, s, {"r2_stat_period": P, "attack": "gap_hide",
                                     "num_byzantine": 1})
                    add(MAIN, d, s, {"r2_stat_period": P, "r2_stat_dp": False})
            elif group == "pilot2":
                # Floor c = 0.05 and EMA smoothing of the released gap.
                for P in (1, 10):
                    for m in (MAIN, "clamp-rt-ema-f"):
                        add(m, d, s, {"r2_stat_period": P})
                        add(m, d, s, {"r2_stat_period": P, "attack": "gap_hide",
                                      "num_byzantine": 1})
                    add(MAIN, d, s, {"r2_stat_period": P, "r2_stat_dp": False})
            elif group == "e1":
                for m in (MAIN, "clamp-rt-gcn-f"):
                    add(m, d, s, {})
            elif group == "e3":
                add(MAIN, d, s, {"num_clients": 10, "attack": "none", "num_byzantine": 0})
                for attack in base.E3_ATTACKS:
                    for f in (1, 2, 3, 4):
                        add(MAIN, d, s, {"num_clients": 10, "attack": attack,
                                         "num_byzantine": f})
            elif group == "e4":
                for K, fs in ((5, (1, 2)), (10, (1, 2, 3))):
                    add(MAIN, d, s, {"num_clients": K})
                    add("clamp-f", d, s, {"num_clients": K})
                    for f in fs:
                        for a in GAP_ATTACKS:
                            add(MAIN, d, s, {"num_clients": K, "attack": a, "num_byzantine": f})
                        # Old sums channel under the same budget of liars.
                        add("clamp-f", d, s, {"num_clients": K, "attack": "stat_falsify",
                                              "num_byzantine": f})
            elif group == "e5":
                for eps in (1.0, 2.0, 4.0, 8.0):
                    for P in (1, 5, 10):
                        add(MAIN, d, s, {"dp_epsilon": eps, "r2_stat_period": P})
                for P in (1, 10):
                    add(MAIN, d, s, {"r2_stat_dp": False, "r2_stat_period": P})
            elif group == "e6":
                add("clamp-rt-b0-f", d, s, {})
                add(MAIN, d, s, {"r2_trim_b": 2})
                for c in (0.05, 0.2):
                    add(MAIN, d, s, {"r2_floor_frac": c})
                add("clamp-rt", d, s, {})
            else:
                raise ValueError(group)
    return jobs


def _pending(jobs):
    return [j for j in jobs if not ResultLogger(j["out_dir"]).exists(
        run_id_for(build_config(j["method"], j["dataset"], j["seed"], **j["overrides"]),
                   j["tag"]))]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="pilot", choices=list(GROUPS) + ["all"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seeds", default=None)
    p.add_argument("--datasets", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    seeds = base._parse_seeds(a.seeds) if a.seeds else None
    dss = a.datasets.split(",") if a.datasets else None
    groups = [g for g in GROUPS if not g.startswith("pilot")] if a.group == "all" else [a.group]
    jobs = [j for g in groups for j in expand(g, dss, seeds)]
    print(f"total: {len(jobs)} jobs")
    if a.dry_run:
        for j in jobs[:20]:
            print(j["group"], j["method"], j["dataset"], j["seed"], j["tag"])
        return
    pending = _pending(jobs)
    print(f"{len(jobs) - len(pending)} done, {len(pending)} to run")
    if not pending:
        return
    ctx = mp.get_context("spawn")
    t0, n = time.time(), 0
    with ctx.Pool(a.workers, initializer=base._init_worker) as pool:
        for rid, st, _ in pool.imap_unordered(base._run_job, pending, chunksize=1):
            n += 1
            print(f"[{n}/{len(pending)}] {st} {rid} "
                  f"ETA {(time.time() - t0) / n * (len(pending) - n) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
