"""Experiment runner for the CLAMP paper (groups e1-e6).

Each group expands to a list of jobs (method, dataset, seed, overrides). Jobs run
in a spawn-context process pool, one thread per worker. A job whose result file
already exists is skipped, so an interrupted campaign resumes where it stopped.

Usage:
    python -m experiments.run_clamp_suite --group e1 --workers 8
    python -m experiments.run_clamp_suite --group all --dry-run
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

from experiments.methods import METHODS
from experiments.run_experiment import build_config, run_one, run_id_for
from src.utils.logging_utils import ResultLogger

OUT_ROOT = os.path.join("results", "clamp")

# Shared by every group. Anything a group sets explicitly overrides these.
COMMON = dict(rounds=100, num_clients=5, partition="dirichlet", dirichlet_alpha=0.3,
              fairness_budget=0.03, dp_epsilon=8.0)

ALL_DATASETS = ["german", "bail", "credit", "pokec_z"]

GROUPS = {
    "e1": dict(seeds=range(10), datasets=ALL_DATASETS),
    "e2": dict(seeds=range(5), datasets=["bail", "credit", "pokec_z"]),
    "e3": dict(seeds=range(3), datasets=["bail", "pokec_z"]),
    "e4": dict(seeds=range(3), datasets=["bail", "pokec_z", "credit"]),
    "e5": dict(seeds=range(5), datasets=["german", "bail"]),
    # e5 at a binding fairness budget (tau = 0.005; the 0.03 budget never binds on bail).
    "e5b": dict(seeds=range(5), datasets=["bail"]),
    "e6": dict(seeds=range(3), datasets=["bail", "credit"]),
}

# CLAMP's main method is clamp-f (sign direction, feasible-iterate output). The
# last-iterate methods (clamp, clamp-gcn, clamp-clipmean) are kept as ablations.
MAIN = "clamp-f"
E1_METHODS = ["fedavg-gcn", "fedavg-gat", "fairfed", "fairgnn", "fairsin", "f2gnn",
              "fairgfl", "fedgraphfair", "puffle", "fedfact", "fedfdp",
              "clamp-clipmean", "clamp", "clamp-gcn",
              "clamp-f", "clamp-gcn-f", "clamp-select"]
E3_ATTACKS = ["sign_flip", "scaling", "alie", "ipm", "fairness_poison"]
# (method, extra overrides, label). The median/trimmed_mean/krum rows are FedAvg-GAT
# with the server aggregator replaced.
E3_ROWS = [("fedavg-gat", {}, "fedavg-gat"),
           (MAIN, {}, MAIN),
           # clipped mean (eta = 0, last iterate): the no-controller reference.
           ("clamp-clipmean", {}, "clamp-clipmean"),
           ("fedavg-gat", {"aggregator": "median"}, "fedavg-gat[median]"),
           ("fedavg-gat", {"aggregator": "trimmed_mean"}, "fedavg-gat[trimmed_mean]"),
           ("fedavg-gat", {"aggregator": "krum"}, "fedavg-gat[krum]")]


def _clamp_period(dataset: str) -> int:
    # 10 for the small graphs (german, nba, facebook), 1 otherwise.
    return 10 if dataset in ("german", "nba", "facebook") else 1


def _job(group: str, method: str, dataset: str, seed: int, overrides: Dict) -> Dict:
    ov = dict(COMMON)
    ov.update(overrides)
    if method.startswith("clamp") and "r2_stat_period" not in ov:
        ov["r2_stat_period"] = _clamp_period(dataset)
    # Tag: every override that differs from the shared settings. Two jobs with the
    # same method/dataset/seed then get distinct run ids.
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

    def add(method, dataset, seed, overrides):
        jobs.append(_job(group, method, dataset, seed, overrides))

    for dataset in ds:
        for seed in sd:
            if group == "e1":
                for m in E1_METHODS:
                    add(m, dataset, seed, {})
            elif group == "e2":
                for tau in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
                    add(MAIN, dataset, seed, {"fairness_budget": tau})
            elif group == "e3":
                for m, extra, _ in E3_ROWS:
                    add(m, dataset, seed, {"num_clients": 10, **extra, "attack": "none",
                                           "num_byzantine": 0})
                for attack in E3_ATTACKS:
                    for b in [1, 2, 3, 4]:
                        for m, extra, _ in E3_ROWS:
                            add(m, dataset, seed, {"num_clients": 10, **extra,
                                                   "attack": attack, "num_byzantine": b})
            elif group == "e4":
                add(MAIN, dataset, seed, {"num_clients": 10, "attack": "none",
                                             "num_byzantine": 0})
                for attack in ["stat_falsify", "stat_falsify_mid"]:
                    for b in [1, 2, 3, 4]:
                        add(MAIN, dataset, seed, {"num_clients": 10, "attack": attack,
                                                     "num_byzantine": b})
            elif group == "e5":
                for eps in [1.0, 2.0, 4.0, 8.0, 16.0]:
                    for P in [1, 5, 10]:
                        add(MAIN, dataset, seed, {"dp_epsilon": eps, "r2_stat_period": P})
                add(MAIN, dataset, seed, {"r2_stat_dp": False,
                                             "r2_stat_period": _clamp_period(dataset)})
            elif group == "e5b":
                tau = 0.005
                for eps in [1.0, 2.0, 4.0, 8.0, 16.0]:
                    for P in [1, 5, 10]:
                        add(MAIN, dataset, seed, {"fairness_budget": tau, "dp_epsilon": eps,
                                                     "r2_stat_period": P})
                add(MAIN, dataset, seed, {"fairness_budget": tau, "r2_stat_dp": False,
                                             "r2_stat_period": _clamp_period(dataset)})
                # Unconstrained reference: no Lagrangian term, so tau does not enter.
                add("clamp-clipmean", dataset, seed, {"fairness_budget": tau,
                                                      "r2_stat_period": _clamp_period(dataset)})
            elif group == "e6":
                # One factor at a time around the default (clip 1.0, step 0.5, noise 0).
                triples = [(1.0, 0.5, 0.0), (0.5, 0.5, 0.0), (2.0, 0.5, 0.0),
                           (1.0, 0.1, 0.0), (1.0, 2.0, 0.0),
                           (1.0, 0.5, 0.01), (1.0, 0.5, 0.05)]
                for c, lam, noise in triples:
                    add(MAIN, dataset, seed, {"r2_clip_norm": c, "r2_lambda_step": lam,
                                              "r2_update_noise": noise})
                # Output-rule and direction-rule ablations at the default triple.
                add("clamp", dataset, seed, {"r2_clip_norm": 1.0, "r2_lambda_step": 0.5,
                                             "r2_update_noise": 0.0})
                add("clamp-p", dataset, seed, {"r2_clip_norm": 1.0, "r2_lambda_step": 0.5,
                                               "r2_update_noise": 0.0})
            else:
                raise ValueError(f"unknown group {group!r}")
    return jobs


def _run_job(job: Dict) -> Tuple[str, str, str]:
    """Worker entry point. Returns (run_id, status, error text)."""
    cfg = build_config(job["method"], job["dataset"], job["seed"], **job["overrides"])
    run_id = run_id_for(cfg, job["tag"])
    try:
        logger = ResultLogger(job["out_dir"])
        _, final = run_one(cfg, logger, tag=job["tag"])
        return run_id, ("skipped" if final is None else "ok"), ""
    except Exception:
        err = traceback.format_exc()
        os.makedirs(job["out_dir"], exist_ok=True)
        with open(os.path.join(job["out_dir"], "errors.jsonl"), "a") as f:
            f.write(json.dumps({"run_id": run_id, "error": err}) + "\n")
        return run_id, "error", err


def _init_worker():
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def _parse_seeds(text: str) -> List[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _pending(jobs: List[Dict]) -> List[Dict]:
    out = []
    for job in jobs:
        cfg = build_config(job["method"], job["dataset"], job["seed"], **job["overrides"])
        if not ResultLogger(job["out_dir"]).exists(run_id_for(cfg, job["tag"])):
            out.append(job)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="e1", choices=list(GROUPS) + ["all"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seeds", default=None, help="e.g. 0-9 or 0,2,4 (overrides the group)")
    p.add_argument("--datasets", default=None, help="comma list (overrides the group)")
    p.add_argument("--out", default=OUT_ROOT, help="root; each group writes <out>/<group>")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="run at most K pending jobs")
    args = p.parse_args(argv)

    seeds = _parse_seeds(args.seeds) if args.seeds else None
    datasets = args.datasets.split(",") if args.datasets else None
    groups = list(GROUPS) if args.group == "all" else [args.group]

    jobs: List[Dict] = []
    for g in groups:
        g_jobs = expand(g, datasets, seeds)
        for j in g_jobs:
            j["out_dir"] = os.path.join(args.out, g)
        jobs.extend(g_jobs)
        print(f"group {g}: {len(g_jobs)} jobs")
    print(f"total: {len(jobs)} jobs")

    if args.dry_run:
        for j in jobs[:20]:
            print(f"  {j['group']} {j['method']} {j['dataset']} s{j['seed']} "
                  f"tag={j['tag'] or '-'}")
        return

    pending = _pending(jobs)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(jobs) - len(pending)} already done, {len(pending)} to run "
          f"with {args.workers} workers")
    if not pending:
        return

    ctx = mp.get_context("spawn")
    done, errors, t0 = 0, 0, time.time()
    total = len(pending)
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for run_id, status, _ in pool.imap_unordered(_run_job, pending, chunksize=1):
            done += 1
            errors += status == "error"
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done}/{total}] {status:7s} {run_id}  ETA {eta / 60:.1f} min",
                  flush=True)
    print(f"finished: {done} jobs, {errors} errors "
          f"(see <out>/<group>/errors.jsonl)")


if __name__ == "__main__":
    main()
