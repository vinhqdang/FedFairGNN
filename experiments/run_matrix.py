"""Full experiment matrix for the manuscript, organised by study.

Every run logs a JSON record (resumable: completed run_ids are skipped). Launch
in the background and let it stream results to results/summary.jsonl.

    python -m experiments.run_matrix --study all
    python -m experiments.run_matrix --study main,ablation,privacy,robustness

Studies: main, ablation, privacy, pareto, robustness, scalability, partition.
"""
from __future__ import annotations

import argparse
import itertools
import time
import traceback

from src.config import ExperimentConfig
from src.utils.logging_utils import ResultLogger
from experiments.methods import (METHODS, apply_method, FAIR_BASELINES,
                                  ROBUST_AGGREGATORS)
from experiments.run_experiment import run_one

# per-dataset training budget (rounds, clients) -- sized for CPU feasibility
ROUNDS = {"german": 60, "bail": 50, "credit": 40, "elliptic": 25, "pokec_z": 30, "synthetic": 40}
CLIENTS = {"german": 3, "bail": 5, "credit": 5, "elliptic": 10, "pokec_z": 10, "synthetic": 5}
SEEDS = [0, 1, 2]
# fewer seeds on the expensive large datasets (still mean+/-std)
DS_SEEDS = {"german": [0, 1, 2], "bail": [0, 1, 2], "credit": [0, 1],
            "elliptic": [0, 1], "pokec_z": [0, 1]}


def cfg_for(method, dataset, seed, **ov):
    c = ExperimentConfig(dataset=dataset, seed=seed,
                         rounds=ROUNDS.get(dataset, 40),
                         num_clients=CLIENTS.get(dataset, 5),
                         local_epochs=2, hidden_channels=64)
    apply_method(c, method)
    for k, v in ov.items():
        setattr(c, k, v)
    return c


def jobs_main():
    # fast datasets first so the core comparison lands early
    methods = FAIR_BASELINES + ["dp-fedavg", "fedfairgnn-nodp", "fedfairgnn", "ours-robust"]
    for ds in ["german", "bail", "credit", "pokec_z"]:
        for m in methods:
            for s in DS_SEEDS[ds]:
                yield cfg_for(m, ds, s), ""
    # Elliptic (large, crypto) -- key methods only
    for m in ["fedavg-gat", "fairsin", "fedfairgnn-nodp", "fedfairgnn", "ours-robust"]:
        for s in DS_SEEDS["elliptic"]:
            yield cfg_for(m, "elliptic", s), ""


def jobs_ablation():
    for ds in ["german", "bail"]:      # 2 datasets suffice for component ablation
        for m in ["fedavg-gat", "ours-nofser", "ours-nobfwa", "fedfairgnn-nodp"]:
            for s in SEEDS:
                yield cfg_for(m, ds, s), "abl"


def jobs_privacy():
    for ds in ["bail"]:                # privacy-utility curve on Bail
        for eps in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            for m in ["fedfairgnn", "dp-fedavg"]:
                yield cfg_for(m, ds, 0, dp_enabled=True, dp_epsilon=eps), f"eps{eps}"


def jobs_pareto():
    for ds in ["german", "bail", "pokec_z"]:
        for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
            yield cfg_for("fedfairgnn-nodp", ds, 0, fairness_weight=lam), f"lam{lam}"


def jobs_robustness():
    ds = "bail"
    K = 6
    # aggregator x attack at fixed 2/K Byzantine
    for agg in ROBUST_AGGREGATORS:
        for atk in ["gaussian", "fairness_poison", "alie"]:
            c = cfg_for("fedfairgnn-nodp", ds, 0, aggregator=agg,
                        attack=atk, num_byzantine=2, num_clients=K)
            yield c, f"rob_{agg}_{atk}"
    # Byzantine-count sweep for the key defenders under gaussian
    for agg in ["fedavg", "bfwa", "krum", "robust_bfwa"]:
        for b in [0, 1, 2, 3]:
            c = cfg_for("fedfairgnn-nodp", ds, 0, aggregator=agg,
                        attack="gaussian" if b else "none", num_byzantine=b, num_clients=K)
            yield c, f"byz_{agg}_{b}"


def jobs_scalability():
    for K in [3, 5, 10, 20]:
        yield cfg_for("fedfairgnn-nodp", "credit", 0, num_clients=K), f"K{K}"


def jobs_partition():
    for part in ["uniform", "dirichlet", "community"]:
        for a in ([0.1, 0.5, 1.0] if part == "dirichlet" else [0.5]):
            c = cfg_for("fedfairgnn-nodp", "bail", 0, partition=part, dirichlet_alpha=a)
            yield c, f"part_{part}_{a}"


def jobs_competitors2026():
    # 2026 competitors on the fast real datasets (+ crypto where feasible)
    for ds in ["german", "bail"]:
        for m in ["favgnn", "fdp-fair"]:
            for s in DS_SEEDS[ds]:
                yield cfg_for(m, ds, s), "c2026"


STUDIES = {
    "main": jobs_main, "ablation": jobs_ablation, "privacy": jobs_privacy,
    "pareto": jobs_pareto, "robustness": jobs_robustness,
    "scalability": jobs_scalability, "partition": jobs_partition,
    "competitors2026": jobs_competitors2026,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="all")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    studies = list(STUDIES) if args.study == "all" else args.study.split(",")

    logger = ResultLogger(args.out_dir)
    jobs = []
    for st in studies:
        jobs.extend(list(STUDIES[st]()))
    print(f"[matrix] {len(jobs)} jobs across studies={studies}", flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for i, (cfg, tag) in enumerate(jobs):
        try:
            run_id, final = run_one(cfg, logger, tag=tag)
            if final is None:
                skipped += 1
            else:
                done += 1
                print(f"[{i+1}/{len(jobs)}] {run_id}  "
                      f"AUC={final.get('auc',0):.3f} DPD={final.get('dpd',0):.3f} "
                      f"EOD={final.get('eod',0):.3f} ({final.get('wall_s')}s) "
                      f"[elapsed {int(time.time()-t0)}s]", flush=True)
        except Exception:
            failed += 1
            print(f"[{i+1}/{len(jobs)}] FAILED {cfg.exp_name}/{cfg.dataset}/{tag}", flush=True)
            traceback.print_exc()
    print(f"[matrix] done={done} skipped={skipped} failed={failed} "
          f"total_time={int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
