"""Run a single federated experiment and log its result.

Usage:
    python -m experiments.run_experiment --method fedfairgnn --dataset bail --seed 0
"""
from __future__ import annotations

import argparse
import time

from src.config import ExperimentConfig
from src.federated import FederatedTrainer
from src.utils.logging_utils import ResultLogger
from experiments.methods import apply_method


def build_config(method: str, dataset: str, seed: int, **overrides) -> ExperimentConfig:
    cfg = ExperimentConfig(dataset=dataset, seed=seed)
    apply_method(cfg, method)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def run_one(cfg: ExperimentConfig, logger: ResultLogger, tag: str = "", verbose=False):
    run_id = f"{cfg.exp_name}__{cfg.dataset}__{cfg.aggregator}__{cfg.attack}__b{cfg.num_byzantine}" \
             f"__eps{cfg.dp_epsilon if cfg.dp_enabled else 'inf'}__lam{cfg.fairness_weight}" \
             f"__K{cfg.num_clients}__s{cfg.seed}{('__' + tag) if tag else ''}"
    if logger.exists(run_id):
        return run_id, None
    t = time.time()
    res = FederatedTrainer(cfg).run(verbose=verbose)
    res["final"]["wall_s"] = round(time.time() - t, 1)
    logger.save(run_id, cfg.to_dict(), res)
    return run_id, res["final"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="fedfairgnn")
    p.add_argument("--dataset", default="bail")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--num_clients", type=int, default=None)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    cfg = build_config(args.method, args.dataset, args.seed,
                       rounds=args.rounds, num_clients=args.num_clients)
    logger = ResultLogger(args.out_dir)
    run_id, final = run_one(cfg, logger, verbose=True)
    print(run_id, final)


if __name__ == "__main__":
    main()
