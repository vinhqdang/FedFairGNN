"""Ablation grid runner for Issue I2: Decomposing utility and fairness gains.

Evaluates 7 configurations across 3 datasets (pokec_z, credit, bail) x 10 seeds {42..51}:
  C0_FedAvg              -- FedAvg-GCN (baseline non-fair)
  C1_FedAvg_FSER         -- FedAvg-GCN + FSER
  C2_FedAvg_FTGD         -- FedAvg-GCN + FTGD (loss objective, no DP)
  C3_FedAvg_FTGD_DP      -- FedAvg-GCN + FTGD-DP
  C4_BFWA_unconstrained  -- FedAvg-GCN + BFWA (tau -> inf) [Core Control]
  C5_BFWA_constrained    -- FedAvg-GCN + BFWA (tau = 0.05)
  C6_Full_TrustFedGNN    -- TrustFedGNN (FSER + FTGD-DP + BFWA tau=0.05)

Includes per-seed incremental caching, full git provenance, and summary statistics.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated import FederatedTrainer
from src.utils.metrics import weight_oscillation
from experiments.fairshare_common import global_sensitive_homophily


ABLATION_CONFIGS = {
    "C0_FedAvg": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C1_FedAvg_FSER": dict(
        model="trustfedgnn",
        aggregator="fedavg",
        local_fairness=False,
        dp_enabled=False,
    ),
    "C2_FedAvg_FTGD": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=False,
        dp_mode="ftgd",
    ),
    "C3_FedAvg_FTGD_DP": dict(
        model="gcn",
        aggregator="fedavg",
        local_fairness=True,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
    "C4_BFWA_unconstrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=1e6,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C5_BFWA_constrained": dict(
        model="gcn",
        aggregator="bfwa",
        fairness_budget=0.05,
        local_fairness=False,
        dp_enabled=False,
    ),
    "C6_Full_TrustFedGNN": dict(
        model="trustfedgnn",
        aggregator="bfwa",
        fairness_budget=0.05,
        dp_enabled=True,
        dp_mode="ftgd",
        dp_epsilon=8.0,
        dp_delta=1e-5,
    ),
}

DEFAULT_SEEDS = list(range(42, 52))

DATASET_PARAMS = {
    "pokec_z": dict(rounds=50, num_clients=10, dirichlet_alpha=0.3),
    "credit": dict(rounds=50, num_clients=10, dirichlet_alpha=0.3),
    "bail": dict(rounds=50, num_clients=10, dirichlet_alpha=0.3),
}


def get_git_provenance() -> Tuple[str, bool]:
    commit = os.environ.get("FEDFAIR_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    dirty_env = os.environ.get("FEDFAIR_GIT_DIRTY")
    if commit:
        dirty = (dirty_env in ("1", "true", "True"))
        return commit.strip(), dirty
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return c, bool(status)
    except Exception:
        return "unknown", False


def evaluate_single_run(config_name: str, dataset: str, seed: int, device: str) -> dict:
    t0 = time.perf_counter()
    ds_params = DATASET_PARAMS.get(dataset, dict(rounds=50, num_clients=10, dirichlet_alpha=0.3))
    overrides = dict(ABLATION_CONFIGS[config_name])

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        device=device,
        rounds=ds_params["rounds"],
        num_clients=ds_params["num_clients"],
        dirichlet_alpha=ds_params["dirichlet_alpha"],
        **overrides
    )

    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    weights_hist = [r.get("agg_weights") for r in res.get("history", [])]
    omega_w = weight_oscillation(weights_hist)

    # Sensitive homophily h_s of the FULL (pre-partition) graph.
    #
    # This used to be guarded by ``hasattr(trainer, "global_data")``. That
    # attribute does not exist on FederatedTrainer -- it keeps ``clients_data``
    # and ``server_holdout``, never the unpartitioned graph -- so the guard
    # never fired and h_s was recorded as a constant 0.0 for every seed of
    # every run. ``global_sensitive_homophily`` measures it on the same graph
    # the trainer loaded (same dataset/root/seed), which is where a *dataset*
    # property belongs: the induced client subgraphs drop every cross-client
    # edge, so a post-partition h_s would describe the split, not the data.
    h_s = global_sensitive_homophily(cfg)

    final = res["final"]
    return {
        "config_name": config_name,
        "dataset": dataset,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_soft": float(final["dpd_soft"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "omega_w": float(omega_w),
        "pred_std": float(final.get("pred_std", 0.0)),
        "sensitive_homophily": float(h_s),
        "wall_clock_s": float(wall_clock_s),
    }


def main():
    parser = argparse.ArgumentParser(description="Run 7-arm ablation grid across datasets and seeds.")
    parser.add_argument("--datasets", type=str, default="pokec_z,credit,bail",
                        help="Comma-separated datasets (e.g. pokec_z,credit,bail)")
    parser.add_argument("--configs", type=str, default=",".join(ABLATION_CONFIGS.keys()),
                        help="Comma-separated configs")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seed list")
    parser.add_argument("--device", type=str, default=os.environ.get("FEDFAIR_DEVICE", "cpu"),
                        help="Device: cpu or cuda")
    parser.add_argument("--output", type=str, default="results/revision/ablation_grid_results.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = args.device
    out_file = args.output

    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)

    git_commit, git_dirty = get_git_provenance()

    # Load existing results for incremental resumption
    results_store = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                results_store = json.load(f)
        except Exception:
            results_store = {}

    if "_manifest" not in results_store:
        results_store["_manifest"] = {
            "experiment": "ablation_grid_7x3x10",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "device": device,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "configs": ABLATION_CONFIGS,
            "dataset_params": DATASET_PARAMS,
        }

    if "raw_runs" not in results_store:
        results_store["raw_runs"] = []

    # Map existing runs for fast lookup
    completed_keys = set()
    for run in results_store["raw_runs"]:
        completed_keys.add((run["config_name"], run["dataset"], run["seed"]))

    total_tasks = len(datasets) * len(configs) * len(seeds)
    done_count = len(completed_keys)
    print(f"[*] Total planned runs: {total_tasks} | Already completed: {done_count}")

    run_idx = 0
    for ds in datasets:
        for cfg_name in configs:
            for seed in seeds:
                run_idx += 1
                key = (cfg_name, ds, seed)
                if key in completed_keys:
                    print(f"[{run_idx}/{total_tasks}] SKIP (cached): {cfg_name} | {ds} | seed={seed}")
                    continue

                print(f"[{run_idx}/{total_tasks}] RUNNING: {cfg_name} | {ds} | seed={seed} on {device}...", flush=True)
                out = evaluate_single_run(cfg_name, ds, seed, device)
                out["git_commit"] = git_commit
                out["git_dirty"] = git_dirty
                out["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                results_store["raw_runs"].append(out)
                completed_keys.add(key)

                print(f"    -> AUC={out['auc']:.4f}, DPD_hard={out['dpd_hard']:.4f}, EOD={out['eod']:.4f} ({out['wall_clock_s']:.1f}s)", flush=True)

                # Incremental flush to disk
                with open(out_file, "w") as f:
                    json.dump(results_store, f, indent=2)

    # Compute summary statistics
    summary = {}
    for ds in datasets:
        summary[ds] = {}
        for cfg_name in configs:
            runs = [r for r in results_store["raw_runs"] if r["dataset"] == ds and r["config_name"] == cfg_name]
            if not runs:
                continue
            aucs = [r["auc"] for r in runs]
            dpd_hards = [r["dpd_hard"] for r in runs]
            eods = [r["eod"] for r in runs]
            summary[ds][cfg_name] = {
                "n_seeds": len(runs),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "dpd_hard_mean": float(np.mean(dpd_hards)),
                "dpd_hard_std": float(np.std(dpd_hards)),
                "eod_mean": float(np.mean(eods)),
                "eod_std": float(np.std(eods)),
            }
    results_store["summary"] = summary

    with open(out_file, "w") as f:
        json.dump(results_store, f, indent=2)

    print(f"\n[DONE] Saved {len(results_store['raw_runs'])} runs to {out_file}", flush=True)


if __name__ == "__main__":
    main()
