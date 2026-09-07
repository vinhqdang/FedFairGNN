"""Stage 4.3 Part 3: SOTA Baselines Matrix & Hypothesis H1 Verification on Elliptic Bitcoin (203.8k nodes).

Executes SOTA Baselines across random seeds with K=10 clients:
  1. fedavg-gcn     -- McMahan et al., AISTATS 2017 (Standard non-fair FL baseline)
  2. fairgnn        -- Dai & Wang, WSDM 2021 (Adversarial debiasing on embeddings)
  3. fairsin        -- Yang et al., WWW 2024 (Sensitive Info Neutralization via hetero neighbors)
  4. fairfed        -- Ezzeldin et al., AAAI 2023 (Local fairness feedback FL)
  5. fairgfl        -- Zhou et al., IEEE TPDS 2026 (Overlap-aware aggregation reweighting)
  6. fedgraphfair   -- Khan et al., Information Sciences 2026 (Minimax/DRO dual-ascent reweighting)
  7. cgsv           -- Xu et al., NeurIPS 2021 (Cosine Gradient Shapley Valuation without D_val)
  8. ours-nofser    -- M2 Ablation (w/o FSER) for historical comparison
  9. ours-nofser-true -- Clean FSER ablation (beta=0.0 frozen, identical GAT backbone)
  10. fedfairgnn    -- TrustFedGNN Canonical (FSER + FTGD O(1) DP + FU-Shapley + Two-Tier Defense)

Split protocol:
  Stratified random split within labelled nodes (datasets.py:250-256),
  preserving temporal subgroups across train/val/test splits.

Includes full manifest provenance (device, git_commit, git_dirty, torch version).
Outputs:
  - results/sota_elliptic.json
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import platform
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated import FederatedTrainer
from src.utils.metrics import weight_oscillation
from src.utils.provenance import build_manifest, get_git_info
from experiments.fairshare_common import global_sensitive_homophily
from experiments.methods import METHODS, apply_method


SOTA_BASELINES = [
    "fedavg-gcn",
    "fairgnn",
    "fairsin",
    "fairfed",
    "fairgfl",
    "fedgraphfair",
    "cgsv",
    "ours-nofser",
    "ours-nofser-true",
    "fedfairgnn",
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def evaluate_single_run(method_name: str, seed: int, device: str, rounds: int = 30) -> dict:
    t0 = time.perf_counter()
    cfg = ExperimentConfig.canonical(
        dataset="elliptic",
        seed=seed,
        num_clients=10,
        rounds=rounds,
        dirichlet_alpha=0.3,
        device=device,
    )
    apply_method(cfg, method_name)
    if method_name in ("fedfairgnn", "ours-nofser", "ours-nofser-true"):
        assert cfg.aggregator == "fu_shapley", f"Expected fu_shapley aggregator for {method_name}, got {cfg.aggregator}"

    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    weights_hist = [r.get("agg_weights") for r in res["history"]]
    omega_w = weight_oscillation(weights_hist)
    h_s = global_sensitive_homophily(cfg)

    final = res["final"]
    out_dict = {
        "method": method_name,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_soft": float(final["dpd_soft"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "omega_w": float(omega_w),
        "pred_std": float(final["pred_std"]),
        "sensitive_homophily": float(h_s),
        "wall_clock_s": float(wall_clock_s),
    }

    del trainer, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return out_dict


def aggregate_per_seed(runs: List[dict]) -> dict:
    keys = ["auc", "dpd_soft", "dpd_hard", "eod", "omega_w", "pred_std", "wall_clock_s"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    out["sensitive_homophily"] = float(runs[0].get("sensitive_homophily", 0.0)) if runs else 0.0
    return out


def run_elliptic_suite(seeds=None, baselines=None, rounds: int = 30,
                       out_path: str = "results/sota_elliptic.json") -> dict:
    if seeds is None:
        seeds = DEFAULT_SEEDS[:5]
    if baselines is None:
        baselines = SOTA_BASELINES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit, dirty = get_git_info()

    print("=" * 80, flush=True)
    print("  STAGE 4.3 PART 3: SOTA BASELINES MATRIX ON ELLIPTIC BITCOIN (203.8k NODES)", flush=True)
    print(f"  Device: {device.upper()} | Seeds: {seeds} | Clients: K=10 | Rounds: {rounds} | Alpha_Dir: 0.3", flush=True)
    print(f"  Git Commit: {commit} (dirty: {dirty})", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    results = None
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                candidate = json.load(f)
            candidate_commit = candidate.get("manifest", {}).get("git_commit")
            if candidate_commit == commit:
                results = candidate
                print(f">>> Resuming checkpoint from {out_path} (matching commit {commit}).", flush=True)
        except Exception:
            results = None

    if results is None:
        results = {
            "manifest": build_manifest(
                stage="4.3_part3_elliptic_sota_matrix",
                dataset="elliptic",
                num_nodes=203769,
                num_clients=10,
                rounds=rounds,
                dirichlet_alpha=0.3,
                seeds=seeds,
                baselines=baselines,
                device=device,
            ),
            "baselines": {},
            "raw_runs": {},
        }

    total_runs = len(baselines) * len(seeds)
    run_idx = 0
    start_all = time.perf_counter()

    for b_idx, baseline in enumerate(baselines):
        existing_runs = results.get("raw_runs", {}).get(baseline, [])
        completed_seeds = {r["seed"]: r for r in existing_runs}
        if len(completed_seeds) >= len(seeds) and all(s in completed_seeds for s in seeds):
            print(f"\n>>> [SKIPPING] Baseline: {baseline:<15} already complete with {len(seeds)} seeds.", flush=True)
            run_idx += len(seeds)
            continue

        print(f"\n>>> Running Baseline [{b_idx+1}/{len(baselines)}]: {baseline:<15} ({len(completed_seeds)}/{len(seeds)} cached)", flush=True)
        runs = []
        for seed in seeds:
            run_idx += 1
            if seed in completed_seeds:
                res = completed_seeds[seed]
                runs.append(res)
                print(f"  [{run_idx:02d}/{total_runs:02d}] Method: {baseline:<15} | Seed: {seed} ... (Cached: AUC={res['auc']:.4f}, DPD={res['dpd_hard']:.4f})", flush=True)
            else:
                print(f"  [{run_idx:02d}/{total_runs:02d}] Method: {baseline:<15} | Seed: {seed} ...", end="", flush=True)
                res = evaluate_single_run(baseline, seed, device, rounds=rounds)
                runs.append(res)
                print(f" DONE! AUC={res['auc']:.4f}, DPD={res['dpd_hard']:.4f}, EOD={res['eod']:.4f}, w_osc={res['omega_w']:.4f} ({res['wall_clock_s']:.1f}s)", flush=True)

                results.setdefault("raw_runs", {})[baseline] = runs
                results.setdefault("baselines", {})[baseline] = aggregate_per_seed(runs)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

        results.setdefault("baselines", {})[baseline] = aggregate_per_seed(runs)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.perf_counter() - start_all
    print(f"\n[+] Elliptic suite completed in {total_time/60:.1f} mins. Saved to {out_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SOTA baselines on Elliptic Bitcoin dataset.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS[:5])
    parser.add_argument("--baselines", nargs="+", default=SOTA_BASELINES)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--out-json", default="results/sota_elliptic.json")
    args = parser.parse_args()

    run_elliptic_suite(seeds=args.seeds, baselines=args.baselines, rounds=args.rounds,
                       out_path=args.out_json)


if __name__ == "__main__":
    main()
