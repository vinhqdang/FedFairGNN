"""Stage 4.3 Part 1: SOTA Baselines Matrix on Credit Default (30,000 nodes).

Executes 8 SOTA Baselines across 5 seeds {42, 43, 44, 45, 46} with K=10 clients and 50 rounds:
  1. fedavg-gcn     -- McMahan et al., AISTATS 2017 (Standard non-fair FL baseline)
  2. fairgnn        -- Dai & Wang, WSDM 2021 (Adversarial debiasing on embeddings)
  3. fairsin        -- Yang et al., WWW 2024 (Sensitive Info Neutralization via hetero neighbors)
  4. fairfed        -- Ezzeldin et al., AAAI 2023 (Local fairness feedback FL)
  5. fairgfl        -- Zhou et al., IEEE TPDS 2026 (Overlap-aware aggregation reweighting)
  6. fedgraphfair   -- Khan et al., Information Sciences 2026 (Minimax/DRO dual-ascent reweighting)
  7. cgsv           -- Xu et al., NeurIPS 2021 (Cosine Gradient Shapley Valuation without D_val)
  8. fedfairgnn     -- TrustFedGNN Canonical (FSER + FTGD O(1) DP + FU-Shapley + Two-Tier Defense)

Includes full manifest provenance (device, git_commit, git_dirty, torch version),
dual DPD (soft/hard@0.5), EOD, weight oscillation omega_w, and wall_clock_s.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated import FederatedTrainer
from src.utils.metrics import weight_oscillation
from experiments.fairshare_common import global_sensitive_homophily
from experiments.methods import METHODS, apply_method


def _get_git_info() -> Tuple[str, bool]:
    env_commit = os.environ.get("FEDFAIR_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    env_dirty = os.environ.get("FEDFAIR_GIT_DIRTY")
    if env_commit:
        dirty = (env_dirty == "1" or env_dirty == "true" or env_dirty == "True")
        return env_commit.strip(), dirty
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(status)
        return commit, dirty
    except Exception:
        return "unknown", False


SOTA_BASELINES = [
    "fedavg-gcn",
    "fairgnn",
    "fairsin",
    "fairfed",
    "fairgfl",
    "fedgraphfair",
    "cgsv",
    "ours-nofser",
    "fedfairgnn",
]

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def evaluate_single_run(method_name: str, seed: int, device: str) -> dict:
    t0 = time.perf_counter()
    cfg = ExperimentConfig.canonical(
        dataset="credit",
        seed=seed,
        num_clients=10,
        rounds=50,
        dirichlet_alpha=0.3,
        device=device,
    )
    apply_method(cfg, method_name)
    if method_name in ("fedfairgnn", "ours-nofser"):
        assert cfg.aggregator == "fu_shapley", f"Expected fu_shapley aggregator for {method_name}, got {cfg.aggregator}"
    
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0
    
    weights_hist = [r.get("agg_weights") for r in res["history"]]
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit, dirty = _get_git_info()
    out_path = "results/stage4_3_credit_results.json"
    
    # Only adopt existing per-baseline/per-seed data as resumable progress if
    # it was produced by the exact commit currently checked out -- otherwise
    # it's unrelated (possibly stale/pre-fix) prior content and must not be
    # silently reused (see stage4_remediation_runner for the incident this
    # guards against).
    existing_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                candidate = json.load(f)
            candidate_commit = candidate.get("manifest", {}).get("git_commit")
            if candidate_commit == commit:
                existing_results = candidate
                print(f"Loaded existing results from {out_path} (commit {commit} matches) with baselines: {list(existing_results.get('baselines', {}).keys())}")
            else:
                print(f">>> Existing {out_path} was produced by commit {candidate_commit!r}, not the current commit {commit!r} -- treating as unrelated prior content, starting fresh.", flush=True)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    results = {
        "manifest": {
            "stage": "4.3_part1_credit_sota_matrix",
            "dataset": "credit",
            "num_nodes": 30000,
            "num_clients": 10,
            "rounds": 50,
            "dirichlet_alpha": 0.3,
            "seeds": SEEDS,
            "baselines": SOTA_BASELINES,
            "device": device,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": commit,
            "git_dirty": dirty,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "baselines": existing_results.get("baselines", {}),
        "raw_runs": existing_results.get("raw_runs", {}),
    }
    
    print("=" * 80)
    print("  STAGE 4.3 PART 1: SOTA BASELINES MATRIX ON CREDIT DEFAULT (30k NODES)")
    print(f"  Device: {device.upper()} | Seeds: {SEEDS} | Clients: K=10 | Rounds: 50 | Alpha_Dir: 0.3")
    print(f"  Git Commit: {commit} (dirty: {dirty})")
    print("=" * 80)
    
    start_all = time.perf_counter()
    
    for baseline in SOTA_BASELINES:
        existing_runs = results["raw_runs"].get(baseline, [])
        if baseline in ("cgsv", "ours-nofser") and len(existing_runs) < len(SEEDS):
            existing_runs = []  # re-run fresh 10 seeds for changed baselines!
            
        completed_seeds = {r["seed"]: r for r in existing_runs}
        if len(completed_seeds) == len(SEEDS):
            print(f"\n>>> Baseline {baseline:<15} already completed with {len(SEEDS)} seeds. Skipping.", flush=True)
            continue
            
        print(f"\n>>> Running Baseline: {baseline:<15} ({len(completed_seeds)}/{len(SEEDS)} cached)", flush=True)
        runs = []
        for run_idx, seed in enumerate(SEEDS, 1):
            if seed in completed_seeds:
                res = completed_seeds[seed]
                runs.append(res)
                print(f"  [{run_idx:02d}/{len(SEEDS):02d}] Method: {baseline:<15} | Seed: {seed} ... (Cached: AUC={res['auc']:.4f}, DPD={res['dpd_hard']:.4f})", flush=True)
            else:
                print(f"  [{run_idx:02d}/{len(SEEDS):02d}] Method: {baseline:<15} | Seed: {seed} ...", end="", flush=True)
                res = evaluate_single_run(baseline, seed, device)
                runs.append(res)
                print(f" done! AUC={res['auc']:.4f}, DPD_hard={res['dpd_hard']:.4f}, EOD={res['eod']:.4f} ({res['wall_clock_s']:.1f}s)", flush=True)
                
                # Save incremental checkpoint immediately after every seed
                agg = aggregate_per_seed(runs)
                results["baselines"][baseline] = {
                    "summary": agg,
                    "per_seed": runs,
                }
                results["raw_runs"][baseline] = runs
                os.makedirs("results", exist_ok=True)
                with open("results/stage4_3_credit_results.json", "w") as f:
                    json.dump(results, f, indent=2)
            
        agg = aggregate_per_seed(runs)
        results["baselines"][baseline] = {
            "summary": agg,
            "per_seed": runs,
        }
        results["raw_runs"][baseline] = runs
        print(f"  ---> Summary for {baseline}: AUC = {agg['auc']['mean']:.4f} ± {agg['auc']['std']:.4f} | DPD_hard = {agg['dpd_hard']['mean']:.4f} ± {agg['dpd_hard']['std']:.4f} | EOD = {agg['eod']['mean']:.4f} ± {agg['eod']['std']:.4f}", flush=True)
        
        # Save checkpoint after each completed baseline
        os.makedirs("results", exist_ok=True)
        with open("results/stage4_3_credit_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Checkpoint saved to results/stage4_3_credit_results.json]", flush=True)

    total_time = time.perf_counter() - start_all
    results["manifest"]["total_wall_clock_s"] = float(total_time)
    results["manifest"]["seeds"] = SEEDS
    
    # Assert manifest seeds matches actual completed runs
    for b_name, b_data in results["baselines"].items():
        assert len(results["manifest"]["seeds"]) == len(b_data["per_seed"]), (
            f"Manifest seeds count {len(results['manifest']['seeds'])} != {b_name} runs count {len(b_data['per_seed'])}"
        )
    
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"  STAGE 4.3 PART 1 COMPLETE in {total_time/60:.2f} minutes!")
    print(f"  Results written to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
