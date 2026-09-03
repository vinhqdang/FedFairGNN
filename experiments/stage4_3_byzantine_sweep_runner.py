"""Stage 4.3 Part 4: Byzantine Ratio Sweep (|B|/K in {0.1, 0.2, 0.3}) & Hypothesis H2 Verification.

Evaluates Two-Tier Defense (M1) vs No Defense (M6 / CGSV) under Byzantine attacks:
  - Ratios: 10% (1/10), 20% (2/10), 30% (3/10) Byzantine clients on K=10
  - Attacks: 'sign_flip' and 'fairness_poisoning'
  - 5 Seeds: {42, 43, 44, 45, 46}
  - Metrics: w_adv (attacker weight share), AUC, DPD_hard, EOD
  - Hypothesis H2 Verification:
      w_adv(M1) < w_adv(M6) and Two-Tier prevents fairness degradation as attack ratio scales.
"""

from __future__ import annotations

import datetime
import gc
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


SEEDS = [42, 43, 44, 45, 46]
BYZ_RATIOS = [0.1, 0.2, 0.3]  # 1/10, 2/10, 3/10
ATTACKS = ["sign_flip", "fairness_poison"]
MODELS = ["fedfairgnn", "m6_no_defense"]


def evaluate_byzantine_run(model_name: str, attack: str, byz_ratio: float, seed: int, device: str) -> dict:
    t0 = time.perf_counter()
    num_byz = int(round(byz_ratio * 10))
    cfg = ExperimentConfig.canonical(
        dataset="german",
        seed=seed,
        num_clients=10,
        rounds=20,
        dirichlet_alpha=0.3,
        device=device,
        attack=attack,
        num_byzantine=num_byz,
        krum_f=num_byz,
    )
    
    # The two arms MUST differ by fields that src/ actually reads.
    #
    # This block previously set `cfg.fu_cosine_filter` and `cfg.fu_multikrum`,
    # neither of which is a field of ExperimentConfig nor read anywhere in src/.
    # Both arms therefore ran the identical canonical config, results agreed to
    # 3-4 decimals across all 30 seed-pairs, and "Hypothesis H2" was scored on
    # CUDA float non-determinism. ExperimentConfig.__setattr__ now rejects
    # undeclared fields so this cannot recur silently.
    #
    # M6 is wired to the definition the repo already codifies for it in
    # tests/test_canonical_config.py and manuscript/tables/ablation.tex --
    # "CGSV Aggregation (No Server Holdout)" = {fu_val_source, fu_score} --
    # rather than to a mechanism that has no implementation.
    #
    # CAVEAT for whoever reports this: M1 (= canonical) uses aggregator
    # "fu_shapley", which has NO Byzantine screen; the distance screen lives in
    # "robust_fu_shapley". So this comparison is score-rule vs score-rule, and
    # calling M1 a "two-tier defense" overclaims. If the intended contrast is
    # screen-vs-no-screen, set aggregator="robust_fu_shapley" for M1 and say so.
    if model_name == "m6_no_defense":
        cfg.fu_val_source = "pooled"     # target built from all clients, Byzantine included
        cfg.fu_score = "cosine"          # CGSV-style norm-invariant credit
    # else: M1 keeps canonical (server_holdout + dot)


    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0
    
    # Calculate attacker weight share across rounds
    byz_indices = list(range(num_byz))
    hist = res["history"]
    adv_weights = []
    for r_entry in hist:
        w_list = r_entry.get("agg_weights")
        if w_list is not None and len(w_list) == 10:
            byz_w = sum(w_list[i] for i in byz_indices)
            adv_weights.append(byz_w)
            
    mean_w_adv = float(np.mean(adv_weights)) if adv_weights else float("nan")  # NaN, not 0.0: an aggregator that exposes no weight vector
    # (coordinate median, trimmed_mean) never populates adv_weights, and a 0.0
    # there reads as a measured "attacker captured nothing". See
    # experiments/revision/adaptive_poisoner.py for the full note.
    final = res["final"]
    
    out_dict = {
        "model": model_name,
        "attack": attack,
        "byz_ratio": byz_ratio,
        "num_byz": num_byz,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_soft": float(final["dpd_soft"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "w_adv": mean_w_adv,
        "wall_clock_s": float(wall_clock_s),
    }
    
    del trainer, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return out_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    commit, dirty = _get_git_info()
    
    print("=" * 80, flush=True)
    print("  STAGE 4.3 PART 4: BYZANTINE RATIO SWEEP & HYPOTHESIS H2 (5 SEEDS)", flush=True)
    print(f"  Device: {device.upper()} | Seeds: {SEEDS} | Ratios: {BYZ_RATIOS} | Attacks: {ATTACKS}", flush=True)
    print(f"  Git Commit: {commit} (dirty: {dirty})", flush=True)
    print("=" * 80, flush=True)
    
    sweep_results = {}
    total_runs = len(ATTACKS) * len(BYZ_RATIOS) * len(MODELS) * len(SEEDS)
    run_idx = 0
    t0 = time.perf_counter()
    
    for attack in ATTACKS:
        sweep_results[attack] = {}
        for ratio in BYZ_RATIOS:
            sweep_results[attack][str(ratio)] = {}
            for model in MODELS:
                print(f"\n>>> Running Attack: {attack:<18} | Ratio: {ratio:.1f} ({int(ratio*10)}/10 Byz) | Model: {model:<15} (5 Seeds)", flush=True)
                runs = []
                for seed in SEEDS:
                    run_idx += 1
                    print(f"  [{run_idx:02d}/{total_runs:02d}] Seed {seed} ...", end="", flush=True)
                    r_res = evaluate_byzantine_run(model, attack, ratio, seed, device)
                    runs.append(r_res)
                    print(f" done! w_adv={r_res['w_adv']:.4f}, AUC={r_res['auc']:.4f}, DPD_hard={r_res['dpd_hard']:.4f} ({r_res['wall_clock_s']:.1f}s)", flush=True)
                    
                w_adv_vals = [r["w_adv"] for r in runs]
                auc_vals = [r["auc"] for r in runs]
                dpd_vals = [r["dpd_hard"] for r in runs]
                eod_vals = [r["eod"] for r in runs]
                
                sweep_results[attack][str(ratio)][model] = {
                    "summary": {
                        "w_adv": {"mean": float(np.mean(w_adv_vals)), "std": float(np.std(w_adv_vals, ddof=1))},
                        "auc": {"mean": float(np.mean(auc_vals)), "std": float(np.std(auc_vals, ddof=1))},
                        "dpd_hard": {"mean": float(np.mean(dpd_vals)), "std": float(np.std(dpd_vals, ddof=1))},
                        "eod": {"mean": float(np.mean(eod_vals)), "std": float(np.std(eod_vals, ddof=1))},
                    },
                    "per_seed": runs,
                }
                
    total_time = time.perf_counter() - t0
    
    # Evaluate Hypothesis H2
    # H2: Two-Tier defense achieves lower w_adv than M6 in >= 4/5 seeds across >= 2/3 ratio levels
    h2_eval = {}
    for attack in ATTACKS:
        r_wins = {}
        for ratio in BYZ_RATIOS:
            m1_runs = sweep_results[attack][str(ratio)]["fedfairgnn"]["per_seed"]
            m6_runs = sweep_results[attack][str(ratio)]["m6_no_defense"]["per_seed"]
            m1_w = [r["w_adv"] for r in m1_runs]
            m6_w = [r["w_adv"] for r in m6_runs]
            wins = int(np.sum(np.array(m1_w) < np.array(m6_w)))
            r_wins[str(ratio)] = f"{wins}/{len(SEEDS)}"
        h2_eval[attack] = r_wins
        
    output = {
        "manifest": {
            "stage": "4.3_part4_byzantine_sweep",
            "dataset": "german",
            "num_clients": 10,
            "seeds": SEEDS,
            "byz_ratios": BYZ_RATIOS,
            "attacks": ATTACKS,
            "models": MODELS,
            "git_commit": commit,
            "git_dirty": dirty,
            "device": device,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "wall_clock_s": total_time,
        },
        "hypothesis_h2_evaluation": h2_eval,
        "results": sweep_results,
    }
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/stage4_3_byzantine_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print("\n" + "=" * 80, flush=True)
    print(f"  STAGE 4.3 PART 4 COMPLETE in {total_time/60:.2f} minutes!", flush=True)
    print(f"  Hypothesis H2 Attacker Weight Share Wins (M1 < M6): {h2_eval}", flush=True)
    print(f"  Results saved to: {out_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
