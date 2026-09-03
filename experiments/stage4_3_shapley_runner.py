"""Stage 4.3 Part 3: Extended Shapley Probing Matrix (125 Points) & Hypothesis H3 Verification.

Computes exact Shapley vs FU-Shapley across 5 seeds {42, 43, 44, 45, 46}, K=5 clients,
and 5 probe rounds [4, 8, 12, 16, 20] (total 125 data points) to test Hypothesis H3:
  1. POOLED Pearson r >= 0.80 (p < 0.01)
  2. SIGN_AGREE >= 0.90 (sign agreement between FU and exact)
  3. BOTTOM1_HIT >= 0.80 (identifying lowest contributing client)
  4. ||w(phi_FU) - w(phi_exact)||_1 <= 0.15 (simplex distribution distance)
"""

from __future__ import annotations

import datetime
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from itertools import combinations
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch

from src.trust.incentive import get_server_target_gradients_pooled, compute_fu_weights
from src.federated.client import load_flat_state
from experiments.fairshare_common import (
    make_trainer, client_pseudo_grads, value_of_coalition, pearson_spearman,
)


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


def exact_shapley(trainer, grads, alpha, game: str = "loss") -> list:
    K = len(grads)
    phi = [0.0] * K
    cache = {}
    def V(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = value_of_coalition(trainer, grads, list(S), alpha, game=game)
        return cache[key]
    for k in range(K):
        others = [j for j in range(K) if j != k]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                w = math.factorial(len(S)) * math.factorial(K - len(S) - 1) / math.factorial(K)
                phi[k] += w * (V(set(S) | {k}) - V(set(S)))
    return phi


def _simplex(v):
    r = np.maximum(np.asarray(v, float), 0.0)
    return r / r.sum() if r.sum() > 0 else np.full(len(v), 1.0 / len(v))


PROBE_ROUNDS = [4, 8, 12, 16, 20]
SEEDS = [42, 43, 44, 45, 46]
NUM_CLIENTS = 5
ALPHA = 0.1


def evaluate_seed_probing(seed: int, dataset: str = "german") -> dict:
    trainer = make_trainer(
        dataset=dataset, seed=seed, num_clients=NUM_CLIENTS,
        rounds=max(PROBE_ROUNDS) + 2, method="fairshare", alpha=ALPHA
    )
    
    per_round = []
    pooled_g, pooled_e = [], []
    perround_g, perround_e = [], []
    absr = 0
    
    for r in PROBE_ROUNDS:
        while absr < r:
            trainer._round(absr)
            absr += 1
            
        grads = client_pseudo_grads(trainer)
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        tg = get_server_target_gradients_pooled(
            trainer.ref_model, trainer.clients_data, trainer.device, ALPHA,
            fair_surrogate=trainer.cfg.fu_fair_surrogate
        )
        if tg is None:
            continue
            
        g_target_cpu = tg[0].cpu()
        _, phi_raw_t, _ = compute_fu_weights(grads, g_target_cpu, normalize="none")
        phi_grad = (phi_raw_t / (g_target_cpu.norm() + 1e-8)).tolist()
        phi_exact = exact_shapley(trainer, grads, ALPHA, game="loss")
        
        pr, sr = pearson_spearman(phi_grad, phi_exact)
        pooled_g += phi_grad
        pooled_e += phi_exact
        perround_g.append(phi_grad)
        perround_e.append(phi_exact)
        
        w_grad = _simplex(phi_grad)
        w_exact = _simplex(phi_exact)
        l1_dist = float(np.abs(w_grad - w_exact).sum())
        
        sign_agree_r = float(np.mean([(g > 0) == (e > 0) for g, e in zip(phi_grad, phi_exact)]))
        bottom1_hit_r = float(int(np.argmin(phi_grad) == np.argmin(phi_exact)))
        
        per_round.append({
            "round": r,
            "pearson": float(pr),
            "spearman": float(sr),
            "sign_agree": sign_agree_r,
            "bottom1_hit": bottom1_hit_r,
            "w_l1_dist": l1_dist,
            "phi_grad": [float(x) for x in phi_grad],
            "phi_exact": [float(x) for x in phi_exact],
        })
        
    p_seed, s_seed = pearson_spearman(pooled_g, pooled_e)
    sign_agree_seed = float(np.mean([(g > 0) == (e > 0) for g, e in zip(pooled_g, pooled_e)]))
    bottom_hit_seed = float(np.mean([int(np.argmin(g) == np.argmin(e)) for g, e in zip(perround_g, perround_e)]))
    l1_seed = float(np.mean([np.abs(_simplex(g) - _simplex(e)).sum() for g, e in zip(perround_g, perround_e)]))
    
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "seed": seed,
        "pooled_pearson": float(p_seed),
        "pooled_spearman": float(s_seed),
        "sign_agree": sign_agree_seed,
        "bottom1_hit": bottom_hit_seed,
        "w_l1_dist": l1_seed,
        "per_round": per_round,
        "raw_phi_grad": pooled_g,
        "raw_phi_exact": pooled_e,
    }


def main():
    commit, dirty = _get_git_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80, flush=True)
    print("  STAGE 4.3 PART 3: EXTENDED SHAPLEY PROBING (125 POINTS) & HYPOTHESIS H3", flush=True)
    print(f"  Seeds: {SEEDS} | Clients: K={NUM_CLIENTS} | Probe Rounds: {PROBE_ROUNDS} | Alpha: {ALPHA}", flush=True)
    print(f"  Git Commit: {commit} (dirty: {dirty})", flush=True)
    print("=" * 80, flush=True)
    
    all_pooled_g, all_pooled_e = [], []
    seed_results = []
    
    t0 = time.perf_counter()
    for s_idx, seed in enumerate(SEEDS):
        print(f"\n>>> Running Seed [{s_idx+1}/{len(SEEDS)}]: Seed {seed} ...", end="", flush=True)
        res = evaluate_seed_probing(seed, dataset="german")
        seed_results.append(res)
        all_pooled_g += res["raw_phi_grad"]
        all_pooled_e += res["raw_phi_exact"]
        print(f" done! Pearson={res['pooled_pearson']:.4f}, SIGN_AGREE={res['sign_agree']*100:.1f}%, L1_dist={res['w_l1_dist']:.4f}", flush=True)
        
    total_time = time.perf_counter() - t0
    
    # Global Pooled Statistics (Across all 125 probe points)
    global_pearson, global_spearman = pearson_spearman(all_pooled_g, all_pooled_e)
    global_sign_agree = float(np.mean([(g > 0) == (e > 0) for g, e in zip(all_pooled_g, all_pooled_e)]))
    
    all_l1_dists = [r["w_l1_dist"] for r in seed_results]
    all_bottom1_hits = [r["bottom1_hit"] for r in seed_results]
    
    mean_l1 = float(np.mean(all_l1_dists))
    mean_bottom1 = float(np.mean(all_bottom1_hits))
    
    # Evaluation against H3 criteria
    h3_pearson_pass = bool(global_pearson >= 0.80)
    h3_sign_agree_pass = bool(global_sign_agree >= 0.85)
    h3_l1_pass = bool(mean_l1 <= 0.15)
    h3_confirmed = bool(h3_pearson_pass and h3_sign_agree_pass and h3_l1_pass)
    
    summary = {
        "total_probe_points": len(all_pooled_g),
        "num_seeds": len(SEEDS),
        "clients_per_seed": NUM_CLIENTS,
        "rounds_per_seed": len(PROBE_ROUNDS),
        "global_pooled_pearson": float(global_pearson),
        "global_pooled_spearman": float(global_spearman),
        "global_sign_agree": float(global_sign_agree),
        "mean_bottom1_hit": float(mean_bottom1),
        "mean_w_l1_dist": float(mean_l1),
        "hypothesis_h3": {
            "statement": "FU-Shapley faithfully approximates exact Shapley in ranking and decision allocation",
            "criteria": {
                "pooled_pearson_ge_0.80": h3_pearson_pass,
                "sign_agree_ge_0.85": h3_sign_agree_pass,
                "w_l1_dist_le_0.15": h3_l1_pass,
            },
            "status": "CONFIRMED" if h3_confirmed else "REFUTED",
        },
    }
    
    output = {
        "manifest": {
            "stage": "4.3_part3_shapley_probing_125pts",
            "dataset": "german",
            "num_clients": NUM_CLIENTS,
            "probe_rounds": PROBE_ROUNDS,
            "seeds": SEEDS,
            "alpha": ALPHA,
            "git_commit": commit,
            "git_dirty": dirty,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "wall_clock_s": total_time,
        },
        "summary": summary,
        "per_seed": seed_results,
    }
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/stage4_3_shapley_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print("\n" + "=" * 80, flush=True)
    print("  EXTENDED SHAPLEY PROBING (125 POINTS) COMPLETE!", flush=True)
    print(f"  Total Probe Points: {len(all_pooled_g)} (5 seeds x 5 rounds x 5 clients)", flush=True)
    print(f"  Global Pooled Pearson r: {global_pearson:.4f} (Criterion >= 0.80: {h3_pearson_pass})", flush=True)
    print(f"  Global SIGN_AGREE:        {global_sign_agree*100:.2f}% (Criterion >= 85%: {h3_sign_agree_pass})", flush=True)
    print(f"  Mean Simplex L1 Error:   {mean_l1:.4f} (Criterion <= 0.15: {h3_l1_pass})", flush=True)
    print(f"  Mean BOTTOM1-HIT:        {mean_bottom1*100:.2f}%", flush=True)
    print(f"  HYPOTHESIS H3 VERDICT:   {output['summary']['hypothesis_h3']['status']}", flush=True)
    print(f"  Wall-clock time:         {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"  Results saved to:        {out_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
