"""Empirical Measurement of Server Holdout Alignment to Population (T4).

Measures cos(g_target^holdout, g_target^pop) across three graph partition topologies:
    1. Uniform (IID)
    2. Dirichlet (alpha = 0.3, demographic non-IID)
    3. Community (Modularity / Louvain clustering)
on Bail Recidivism across 10 seeds {42..51} with K=5 clients over 15 rounds.

Evaluates pre-registered Hypothesis H_T4:
    cos(community) < cos(dirichlet) <= cos(uniform)
Rejection criterion (pre-registered per Z-1 & 04_1 §4.9.2):
    Reject H_T4 if cos(community) >= cos(uniform) - 0.0500 OR Wilcoxon paired p >= 0.05.

Outputs:
  - results/revision/holdout_cosine.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from scipy import stats
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import load_flat_state
from src.trust.incentive import (get_server_target_gradients,
                                 get_server_target_gradients_pooled)
from src.utils.provenance import build_manifest
from experiments.fairshare_common import partition_edge_retention

PARTITIONS = ["uniform", "dirichlet", "community"]
SEEDS = list(range(42, 52))


def _cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    v1_f = v1.flatten().float()
    v2_f = v2.flatten().float()
    n1 = v1_f.norm()
    n2 = v2_f.norm()
    if float(n1) == 0.0 or float(n2) == 0.0:
        return float("nan")
    cos = (v1_f * v2_f).sum() / (n1 * n2)
    return float(cos.clamp(-1.0, 1.0))


def evaluate_cosine_run(partition: str, seed: int, dataset: str = "bail",
                        num_clients: int = 5, rounds: int = 15,
                        device: str = "cpu") -> dict:
    t0 = time.perf_counter()

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        partition=partition,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator="fu_shapley",
        local_fairness=True,
        dp_enabled=True,
        dp_epsilon=8.0,
        dp_delta=1e-5,
    )

    trainer = FederatedTrainer(cfg)
    edge_ret = partition_edge_retention(trainer)

    cos_trajectory = []
    task_cos_trajectory = []
    fair_cos_trajectory = []

    # Measure cos at round 0 (initialization)
    load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
    tg_hold = get_server_target_gradients(
        trainer.ref_model, trainer.server_holdout.to(trainer.device),
        cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
    tg_pop = get_server_target_gradients_pooled(
        trainer.ref_model, trainer.clients_data, trainer.device,
        cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)

    if tg_hold is not None and tg_pop is not None:
        c_target = _cosine_similarity(tg_hold[0], tg_pop[0])
        c_task = _cosine_similarity(tg_hold[1], tg_pop[1])
        c_fair = _cosine_similarity(tg_hold[2], tg_pop[2])
    else:
        c_target = c_task = c_fair = float("nan")

    init_cos = {"round": 0, "cos_target": c_target, "cos_task": c_task, "cos_fair": c_fair}
    cos_trajectory.append(c_target)
    task_cos_trajectory.append(c_task)
    fair_cos_trajectory.append(c_fair)

    # Hook trainer._round to measure target cosine before every round's aggregation
    orig_round = trainer._round

    def monitored_round(t):
        rec = orig_round(t)
        # Compute cosine at the end of round t
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        h = get_server_target_gradients(
            trainer.ref_model, trainer.server_holdout.to(trainer.device),
            cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
        p = get_server_target_gradients_pooled(
            trainer.ref_model, trainer.clients_data, trainer.device,
            cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
        if h is not None and p is not None:
            c_tar = _cosine_similarity(h[0], p[0])
            c_tsk = _cosine_similarity(h[1], p[1])
            c_far = _cosine_similarity(h[2], p[2])
        else:
            c_tar = c_tsk = c_far = float("nan")
        cos_trajectory.append(c_tar)
        task_cos_trajectory.append(c_tsk)
        fair_cos_trajectory.append(c_far)
        rec["cos_target"] = c_tar
        rec["cos_task"] = c_tsk
        rec["cos_fair"] = c_far
        return rec

    trainer._round = monitored_round
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    valid_cos = [x for x in cos_trajectory if not math.isnan(x)]
    mean_cos = float(np.mean(valid_cos)) if valid_cos else float("nan")

    final = res.get("final", {})
    return {
        "partition": partition,
        "seed": seed,
        "dataset": dataset,
        "rounds": rounds,
        "num_clients": num_clients,
        "edge_retention": float(edge_ret["edge_retention"]),
        "mean_cos_target": mean_cos,
        "init_cos_target": cos_trajectory[0],
        "terminal_cos_target": cos_trajectory[-1],
        "mean_cos_task": float(np.mean([x for x in task_cos_trajectory if not math.isnan(x)])),
        "mean_cos_fair": float(np.mean([x for x in fair_cos_trajectory if not math.isnan(x)])),
        "cos_trajectory": cos_trajectory,
        "auc": float(final.get("auc", float("nan"))),
        "dpd_hard": float(final.get("dpd_hard", float("nan"))),
        "eod": float(final.get("eod", float("nan"))),
        "wall_clock_s": wall_clock_s,
    }


def run_holdout_cosine(out_json: str = "results/revision/holdout_cosine.json",
                       dataset: str = "bail",
                       partitions: List[str] = PARTITIONS,
                       seeds: List[int] = SEEDS,
                       device: str = "cpu"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    records = []
    total = len(partitions) * len(seeds)
    idx = 0

    print(f"[*] Running Holdout Cosine suite ({total} runs)...", flush=True)

    for p in partitions:
        for s in seeds:
            idx += 1
            print(f"[{idx}/{total}] RUNNING: part={p} | seed={s}...", flush=True)
            rec = evaluate_cosine_run(p, s, dataset=dataset, device=device)
            records.append(rec)
            print(f"    -> cos_target={rec['mean_cos_target']:.4f} (init={rec['init_cos_target']:.4f}, term={rec['terminal_cos_target']:.4f}), AUC={rec['auc']:.4f} ({rec['wall_clock_s']:.1f}s)", flush=True)

            out_payload = {
                "manifest": build_manifest(dataset=dataset, rounds=15, device=device),
                "partitions": partitions,
                "seeds": seeds,
                "records": records,
            }
            with open(out_json, "w") as f:
                json.dump(out_payload, f, indent=2)

    # Statistical evaluation
    summary = {}
    for p in partitions:
        matched = [r for r in records if r["partition"] == p]
        summary[p] = {
            "mean_cos_target_mean": float(np.mean([r["mean_cos_target"] for r in matched])),
            "mean_cos_target_std": float(np.std([r["mean_cos_target"] for r in matched])),
            "init_cos_target_mean": float(np.mean([r["init_cos_target"] for r in matched])),
            "init_cos_target_std": float(np.std([r["init_cos_target"] for r in matched])),
            "terminal_cos_target_mean": float(np.mean([r["terminal_cos_target"] for r in matched])),
            "terminal_cos_target_std": float(np.std([r["terminal_cos_target"] for r in matched])),
            "mean_cos_task_mean": float(np.mean([r["mean_cos_task"] for r in matched])),
            "mean_cos_fair_mean": float(np.mean([r["mean_cos_fair"] for r in matched])),
            "edge_retention_mean": float(np.mean([r["edge_retention"] for r in matched])),
            "auc_mean": float(np.mean([r["auc"] for r in matched])),
            "dpd_hard_mean": float(np.mean([r["dpd_hard"] for r in matched])),
        }

    # Hypothesis test: H_T4
    uni_cos = [r["mean_cos_target"] for r in records if r["partition"] == "uniform"]
    com_cos = [r["mean_cos_target"] for r in records if r["partition"] == "community"]
    dir_cos = [r["mean_cos_target"] for r in records if r["partition"] == "dirichlet"]

    diff_com_uni = np.array(com_cos) - np.array(uni_cos)
    mean_diff = float(np.mean(diff_com_uni))
    # Paired Wilcoxon signed-rank test
    w_stat, p_val = stats.wilcoxon(com_cos, uni_cos)

    # Pre-registered rejection condition:
    # Reject H_T4 if cos(community) >= cos(uniform) - 0.0500 OR Wilcoxon p >= 0.05
    # (i.e. if community does NOT drop by at least 0.0500 or drop is not statistically significant)
    delta_cos_margin = 0.0500
    mean_drop = float(np.mean(uni_cos) - np.mean(com_cos))
    reject_h_t4 = bool((mean_drop < delta_cos_margin) or (p_val >= 0.05))

    hypothesis_evaluation = {
        "H_T4": "cos(community) < cos(dirichlet) <= cos(uniform)",
        "mean_cos_uniform": float(np.mean(uni_cos)),
        "mean_cos_dirichlet": float(np.mean(dir_cos)),
        "mean_cos_community": float(np.mean(com_cos)),
        "mean_drop_uniform_minus_community": mean_drop,
        "pre_registered_margin_delta_cos": delta_cos_margin,
        "margin_condition_met": bool(mean_drop >= delta_cos_margin),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(p_val),
        "p_value_significant": bool(p_val < 0.05),
        "verdict_reject_H_T4": reject_h_t4,
        "branch": "Negative Branch (H_T4 Rejected)" if reject_h_t4 else "Positive Branch (H_T4 Confirmed)",
    }

    final_payload = {
        "manifest": build_manifest(dataset=dataset, rounds=15, device=device),
        "partitions": partitions,
        "seeds": seeds,
        "summary": summary,
        "hypothesis_test": hypothesis_evaluation,
        "records": records,
    }
    with open(out_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"[+] Saved Holdout Cosine artifact to {out_json}", flush=True)
    print(f"[*] H_T4 Test Result: mean_drop={mean_drop:.4f}, p={p_val:.4e} -> {hypothesis_evaluation['branch']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Holdout cosine similarity measurement.")
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--partitions", nargs="+", default=["uniform", "dirichlet", "community"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="results/revision/holdout_cosine.json")
    a = ap.parse_args()

    run_holdout_cosine(out_json=a.out_json, dataset=a.dataset,
                       partitions=a.partitions, seeds=a.seeds,
                       device=a.device)


if __name__ == "__main__":
    main()
