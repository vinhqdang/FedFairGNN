"""T12 -- Adversarial Influence Instrumentation (I_adv) for Two-Tier Defense (P0-A4).

Quantifies adversarial influence vector norm:
    I_adv = || sum_{k in A} w_k * g_k_eff ||_2
where g_k_eff is the effective client payload transmitted to aggregation
(clipped if norm rescaling is active, raw if not).

Evaluates 2x2 factorial arms on German Credit (K=5, 1 Byzantine client):
    - raw: fu_shapley, fu_grad_clip=0.0
    - rescale_only: fu_shapley, fu_grad_clip=10.0
    - median_only: robust_fu_shapley, fu_grad_clip=0.0
    - both: robust_fu_shapley, fu_grad_clip=10.0
Across two threat models:
    - scaling: c = 100.0
    - stealth: rho = 0.85

Outputs:
    results/revision/adv_influence_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.aggregation import aggregate
from src.federated.client import load_flat_state
from src.trust.incentive import (get_server_target_gradients,
                                 get_server_target_gradients_pooled)
from src.utils.provenance import build_manifest, resolve_actual_device
from experiments.revision.adaptive_poisoner import craft_stealth_poison_updates

ARMS = {
    "raw": {"aggregator": "fu_shapley", "clip": 0.0, "label": "Raw Aggregation (no defense)"},
    "rescale_only": {"aggregator": "fu_shapley", "clip": 10.0, "label": "+Norm-Rescaling only"},
    "median_only": {"aggregator": "robust_fu_shapley", "clip": 0.0, "label": "+Median-Screen only"},
    "both": {"aggregator": "robust_fu_shapley", "clip": 10.0, "label": "Dual Shield (+Both)"},
}

ATTACKS = ["scaling", "stealth"]
DEFAULT_SEEDS = list(range(42, 52))


def run_adv_influence_seed(
    arm: str, attack: str, seed: int,
    dataset: str = "german", num_clients: int = 5,
    num_byzantine: int = 1, rounds: int = 20,
    device: str = "cpu"
) -> Dict:
    t0 = time.perf_counter()
    arm_spec = ARMS[arm]
    byz_indices = set(range(num_byzantine))

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator=arm_spec["aggregator"],
        attack="scaling" if attack == "scaling" else "fairness_poison",
        num_byzantine=num_byzantine,
        krum_f=num_byzantine,
        attack_intensity=100.0 if attack == "scaling" else 1.0,
        fu_grad_clip=arm_spec["clip"],
        dp_enabled=False,
    )

    trainer = FederatedTrainer(cfg)
    per_round_i_adv = []
    per_round_w_adv = []

    def custom_round(t):
        updates, metas = [], []
        for c in trainer.clients:
            c.set_flat(trainer.global_flat)
            c.train()
            g_k = trainer.global_flat - c.get_flat()
            updates.append(g_k)
            metas.append(c.meta())

        # Inject adversarial payload
        if attack == "scaling":
            for b_idx in byz_indices:
                updates[b_idx] = updates[b_idx] * 100.0
        elif attack == "stealth":
            updates, metas = craft_stealth_poison_updates(updates, metas, list(byz_indices), rho=0.85)

        # Server target gradient
        g_target = g_task = g_fair = None
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        if trainer.server_holdout is not None:
            tg = get_server_target_gradients(
                trainer.ref_model, trainer.server_holdout.to(trainer.device),
                cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
        else:
            tg = get_server_target_gradients_pooled(
                trainer.ref_model, trainer.clients_data, trainer.device,
                cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
        if tg is not None:
            g_target, g_task, g_fair = (g.cpu() for g in tg)

        g_agg, info = aggregate(
            cfg.aggregator, updates, metas,
            tau=cfg.fairness_budget,
            fw_iters=cfg.fw_iterations,
            dual_step=cfg.dual_step_size,
            krum_f=cfg.krum_f,
            state=trainer._agg_state,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=cfg.fu_alpha, fu_beta_ema=cfg.fu_ema_beta,
            fu_normalize=cfg.fu_normalize, fu_score=cfg.fu_score,
            fu_grad_clip=arm_spec["clip"],
            bfwa_persist_dual=cfg.bfwa_persist_dual,
        )

        trainer.global_flat = trainer.global_flat - g_agg

        # Record I_adv and w_adv
        weights = info.get("weights")
        if weights is not None:
            w_arr = torch.tensor(weights, dtype=torch.float32)
            w_byz = float(sum(w_arr[i].item() for i in byz_indices))
            per_round_w_adv.append(w_byz)

            # Compute effective payload (clipped if arm_spec["clip"] > 0)
            if arm_spec["clip"] > 0.0:
                clipped_updates = []
                for u in updates:
                    unorm = u.norm()
                    if unorm > arm_spec["clip"]:
                        clipped_updates.append(u * (arm_spec["clip"] / (unorm + 1e-12)))
                    else:
                        clipped_updates.append(u)
                eff_stack = torch.stack(clipped_updates)
            else:
                eff_stack = torch.stack(updates)

            byz_term = (w_arr[list(byz_indices), None] * eff_stack[list(byz_indices)]).sum(0)
            i_adv = float(byz_term.norm().item())
            per_round_i_adv.append(i_adv)
        else:
            per_round_w_adv.append(float("nan"))
            per_round_i_adv.append(float("nan"))

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in trainer.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        rec["g_agg_norm"] = float(g_agg.norm()) if g_agg is not None else float("nan")
        return rec

    trainer._round = custom_round
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    final = res.get("final", {})
    return {
        "arm": arm,
        "arm_label": arm_spec["label"],
        "attack": attack,
        "seed": seed,
        "w_adv": float(np.nanmean(per_round_w_adv)) if per_round_w_adv else float("nan"),
        "i_adv": float(np.nanmean(per_round_i_adv)) if per_round_i_adv else float("nan"),
        "auc": float(final.get("auc", float("nan"))),
        "dpd_hard": float(final.get("dpd_hard", float("nan"))),
        "eod": float(final.get("eod", float("nan"))),
        "wall_clock_s": wall_clock_s,
    }


def main():
    parser = argparse.ArgumentParser(description="T12 I_adv instrumentation suite")
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-byzantine", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed, 2 rounds smoke test")
    parser.add_argument("--output", type=str, default="results/revision/adv_influence_results.json")
    args = parser.parse_args()

    seeds = [42] if args.smoke else DEFAULT_SEEDS
    rounds = 2 if args.smoke else args.rounds

    print(f"=== T12 Adversarial Influence Instrumentation (I_adv) ===")
    print(f"Dataset: {args.dataset}, Clients: {args.num_clients}, Byzantine: {args.num_byzantine}")
    print(f"Rounds: {rounds}, Seeds: {len(seeds)}, Device: {args.device}")

    results_matrix = {atk: {arm: [] for arm in ARMS} for atk in ATTACKS}

    for atk in ATTACKS:
        for arm in ARMS:
            print(f"\nEvaluating Attack: {atk:<10} | Arm: {arm:<15}")
            for s in seeds:
                res = run_adv_influence_seed(
                    arm=arm, attack=atk, seed=s,
                    dataset=args.dataset, num_clients=args.num_clients,
                    num_byzantine=args.num_byzantine, rounds=rounds,
                    device=args.device
                )
                results_matrix[atk][arm].append(res)
                print(f"  Seed {s:2d} -> w_adv: {res['w_adv']:.4f}, I_adv: {res['i_adv']:.4f}, AUC: {res['auc']:.4f}")

    summary = {}
    for atk in ATTACKS:
        summary[atk] = {}
        for arm in ARMS:
            items = results_matrix[atk][arm]
            w_vals = [r["w_adv"] for r in items if not math.isnan(r["w_adv"])]
            i_vals = [r["i_adv"] for r in items if not math.isnan(r["i_adv"])]
            auc_vals = [r["auc"] for r in items if not math.isnan(r["auc"])]
            dpd_vals = [r["dpd_hard"] for r in items if not math.isnan(r["dpd_hard"])]

            summary[atk][arm] = {
                "w_adv_mean": float(np.mean(w_vals)) if w_vals else None,
                "w_adv_std": float(np.std(w_vals, ddof=1)) if len(w_vals) > 1 else 0.0,
                "i_adv_mean": float(np.mean(i_vals)) if i_vals else None,
                "i_adv_std": float(np.std(i_vals, ddof=1)) if len(i_vals) > 1 else 0.0,
                "auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
                "dpd_mean": float(np.mean(dpd_vals)) if dpd_vals else None,
            }

    manifest = build_manifest(dataset=args.dataset, num_seeds=len(seeds), rounds=rounds)
    manifest["device"] = resolve_actual_device(args.device)
    manifest["task"] = "T12_I_adv_instrumentation"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"manifest": manifest, "summary": summary, "raw_runs": results_matrix}, f, indent=2)

    print(f"\n[OK] Results written to {args.output}")


if __name__ == "__main__":
    main()
