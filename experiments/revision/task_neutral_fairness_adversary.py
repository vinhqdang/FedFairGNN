"""T13 -- Task-Neutral Fairness Adversary (P0-A2 / Flaw #1 Resolution).

Evaluates whether the server-side fairness vector (alpha * g_fair) provides
defense against an adversary designed specifically to poison fairness while staying
task-neutral:

    adv_dir = - g_fair_perp + eps * ||g_fair_perp|| * unit(g_task)

Theoretical sign-flip condition:
    <adv, g_target> < 0  <=>  eps < (alpha * ||g_fair_perp||) / (||g_task|| * (1 + alpha * c))
    where c = <g_fair, g_task> / (||g_task|| * ||g_fair_perp||)

Sweeps:
    eps in {0.0025, 0.005, 0.01, 0.015, 0.02, 0.05, 0.10}
    alpha in {0.0, 0.1}
Across German Credit (K=5, R=20, n=20 seeds, CPU).

Outputs:
    results/revision/task_neutral_adversary.json
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
from src.trust.incentive import get_server_target_gradients, get_server_target_gradients_pooled
from src.utils.provenance import build_manifest, resolve_actual_device

DEFAULT_SEEDS = list(range(42, 62))  # 20 seeds
DEFAULT_EPS_LIST = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.05, 0.10]
DEFAULT_ALPHAS = [0.0, 0.1]


def craft_task_neutral_adversary_update(
    benign_updates: List[torch.Tensor],
    g_task: torch.Tensor,
    g_fair: torch.Tensor,
    eps_task: float = 0.02,
) -> torch.Tensor:
    """Construct update orthogonal/positive to g_task but inverted against g_fair:
    adv = -g_fair_perp + eps_task * ||g_fair_perp|| * unit(g_task)
    """
    g_task_flat = g_task.flatten()
    g_fair_flat = g_fair.flatten()

    task_norm_sq = (g_task_flat.norm() ** 2) + 1e-12
    task_proj_coeff = torch.dot(g_fair_flat, g_task_flat) / task_norm_sq
    g_fair_orthogonal = g_fair_flat - task_proj_coeff * g_task_flat
    fair_perp_norm = g_fair_orthogonal.norm() + 1e-12

    unit_task = g_task_flat / (g_task_flat.norm() + 1e-12)
    adv_dir = -g_fair_orthogonal + (eps_task * fair_perp_norm * unit_task)

    # Scale to average benign update norm
    benign_norms = [u.norm().item() for u in benign_updates if u.norm() > 0]
    target_norm = float(np.mean(benign_norms)) if benign_norms else 1.0

    adv_norm = adv_dir.norm() + 1e-12
    adv_update = adv_dir * (target_norm / adv_norm)
    return adv_update.view_as(benign_updates[0])


def run_task_neutral_seed(
    alpha: float, eps: float, seed: int, dataset: str = "german",
    num_clients: int = 5, rounds: int = 20, device: str = "cpu"
) -> Dict:
    t0 = time.perf_counter()
    byz_indices = [0]

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator="fu_shapley",
        attack="none",  # custom injected in round
        num_byzantine=len(byz_indices),
        fu_alpha=alpha,
        fu_grad_clip=10.0,
        dp_enabled=False,
    )

    trainer = FederatedTrainer(cfg)
    per_round_w_adv = []
    per_round_cos_task = []
    per_round_cos_fair = []
    per_round_cos_target = []

    def custom_round(t):
        updates, metas = [], []
        for c in trainer.clients:
            c.set_flat(trainer.global_flat)
            c.train()
            g_k = trainer.global_flat - c.get_flat()
            updates.append(g_k)
            metas.append(c.meta())

        # Derive server target gradients from clean holdout
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        if trainer.server_holdout is not None:
            tg = get_server_target_gradients(
                trainer.ref_model, trainer.server_holdout.to(trainer.device),
                cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)
        else:
            tg = get_server_target_gradients_pooled(
                trainer.ref_model, trainer.clients_data, trainer.device,
                cfg.fu_alpha, fair_surrogate=cfg.fu_fair_surrogate)

        g_target = g_task = g_fair = None
        if tg is not None:
            g_target, g_task, g_fair = (g.cpu() for g in tg)

        # Craft task-neutral fairness attack for byzantine client
        benign_updates = [updates[i] for i in range(num_clients) if i not in byz_indices]
        if g_task is not None and g_fair is not None:
            adv_update = craft_task_neutral_adversary_update(benign_updates, g_task, g_fair, eps_task=eps)
            for b_idx in byz_indices:
                updates[b_idx] = adv_update

            # Log alignment properties
            adv_flat = adv_update.flatten()
            c_task = float(torch.dot(adv_flat, g_task.flatten()) / (adv_flat.norm() * g_task.flatten().norm() + 1e-12))
            c_fair = float(torch.dot(adv_flat, g_fair.flatten()) / (adv_flat.norm() * g_fair.flatten().norm() + 1e-12))
            c_targ = float(torch.dot(adv_flat, g_target.flatten()) / (adv_flat.norm() * g_target.flatten().norm() + 1e-12))
            per_round_cos_task.append(c_task)
            per_round_cos_fair.append(c_fair)
            per_round_cos_target.append(c_targ)

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
            fu_grad_clip=cfg.fu_grad_clip,
            bfwa_persist_dual=cfg.bfwa_persist_dual,
        )

        trainer.global_flat = trainer.global_flat - g_agg

        weights = info.get("weights")
        if weights is not None:
            per_round_w_adv.append(float(sum(weights[i] for i in byz_indices)))
        else:
            per_round_w_adv.append(float("nan"))

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in trainer.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        return rec

    trainer._round = custom_round
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0
    final = res.get("final", {})

    return {
        "alpha": alpha,
        "eps": eps,
        "seed": seed,
        "w_adv": float(np.nanmean(per_round_w_adv)) if per_round_w_adv else float("nan"),
        "cos_task_mean": float(np.nanmean(per_round_cos_task)) if per_round_cos_task else float("nan"),
        "cos_fair_mean": float(np.nanmean(per_round_cos_fair)) if per_round_cos_fair else float("nan"),
        "cos_target_mean": float(np.nanmean(per_round_cos_target)) if per_round_cos_target else float("nan"),
        "auc": float(final.get("auc", float("nan"))),
        "dpd_hard": float(final.get("dpd_hard", float("nan"))),
        "eod": float(final.get("eod", float("nan"))),
        "wall_clock_s": wall_clock_s,
    }


def main():
    parser = argparse.ArgumentParser(description="T13 Task-Neutral Fairness Adversary Suite")
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eps-task", nargs="+", type=float, default=DEFAULT_EPS_LIST,
                        help="Epsilon task margin list, e.g. 0.005 0.01 0.02 0.05 0.10")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
                        help="Alpha fairness weights, e.g. 0.0 0.1")
    parser.add_argument("--smoke", action="store_true", help="Run 1 seed, 2 rounds smoke test with invariant checks")
    parser.add_argument("--output", type=str, default="results/revision/task_neutral_adversary.json")
    args = parser.parse_args()

    seeds = [42] if args.smoke else DEFAULT_SEEDS
    rounds = 2 if args.smoke else args.rounds
    eps_list = args.eps_task
    alphas = args.alphas

    print(f"=== T13 Task-Neutral Fairness Adversary Campaign ===")
    print(f"Dataset: {args.dataset}, Clients: {args.num_clients}, Rounds: {rounds}, Device: {args.device}")
    print(f"Alphas: {alphas}, Epsilons: {eps_list}, Seeds: {len(seeds)}")

    results_grid = {f"alpha_{a}": {f"eps_{e}": [] for e in eps_list} for a in alphas}

    for a in alphas:
        print(f"\n--- Testing Alpha = {a} ---")
        for e in eps_list:
            print(f"  > Epsilon = {e}:")
            for s in seeds:
                res = run_task_neutral_seed(
                    alpha=a, eps=e, seed=s, dataset=args.dataset,
                    num_clients=args.num_clients, rounds=rounds, device=args.device
                )
                results_grid[f"alpha_{a}"][f"eps_{e}"].append(res)
                print(f"    Seed {s:2d} -> w_adv: {res['w_adv']:.4f}, cos_task: {res['cos_task_mean']:+.4f}, cos_fair: {res['cos_fair_mean']:+.4f}, cos_target: {res['cos_target_mean']:+.4f}")

    # If in smoke mode, verify the three invariants
    if args.smoke:
        print("\n" + "=" * 60)
        print("  SMOKE TEST: VALIDATING THREE THEORETICAL INVARIANTS")
        print("=" * 60)
        for e in eps_list:
            res_a0 = results_grid["alpha_0.0"][f"eps_{e}"][0]
            res_a1 = results_grid["alpha_0.1"][f"eps_{e}"][0]

            # Invariant 1: cos(adv, g_task) > 0 and small
            assert res_a0["cos_task_mean"] > 0.0, f"Invariant 1 Failed: cos_task must be > 0, got {res_a0['cos_task_mean']}"
            # Invariant 2: cos(adv, g_fair) < 0 and significant
            assert res_a0["cos_fair_mean"] < -0.10, f"Invariant 2 Failed: cos_fair must be < -0.10, got {res_a0['cos_fair_mean']}"
            print(f"  eps={e}: Invariants 1 & 2 passed (cos_task={res_a0['cos_task_mean']:+.4f}, cos_fair={res_a0['cos_fair_mean']:+.4f})")

        # Invariant 3: cos(adv, g_target) must flip sign for at least one eps between alpha=0.0 and alpha=0.1
        sign_flips = []
        for e in eps_list:
            c0 = results_grid["alpha_0.0"][f"eps_{e}"][0]["cos_target_mean"]
            c1 = results_grid["alpha_0.1"][f"eps_{e}"][0]["cos_target_mean"]
            if c0 > 0 and c1 < 0:
                sign_flips.append(e)
        print(f"  Invariant 3: Epsilon values with sign flip (alpha=0.0 > 0 and alpha=0.1 < 0): {sign_flips}")
        assert len(sign_flips) > 0, (
            f"Invariant 3 Failed: cos(adv, g_target) must flip sign between alpha=0.0 and alpha=0.1 "
            f"for at least one eps in {eps_list}. Details: "
            f"{[(e, results_grid['alpha_0.0'][f'eps_{e}'][0]['cos_target_mean'], results_grid['alpha_0.1'][f'eps_{e}'][0]['cos_target_mean']) for e in eps_list]}"
        )
        print(f"  [CONFIRMED] Critical threshold eps* is bounded around: {sign_flips}")

    summary = {}
    for a in alphas:
        summary[f"alpha_{a}"] = {}
        for e in eps_list:
            items = results_grid[f"alpha_{a}"][f"eps_{e}"]
            w_vals = [r["w_adv"] for r in items if not math.isnan(r["w_adv"])]
            auc_vals = [r["auc"] for r in items if not math.isnan(r["auc"])]
            dpd_vals = [r["dpd_hard"] for r in items if not math.isnan(r["dpd_hard"])]
            cos_targ_vals = [r["cos_target_mean"] for r in items if not math.isnan(r["cos_target_mean"])]

            summary[f"alpha_{a}"][f"eps_{e}"] = {
                "w_adv_mean": float(np.mean(w_vals)) if w_vals else None,
                "w_adv_std": float(np.std(w_vals, ddof=1)) if len(w_vals) > 1 else 0.0,
                "cos_target_mean": float(np.mean(cos_targ_vals)) if cos_targ_vals else None,
                "auc_mean": float(np.mean(auc_vals)) if auc_vals else None,
                "dpd_mean": float(np.mean(dpd_vals)) if dpd_vals else None,
            }

    manifest = build_manifest(dataset=args.dataset, num_seeds=len(seeds), rounds=rounds)
    manifest["device"] = resolve_actual_device(args.device)
    manifest["task"] = "T13_task_neutral_fairness_adversary"
    manifest["eps_sweep"] = eps_list
    manifest["alphas"] = alphas

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"manifest": manifest, "summary": summary, "raw_runs": results_grid}, f, indent=2)

    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
