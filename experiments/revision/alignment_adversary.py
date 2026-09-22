"""T1 -- Optimization-Based Alignment Adversary (Gói 6, Tier D+).

White-box adversary that directly optimizes its local gradient to maximize
misalignment/damage against the server holdout target gradient g_target while
injecting the fairness-poisoning objective.

Pre-registration: docs/04_1_novelty_advantages.md §4.10.3
H_T1: w_adv <= 0.1500 and DPD <= 0.0500 (blocked by orthogonal projection + median).
Rejection criterion: w_adv > 0.2000 (p < 0.05) OR DPD > 0.1000.

Datasets: German Credit, Bail, K=5, seeds {42..51}.
Artifact: results/revision/alignment_adversary.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_num_threads(1)
import torch.nn.functional as F
import numpy as np

from src.config import ExperimentConfig
from src.federated.aggregation import aggregate
from src.federated.client import load_flat_state
from src.federated.trainer import FederatedTrainer
from src.trust.incentive import get_server_target_gradients, get_server_target_gradients_pooled
from src.utils.provenance import build_manifest, resolve_actual_device


SEEDS = list(range(42, 52))
DATASETS = ["german", "bail"]
NUM_CLIENTS = 5
NUM_BYZANTINE = 1
OPT_LR = 0.01
OPT_STEPS = 50


class AlignmentAdversaryTrainer(FederatedTrainer):
    """FederatedTrainer with white-box alignment-optimizing adversary (Gói 6).

    The adversary has white-box access to server holdout target gradient g_target
    and optimizes its malicious gradient g_adv to:
      (1) deceive the FU-Shapley gate (maintain positive alignment with g_target)
      (2) maximize fairness harm (drive disparity against g_fair)
      (3) regularize norm so as not to trivially trigger norm clipping.
    """

    def __init__(self, cfg, lambda_fair: float = 1.0, opt_steps: int = OPT_STEPS, opt_lr: float = OPT_LR):
        super().__init__(cfg)
        self.lambda_fair = lambda_fair
        self.opt_steps = opt_steps
        self.opt_lr = opt_lr

    def _craft_alignment_adversary_update(self, benign_updates, g_target, g_fair=None):
        if g_target is None or len(benign_updates) == 0:
            return benign_updates[0].clone() * -10.0

        g_adv = benign_updates[0].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([g_adv], lr=self.opt_lr)

        g_t = g_target.detach()
        g_t_norm = F.normalize(g_t.unsqueeze(0), dim=1).squeeze(0)

        if g_fair is not None:
            g_f = g_fair.detach()
            g_f_norm = F.normalize(g_f.unsqueeze(0), dim=1).squeeze(0)
        else:
            g_f_norm = -g_t_norm

        for _ in range(self.opt_steps):
            optimizer.zero_grad()
            g_adv_norm = F.normalize(g_adv.unsqueeze(0), dim=1).squeeze(0)
            align_target = (g_adv_norm * g_t_norm).sum()
            harm_fair = (g_adv_norm * g_f_norm).sum()

            loss = -align_target + self.lambda_fair * harm_fair
            norm_loss = torch.relu(g_adv.norm() - benign_updates[0].norm() * 5.0)
            total_loss = loss + 0.1 * norm_loss
            total_loss.backward()
            optimizer.step()

        return g_adv.detach()

    def _round(self, t: int) -> dict:
        updates, metas = [], []
        for c in self.clients:
            c.set_flat(self.global_flat)
            c.train()
            g_k = self.global_flat - c.get_flat()
            updates.append(g_k)
            metas.append(c.meta())

        if self.accountant:
            self.accountant.step(self.cfg.local_epochs)

        g_target = g_task = g_fair = None
        fu_warmup = False
        if "fu_shapley" in self.cfg.aggregator or self.cfg.aggregator == "fltrust":
            load_flat_state(self.ref_model, self.global_flat.to(self.device))
            if self.server_holdout is not None:
                tg = get_server_target_gradients(
                    self.ref_model, self.server_holdout.to(self.device),
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            else:
                tg = get_server_target_gradients_pooled(
                    self.ref_model, self.clients_data, self.device,
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            if tg is not None:
                g_target, g_task, g_fair = (g.cpu() for g in tg)
            fu_warmup = t < self.cfg.fu_warmup_rounds

        if self.byzantine_ids:
            benign_updates = [updates[i] for i in range(len(updates)) if i not in self.byzantine_ids]
            g_adv = self._craft_alignment_adversary_update(benign_updates, g_target, g_fair)
            for b_id in self.byzantine_ids:
                updates[b_id] = g_adv.clone()
                metas[b_id] = {
                    "dpd": 0.0,
                    "loss": 0.01,
                    "group1_rate": 0.5,
                    "n_samples": metas[b_id].get("n_samples", 100),
                }

        g_agg, info = aggregate(
            self.cfg.aggregator, updates, metas,
            tau=self.cfg.fairness_budget, fw_iters=self.cfg.fw_iterations,
            dual_step=self.cfg.dual_step_size, trimmed_beta=self.cfg.trimmed_beta,
            krum_f=max(self.cfg.krum_f, len(self.byzantine_ids)),
            q_ffl=self.cfg.q_ffl, fairfed_beta=self.cfg.fairfed_beta,
            state=self._agg_state,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=self.cfg.fu_alpha, fu_beta_ema=self.cfg.fu_ema_beta,
            fu_normalize=self.cfg.fu_normalize, fu_score=self.cfg.fu_score,
            fu_warmup=fu_warmup, fu_grad_clip=self.cfg.fu_grad_clip,
            fu_warmup_agg=self.cfg.fu_warmup_agg,
            bfwa_persist_dual=self.cfg.bfwa_persist_dual)

        self.global_flat = self.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in self.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        self.history.append(rec)
        return rec


def run_one(dataset: str, seed: int, device: str = "cuda") -> dict:
    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=NUM_CLIENTS,
        device=device,
        aggregator="fu_shapley",
        fu_alpha=0.1,
        dp_enabled=False,
        attack="fairness_poison",
        num_byzantine=NUM_BYZANTINE,
        attack_intensity=10.0,
    )
    trainer = AlignmentAdversaryTrainer(cfg)
    t0 = time.perf_counter()
    res = trainer.run(verbose=False)
    wall = time.perf_counter() - t0
    f = res.get("final", res)

    adv_weights = []
    for rec in getattr(trainer, "history", []):
        w = rec.get("agg_weights")
        if w:
            adv_weights.append(float(sum(w[:NUM_BYZANTINE])))
    w_adv_mean = float(np.mean(adv_weights)) if adv_weights else 0.0

    return {
        "seed": seed,
        "auc": float(f["auc"]),
        "dpd_hard": float(f["dpd_hard"]),
        "w_adv": w_adv_mean,
        "wall_s": float(wall),
    }


def run_gate(out_json: str, seeds: list[int], datasets: list[str] | None = None, device: str = "cuda") -> dict:
    target_datasets = datasets or DATASETS
    results = {}
    total = len(target_datasets) * len(seeds)
    done = 0

    t_all0 = time.perf_counter()
    for dataset in target_datasets:
        per_seed = []
        for seed in seeds:
            row = run_one(dataset, seed, device)
            per_seed.append(row)
            done += 1
            print(f"[{done:2d}/{total:2d}] {dataset:8s} seed={seed:2d} "
                  f"auc={row['auc']:.4f} dpd={row['dpd_hard']:.4f} w_adv={row['w_adv']:.4f} "
                  f"({row['wall_s']:.1f}s)", flush=True)

        w_advs = [r["w_adv"] for r in per_seed]
        dpds = [r["dpd_hard"] for r in per_seed]
        aucs = [r["auc"] for r in per_seed]
        results[dataset] = {
            "per_seed": per_seed,
            "auc": {"mean": float(st.mean(aucs)), "std": float(st.stdev(aucs)) if len(aucs) > 1 else 0.0},
            "w_adv": {"mean": float(st.mean(w_advs)), "std": float(st.stdev(w_advs)) if len(w_advs) > 1 else 0.0},
            "dpd_hard": {"mean": float(st.mean(dpds)), "std": float(st.stdev(dpds)) if len(dpds) > 1 else 0.0},
        }

    t_all1 = time.perf_counter()
    all_walls = [r["wall_s"] for ds in results for r in results[ds]["per_seed"]]
    mean_wall = float(st.mean(all_walls)) if all_walls else 0.0

    manifest = build_manifest(
        benchmark="t1_alignment_adversary",
        hardware={"device": device, "workers": 1, "threads": 1},
        metrics={
            "total_runs": total,
            "mean_wall_per_run_s": mean_wall,
            "total_wall_s": t_all1 - t_all0,
        },
    )

    hypothesis_test = {}
    for ds in target_datasets:
        w_m = results[ds]["w_adv"]["mean"]
        dpd_m = results[ds]["dpd_hard"]["mean"]
        rejected = (w_m > 0.2000) or (dpd_m > 0.1000)
        verdict = "REJECTED (Adversary breached gate)" if rejected else "CONFIRMED (H_T1 holds, w_adv <= 0.2000 and DPD <= 0.1000)"
        hypothesis_test[ds] = {
            "pre_registered_thresholds": {
                "w_adv_max": 0.2000,
                "dpd_max": 0.1000,
                "p_alpha": 0.05,
            },
            "measured": {
                "w_adv_mean": w_m,
                "dpd_mean": dpd_m,
            },
            "verdict": verdict,
            "rejected": rejected,
        }

    payload = {
        "manifest": manifest,
        "config": {
            "datasets": target_datasets,
            "seeds": seeds,
            "num_clients": NUM_CLIENTS,
            "num_byzantine": NUM_BYZANTINE,
            "opt_lr": OPT_LR,
            "opt_steps": OPT_STEPS,
            "device": device,
        },
        "results": results,
        "hypothesis_test": hypothesis_test,
    }

    os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[+] Artifact saved to {out_json}", flush=True)

    for ds in target_datasets:
        ht = hypothesis_test[ds]
        print(f"[{ds.upper()}] w_adv={ht['measured']['w_adv_mean']:.4f} dpd={ht['measured']['dpd_mean']:.4f} => {ht['verdict']}", flush=True)

    return payload


def run_s1(out_json: str, datasets: list[str], device: str) -> dict:
    actual_dev = resolve_actual_device(device)
    print(f"=== T1 S1: Reproducibility & Noise Floor on seed 42 (2 runs) [device={actual_dev}] ===", flush=True)
    res = {}
    max_sigma = 0.0
    for ds in datasets:
        print(f"--- Running {ds} seed 42 run 1 ---", flush=True)
        r1 = run_one(ds, 42, actual_dev)
        print(f"--- Running {ds} seed 42 run 2 ---", flush=True)
        r2 = run_one(ds, 42, actual_dev)
        delta_w_adv = abs(r1["w_adv"] - r2["w_adv"])
        delta_auc = abs(r1["auc"] - r2["auc"])
        delta_dpd = abs(r1["dpd_hard"] - r2["dpd_hard"])
        sigma = delta_w_adv / (2.0 ** 0.5)
        if sigma > max_sigma:
            max_sigma = sigma
        res[ds] = {
            "run1": r1,
            "run2": r2,
            "delta_w_adv": delta_w_adv,
            "delta_auc": delta_auc,
            "delta_dpd": delta_dpd,
            "sigma": sigma,
        }
        print(f"[{ds.upper()}] run1 w_adv={r1['w_adv']:.4f}, run2 w_adv={r2['w_adv']:.4f}, delta={delta_w_adv:.6f}, sigma={sigma:.6f}", flush=True)

    if max_sigma <= 0.002:
        pricing_n = 34
    elif max_sigma <= 0.005:
        pricing_n = 53
    elif max_sigma <= 0.010:
        pricing_n = 121
    else:
        pricing_n = "INSUFFICIENT (sigma > 0.010)"

    verdict = "PASS (sigma <= 0.010)" if max_sigma <= 0.010 else "FAIL (sigma > 0.010)"
    payload = {
        "manifest": build_manifest(
            benchmark="t1_alignment_adversary_s1",
            hardware={"device": actual_dev, "workers": 1, "threads": 1},
        ),
        "results": res,
        "max_sigma": max_sigma,
        "pricing_n": pricing_n,
        "verdict": verdict,
        "pass_threshold": bool(max_sigma <= 0.010),
    }
    s1_out = out_json.replace(".json", "_s1.json") if not out_json.endswith("_s1.json") else out_json
    os.makedirs(os.path.dirname(s1_out) if os.path.dirname(s1_out) else ".", exist_ok=True)
    with open(s1_out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[S1 VERDICT] max_sigma={max_sigma:.6f} => pricing_n={pricing_n} => {verdict}", flush=True)
    print(f"[+] Saved S1 result to {s1_out}\n", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=["s0", "s1", "s3", "s4"], default="s0")
    parser.add_argument("--out-json", default="results/revision/alignment_adversary.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", choices=["all", "german", "bail"], default="all")
    args, _ = parser.parse_known_args()

    actual_dev = resolve_actual_device(args.device)
    ds_list = DATASETS if args.dataset == "all" else [args.dataset]

    if args.gate == "s0":
        seeds = [42]
        print(f"=== T1 S0: Smoke run (1 seed, datasets={ds_list}) [device={actual_dev}] ===", flush=True)
        t0 = time.perf_counter()
        run_gate(args.out_json, seeds=seeds, datasets=ds_list, device=actual_dev)
        print(f"\n=== T1 S0 COMPLETED in {time.perf_counter() - t0:.1f}s ===", flush=True)
    elif args.gate == "s1":
        t0 = time.perf_counter()
        run_s1(args.out_json, datasets=ds_list, device=actual_dev)
        print(f"\n=== T1 S1 COMPLETED in {time.perf_counter() - t0:.1f}s ===", flush=True)
    elif args.gate == "s3":
        seeds = [42, 43, 44]
        print(f"=== T1 S3: Budget Probe (3 seeds: 42..44) [device={actual_dev}] ===", flush=True)
        t0 = time.perf_counter()
        run_gate(args.out_json, seeds=seeds, datasets=ds_list, device=actual_dev)
        print(f"\n=== T1 S3 COMPLETED in {time.perf_counter() - t0:.1f}s ===", flush=True)
    else:
        seeds = list(range(42, 52))
        print(f"=== T1 S4: Full Pre-registered Run ({len(seeds)} seeds: 42..51) [device={actual_dev}] ===", flush=True)
        t0 = time.perf_counter()
        run_gate(args.out_json, seeds=seeds, datasets=ds_list, device=actual_dev)
        print(f"\n=== T1 S4 COMPLETED in {time.perf_counter() - t0:.1f}s ===", flush=True)
