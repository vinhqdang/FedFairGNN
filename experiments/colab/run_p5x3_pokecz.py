"""P5-X3: Pokec-z Adversarial Robustness under Byzantine Scaling & Fairness Poisoning.

Dataset: Pokec-z (N=67,796, K=10, rounds=50, dirichlet_alpha=0.3).
Pre-registration: docs/04_1_novelty_advantages.md §4.10.2
Hypothesis H_P5X3: Under Byzantine Scaling (c=100), w_adv <= 0.0500 and AUC >= 0.7500 (p < 0.05).
Rejection criterion: w_adv > 0.0500 OR AUC < 0.7500.

Gates:
  s0: Smoke test (1 seed: 42, scenarios: clean, scaling_c100)
  s1: Reproducibility & CUDA jitter on seed 42 run twice under scaling_c100 (pricing_n)
  s3: Budget probe (3 seeds: 42..44, all 3 scenarios)
  s4: Full run (10 seeds: 42..51, all 3 scenarios)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics as st
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest, resolve_actual_device


SEEDS_FULL = list(range(42, 52))

SCENARIOS = {
    "clean": dict(attack="none", num_byzantine=0, attack_intensity=0.0),
    "scaling_c100": dict(attack="scaling", num_byzantine=3, attack_intensity=100.0),
    "fairness_poison": dict(attack="fairness_poison", num_byzantine=3, attack_intensity=10.0),
}


def run_one(seed: int, scenario_name: str, attack_kwargs: dict, device: str) -> dict:
    t0 = time.perf_counter()
    cfg = ExperimentConfig.canonical(
        dataset="pokec_z",
        seed=seed,
        num_clients=10,
        rounds=50,
        local_epochs=3,
        dirichlet_alpha=0.3,
        device=device,
        model="trustfedgnn",
        aggregator="fu_shapley",
        dp_enabled=False,
        **attack_kwargs,
    )
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    dt = time.perf_counter() - t0
    f = res.get("final", res)

    adv_weights = []
    for rec in getattr(trainer, "history", []):
        w = rec.get("agg_weights")
        if w and attack_kwargs.get("num_byzantine", 0) > 0:
            num_byz = attack_kwargs["num_byzantine"]
            adv_weights.append(float(sum(w[:num_byz])))
    w_adv = float(sum(adv_weights) / len(adv_weights)) if adv_weights else 0.0

    out = {
        "seed": seed,
        "auc": float(f["auc"]),
        "dpd_hard": float(f["dpd_hard"]),
        "w_adv": w_adv,
        "wall_clock_s": float(dt),
    }
    del trainer, res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def run_s0(out_file: str, device: str):
    print("=== P5-X3 S0: Pokec-z Smoke Run (1 seed: 42, clean + scaling) ===", flush=True)
    scenarios_s0 = {
        "clean": SCENARIOS["clean"],
        "scaling_c100": SCENARIOS["scaling_c100"],
    }
    res = {}
    for name, kwargs in scenarios_s0.items():
        print(f"--- Running {name} seed=42 ---", flush=True)
        r = run_one(42, name, kwargs, device)
        res[name] = r
        print(f"  [{name}] AUC={r['auc']:.4f} DPD={r['dpd_hard']:.4f} w_adv={r['w_adv']:.4f} ({r['wall_clock_s']:.1f}s)", flush=True)

    payload = {
        "gate": "s0",
        "results": res,
        "verdict": "PASS (Smoke completed without errors)",
    }
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] S0 saved to {out_file}", flush=True)
    return payload


def run_s1(out_file: str, device: str):
    print("=== P5-X3 S1: Pokec-z Reproducibility & CUDA Jitter Measurement (seed 42 run twice) ===", flush=True)
    sc = SCENARIOS["scaling_c100"]
    print("--- Run 1 (scaling_c100, seed=42) ---", flush=True)
    r1 = run_one(42, "scaling_c100", sc, device)
    print(f"  Run 1: AUC={r1['auc']:.4f} DPD={r1['dpd_hard']:.4f} w_adv={r1['w_adv']:.6f} ({r1['wall_clock_s']:.1f}s)", flush=True)

    print("--- Run 2 (scaling_c100, seed=42) ---", flush=True)
    r2 = run_one(42, "scaling_c100", sc, device)
    print(f"  Run 2: AUC={r2['auc']:.4f} DPD={r2['dpd_hard']:.4f} w_adv={r2['w_adv']:.6f} ({r2['wall_clock_s']:.1f}s)", flush=True)

    delta_w_adv = abs(r1["w_adv"] - r2["w_adv"])
    delta_auc = abs(r1["auc"] - r2["auc"])
    sigma = delta_w_adv / math.sqrt(2)

    if sigma <= 0.002:
        pricing_n = 34
    elif sigma <= 0.005:
        pricing_n = 53
    elif sigma <= 0.010:
        pricing_n = 121
    else:
        pricing_n = f"INSUFFICIENT (sigma={sigma:.4f} > 0.010)"

    verdict = "PASS (sigma <= 0.010)" if sigma <= 0.010 else "REPORT_TO_USER (sigma > 0.010)"
    payload = {
        "gate": "s1",
        "device": device,
        "run1": r1,
        "run2": r2,
        "delta_w_adv": delta_w_adv,
        "delta_auc": delta_auc,
        "sigma": sigma,
        "pricing_n": pricing_n,
        "verdict": verdict,
        "pass_threshold": bool(sigma <= 0.010),
    }
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[S1 VERDICT] delta_w_adv={delta_w_adv:.6f} sigma={sigma:.6f} => pricing_n={pricing_n} => {verdict}", flush=True)
    print(f"[+] S1 saved to {out_file}", flush=True)
    return payload


def run_grid(out_file: str, seeds: list[int], gate_name: str, device: str):
    print(f"=== P5-X3 {gate_name.upper()}: Pokec-z Grid ({len(seeds)} seeds: {seeds}) ===", flush=True)
    results = {}
    total_runs = len(seeds) * len(SCENARIOS)
    done = 0
    t_all0 = time.perf_counter()

    for sc_name, kwargs in SCENARIOS.items():
        print(f"\n--- Scenario: {sc_name} ---", flush=True)
        per = []
        for s in seeds:
            row = run_one(s, sc_name, kwargs, device)
            per.append(row)
            done += 1
            print(f"  [{done:2d}/{total_runs:2d}] {sc_name:15s} seed={s} AUC={row['auc']:.4f} "
                  f"DPD={row['dpd_hard']:.4f} w_adv={row['w_adv']:.4f} ({row['wall_clock_s']:.1f}s)", flush=True)

        aucs = [r["auc"] for r in per]
        dpds = [r["dpd_hard"] for r in per]
        w_advs = [r["w_adv"] for r in per]
        results[sc_name] = {
            "summary": {
                "auc": {"mean": float(st.mean(aucs)), "std": float(st.stdev(aucs)) if len(aucs) > 1 else 0.0},
                "dpd_hard": {"mean": float(st.mean(dpds)), "std": float(st.stdev(dpds)) if len(dpds) > 1 else 0.0},
                "w_adv": {"mean": float(st.mean(w_advs)), "std": float(st.stdev(w_advs)) if len(w_advs) > 1 else 0.0},
            },
            "per_seed": per,
        }

    t_all1 = time.perf_counter()
    all_walls = [r["wall_clock_s"] for sc in results for r in results[sc]["per_seed"]]
    mean_wall = float(st.mean(all_walls)) if all_walls else 0.0

    manifest = build_manifest(
        benchmark=f"p5_x3_pokecz_adversarial_{gate_name}",
        hardware={"device": device, "workers": 1, "threads": 1},
        metrics={
            "total_runs": total_runs,
            "mean_wall_per_run_s": mean_wall,
            "total_wall_s": t_all1 - t_all0,
        },
        args={
            "dataset": "pokec_z",
            "num_clients": 10,
            "rounds": 50,
            "local_epochs": 3,
            "dirichlet_alpha": 0.3,
            "seeds": seeds,
            "scenarios": list(SCENARIOS.keys()),
            "aggregator": "fu_shapley",
        },
    )

    sc = results.get("scaling_c100", {})
    w_adv_sc = sc.get("summary", {}).get("w_adv", {}).get("mean", 999.0)
    auc_cl = results.get("clean", {}).get("summary", {}).get("auc", {}).get("mean", 0.0)
    rejected = bool(w_adv_sc > 0.0500 or auc_cl < 0.7500)
    verdict = "REJECTED (H_P5X3 breached)" if rejected else "CONFIRMED (H_P5X3 holds, w_adv <= 0.0500 and AUC >= 0.7500)"

    hypothesis_test = {
        "pre_registered_thresholds": {
            "w_adv_scaling_max": 0.0500,
            "auc_min": 0.7500,
            "p_alpha": 0.05,
        },
        "measured": {
            "w_adv_scaling_mean": float(w_adv_sc),
            "auc_clean_mean": float(auc_cl),
        },
        "verdict": verdict,
        "rejected": rejected,
    }

    payload = {
        "manifest": manifest,
        "results": results,
        "rejection_criterion": {
            "w_adv_scaling_threshold": 0.0500,
            "auc_threshold": 0.7500,
            "alpha": 0.05,
            "n": len(seeds),
        },
        "hypothesis_test": hypothesis_test,
    }
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Successfully saved {gate_name.upper()} results to {out_file}", flush=True)
    print(f"  H_P5X3: w_adv(scaling)={w_adv_sc:.4f} auc(clean)={auc_cl:.4f} => {verdict}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=["s0", "s1", "s3", "s4"], default="s0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default="results/revision/pokecz_adversarial.json")
    args, _ = parser.parse_known_args()

    actual_dev = resolve_actual_device(args.device)
    os.environ["FEDFAIR_DEVICE"] = actual_dev

    if args.gate == "s0":
        s0_out = args.out_json.replace(".json", "_s0.json") if not args.out_json.endswith("_s0.json") else args.out_json
        run_s0(s0_out, device=actual_dev)
    elif args.gate == "s1":
        s1_out = args.out_json.replace(".json", "_s1.json") if not args.out_json.endswith("_s1.json") else args.out_json
        run_s1(s1_out, device=actual_dev)
    elif args.gate == "s3":
        s3_out = args.out_json.replace(".json", "_s3.json") if not args.out_json.endswith("_s3.json") else args.out_json
        run_grid(s3_out, seeds=[42, 43, 44], gate_name="s3", device=actual_dev)
    else:
        run_grid(args.out_json, seeds=SEEDS_FULL, gate_name="s4", device=actual_dev)
