"""P5-X1 S1 -- Reproducibility & CUDA same-seed jitter measurement on Bail-GPU.

Evaluates 1 seed (seed 42) twice on Bail under poison_honest for A1 (fltrust)
and A4 (fu_shapley).
Measures:
    delta_w_adv = |w_adv_run1 - w_adv_run2|
    sigma = delta_w_adv / sqrt(2)
Pricing rule:
    sigma ~ 0.002 -> 34 seeds
    sigma ~ 0.005 -> 53 seeds
    sigma ~ 0.010 -> 121 seeds
Guard: If sigma > 0.010, report back before opening S4.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.fltrust_delta_grid import run_one, ARMS, SCENARIOS


def run_s1():
    print("=" * 80, flush=True)
    print("=== P5-X1 S1: Bail-GPU CUDA Same-Seed Jitter Measurement ===", flush=True)
    print("=" * 80, flush=True)

    dataset = "bail"
    scenario = "poison_honest"
    seed = 42
    arms_to_test = ["A1_fltrust", "A4_fu_shapley"]

    results = {}
    max_sigma = 0.0

    for arm in arms_to_test:
        print(f"\n--- Testing Arm: {arm} (seed={seed}, scenario={scenario}) ---", flush=True)
        # Run 1
        t0 = time.perf_counter()
        r1 = run_one(
            dataset=dataset,
            arm_name=arm,
            arm_kwargs=ARMS[arm],
            scenario_name=scenario,
            attack_kwargs=SCENARIOS[scenario],
            seed=seed,
            device="cuda",
        )
        t1 = time.perf_counter()
        print(f"Run 1: AUC={r1['auc']:.4f} DPD={r1['dpd_hard']:.4f} w_adv={r1['w_adv']:.6f} ({t1-t0:.2f}s)", flush=True)

        # Run 2
        t0 = time.perf_counter()
        r2 = run_one(
            dataset=dataset,
            arm_name=arm,
            arm_kwargs=ARMS[arm],
            scenario_name=scenario,
            attack_kwargs=SCENARIOS[scenario],
            seed=seed,
            device="cuda",
        )
        t1 = time.perf_counter()
        print(f"Run 2: AUC={r2['auc']:.4f} DPD={r2['dpd_hard']:.4f} w_adv={r2['w_adv']:.6f} ({t1-t0:.2f}s)", flush=True)

        delta_w = abs(r1["w_adv"] - r2["w_adv"])
        delta_auc = abs(r1["auc"] - r2["auc"])
        delta_dpd = abs(r1["dpd_hard"] - r2["dpd_hard"])
        sigma = delta_w / math.sqrt(2)
        max_sigma = max(max_sigma, sigma)

        results[arm] = {
            "run1": r1,
            "run2": r2,
            "delta_w_adv": delta_w,
            "delta_auc": delta_auc,
            "delta_dpd": delta_dpd,
            "sigma": sigma,
        }

        print(f"  --> delta_w_adv = {delta_w:.8f} | sigma = {sigma:.8f} | delta_auc = {delta_auc:.8f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("=== S1 JITTER & SEED BUDGET PRICING SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    for arm, res in results.items():
        sig = res["sigma"]
        print(f"Arm {arm:<16}: delta_w_adv={res['delta_w_adv']:.6f}, sigma={sig:.6f}", flush=True)

    print(f"\nMax measured sigma across arms: {max_sigma:.6f}")
    if max_sigma <= 0.002:
        pricing_n = 34
    elif max_sigma <= 0.005:
        pricing_n = 53
    elif max_sigma <= 0.010:
        pricing_n = 121
    else:
        pricing_n = ">121 (EXCEEDED THRESHOLD 0.010)"

    print(f"Pricing reference budget: sigma={max_sigma:.6f} -> n ~ {pricing_n} seeds", flush=True)

    pass_threshold = max_sigma <= 0.010
    verdict = "PASS (sigma <= 0.010)" if pass_threshold else "FAIL_EXCEEDED (sigma > 0.010)"
    print(f"S1 Verdict: {verdict}", flush=True)

    os.makedirs("/content/results", exist_ok=True)
    out_path = "/content/results/p5x1_bail_s1.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "max_sigma": max_sigma,
            "pricing_n": str(pricing_n),
            "verdict": verdict,
            "pass_threshold": pass_threshold,
        }, f, indent=2)
    print(f"[+] Saved S1 report to {out_path}", flush=True)
    return pass_threshold


if __name__ == "__main__":
    ok = run_s1()
    if not ok:
        sys.exit(1)
