"""Unified Colab & Remote Runner for Remaining Experiments (T12, T13, T14, T15).

Orchestrates the 4 remaining experimental campaigns:
    - T12: Adversarial Influence Instrumentation (I_adv)
    - T13: Task-Neutral Fairness Adversary (Resolving Flaw #1)
    - T14: FLTrust + EMA History Smoothing on Pokec-z
    - T15: Skewed Holdout Stress Test (Representativeness Bounds)

Usage:
    python3 experiments/colab/run_remaining_experiments.py --tasks t12 t13 t14 t15
    python3 experiments/colab/run_remaining_experiments.py --smoke --tasks t12 t13 t15
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
sys.path.insert(0, os.path.abspath("."))

RESULTS_DIR = "/content/results/fairshare" if os.path.exists("/content") else "results/revision"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_cmd(cmd: str) -> int:
    print(f"\n[RUNNING] {cmd}", flush=True)
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.perf_counter() - t0
    status = "SUCCESS" if ret.returncode == 0 else "FAILED"
    print(f"[{status}] in {elapsed:.1f}s (Exit {ret.returncode})\n", flush=True)
    return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="Unified Runner for T12-T15 Experiments")
    parser.add_argument("--tasks", nargs="+", default=["t12", "t13", "t14", "t15"],
                        choices=["t12", "t13", "t14", "t15"])
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (1 seed, 2 rounds)")
    parser.add_argument("--device", type=str, default="cuda" if os.path.exists("/content") else "cpu")
    args = parser.parse_args()

    smoke_flag = "--smoke" if args.smoke else ""

    print("=" * 80)
    print("  REMAINING EXPERIMENTAL CAMPAIGNS (T12, T13, T14, T15)")
    print(f"  Selected Tasks: {args.tasks} | Smoke: {args.smoke} | Device: {args.device}")
    print(f"  Output Directory: {RESULTS_DIR}")
    print("=" * 80)

    # 0. Pre-flight bit-exact verification for FLTrust
    if args.smoke:
        print("\n>>> Pre-flight: FLTrust CPU Bit-Exactness & Hash Verification")
        cmd_bitexact = "python3 tests/test_fltrust_bitexact.py"
        if run_cmd(cmd_bitexact) != 0:
            print("❌ FLTrust bit-exact verification failed! Terminating.")
            sys.exit(1)

    # 1. T12: I_adv instrumentation (German K=5, R=20, device: cpu)
    if "t12" in args.tasks:
        print("\n>>> Task T12: Adversarial Influence Instrumentation (I_adv)")
        out_t12 = os.path.join(RESULTS_DIR, "adv_influence_results.json")
        cmd_t12 = (
            f"python3 experiments/revision/adv_influence_instrumentation.py "
            f"--dataset german --num-clients 5 --rounds 20 --device cpu "
            f"--output {out_t12} {smoke_flag}"
        )
        if run_cmd(cmd_t12) != 0:
            print("❌ T12 failed.")

    # 2. T13: Task-neutral fairness adversary (German K=5, R=20, device: cpu)
    if "t13" in args.tasks:
        print("\n>>> Task T13: Task-Neutral Fairness Adversary (Alpha Scope)")
        out_t13 = os.path.join(RESULTS_DIR, "task_neutral_adversary.json")
        cmd_t13 = (
            f"python3 experiments/revision/task_neutral_fairness_adversary.py "
            f"--dataset german --num-clients 5 --rounds 20 --device cpu "
            f"--eps-task 0.005 0.01 0.02 0.05 0.10 --alphas 0.0 0.1 "
            f"--output {out_t13} {smoke_flag}"
        )
        if run_cmd(cmd_t13) != 0:
            print("❌ T13 failed.")

    # 3. T14: FLTrust + EMA (Pokec-z K=10, R=50, device: cuda)
    if "t14" in args.tasks:
        print("\n>>> Task T14: FLTrust + EMA Smoothing on Pokec-z")
        dev_t14 = "cuda" if (args.device == "cuda" or os.path.exists("/content")) else "cpu"
        out_t14 = os.path.join(RESULTS_DIR, "fltrust_ema_results.json")
        cmd_t14 = (
            f"python3 experiments/revision/fltrust_ema_benchmark.py "
            f"--dataset pokec_z --num-clients 10 --rounds 50 --device {dev_t14} "
            f"--output {out_t14} {smoke_flag}"
        )
        if run_cmd(cmd_t14) != 0:
            print("❌ T14 failed.")

    # 4. T15: Skewed Holdout (German + Bail K=5, R=20, device: cpu)
    if "t15" in args.tasks:
        print("\n>>> Task T15: Skewed Holdout Representativeness Stress Test (German + Bail)")
        out_t15_german = os.path.join(RESULTS_DIR, "skewed_holdout_results.json")
        cmd_t15_german = (
            f"python3 experiments/revision/skewed_holdout_stress.py "
            f"--dataset german --num-clients 5 --rounds 20 --device cpu "
            f"--output {out_t15_german} {smoke_flag}"
        )
        if run_cmd(cmd_t15_german) != 0:
            print("❌ T15 (German) failed.")

        out_t15_bail = os.path.join(RESULTS_DIR, "skewed_holdout_bail_results.json")
        cmd_t15_bail = (
            f"python3 experiments/revision/skewed_holdout_stress.py "
            f"--dataset bail --num-clients 5 --rounds 20 --device cpu "
            f"--output {out_t15_bail} {smoke_flag}"
        )
        if run_cmd(cmd_t15_bail) != 0:
            print("❌ T15 (Bail) failed.")

    print("\n[ALL TASKS PROCESSED]")


if __name__ == "__main__":
    main()
