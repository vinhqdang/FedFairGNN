"""P5-X1 S4 -- Full 30-seed 2x2 grid on Bail (K=5) on Colab GPU.

Evaluates 4 arms across 30 seeds {42..71} under clean and poison_honest:
    (1) A1_fltrust: FLTrust verbatim
    (2) A2_fltrust_fair: FLTrust + alpha * g_fair
    (3) A3_fu_alpha0: FU-Shapley alpha=0
    (4) A4_fu_shapley: FU-Shapley alpha=0.1

Total runs: 2 scenarios x 4 arms x 30 seeds = 240 runs.
Output: /content/results/fltrust_delta_grid_bail_gpu.json
"""
from __future__ import annotations

import json
import os
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.fltrust_delta_grid import run_gate, ARMS


SEEDS_S4 = list(range(42, 72))  # 30 seeds
SCENARIOS_BAIL = ["clean", "poison_honest"]  # fairness_poison omitted as identical to poison_honest
ARMS_ALL = list(ARMS.keys())


def run_s4():
    print("=" * 80, flush=True)
    print(f"=== STARTING P5-X1 S4: Bail-GPU ({len(SEEDS_S4)} seeds x 4 arms x 2 scenarios = {len(SEEDS_S4)*8} runs) ===", flush=True)
    print(f"Scenarios: {SCENARIOS_BAIL}", flush=True)
    print(f"Arms: {ARMS_ALL}", flush=True)
    print(f"Expected ~{int(len(SEEDS_S4)*8*7.87/60)} min on GPU T4/A100 (7.87s/run)", flush=True)
    print("=" * 80, flush=True)

    os.makedirs("/content/results", exist_ok=True)
    out_json = "/content/results/fltrust_delta_grid_bail_gpu.json"

    t0 = time.perf_counter()
    res = run_gate(
        out_json=out_json,
        seeds=SEEDS_S4,
        datasets=["bail"],
        scenarios=SCENARIOS_BAIL,
        arms=ARMS_ALL,
        device="cuda",
        workers=1,
    )
    t1 = time.perf_counter()
    total_sec = t1 - t0
    print(f"\n=== P5-X1 S4 BAIL-GPU COMPLETE in {total_sec:.1f}s ({total_sec/(len(SEEDS_S4)*8):.2f}s/run) ===", flush=True)
    print(f"[+] Artifact saved to: {out_json}", flush=True)
    return res


if __name__ == "__main__":
    run_s4()
