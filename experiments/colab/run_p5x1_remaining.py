"""Run remaining runs of P5-X1 S4 on Bail-GPU with incremental checkpointing.

Resumes from 173 runs already completed, executing the final 67 runs,
saving each result to disk immediately after completion.
When all 240 runs are present, produces /content/results/fltrust_delta_grid_bail_gpu.json.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time

if os.path.exists("/content/FedFairGNN"):
    os.chdir("/content/FedFairGNN")
    sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.fltrust_delta_grid import (
    ARMS,
    SCENARIOS,
    _worker_entry,
)
from src.utils.provenance import build_manifest


SEEDS_S4 = list(range(42, 72))  # 30 seeds
SCENARIOS_BAIL = ["clean", "poison_honest"]
ARMS_ALL = list(ARMS.keys())


def main():
    os.makedirs("/content/results", exist_ok=True)
    chk_path = "/content/results/bail_gpu_checkpoint.json"
    init_173_path = "/content/results/bail_gpu_first_173_runs.json"
    final_out = "/content/results/fltrust_delta_grid_bail_gpu.json"

    completed_rows = []
    if os.path.exists(chk_path):
        with open(chk_path) as f:
            completed_rows = json.load(f)
    elif os.path.exists(init_173_path):
        with open(init_173_path) as f:
            completed_rows = json.load(f)

    done_keys = set()
    for r in completed_rows:
        done_keys.add((r["dataset"], r["scenario"], r["arm"], r["seed"]))

    print(f"[*] Loaded {len(completed_rows)} previously completed runs.")

    # Generate all 240 tasks
    all_tasks = []
    for scenario in SCENARIOS_BAIL:
        for arm in ARMS_ALL:
            for seed in SEEDS_S4:
                all_tasks.append(("bail", scenario, arm, seed, "cuda"))

    remaining_tasks = [t for t in all_tasks if (t[0], t[1], t[2], t[3]) not in done_keys]
    print(f"[*] Total required: {len(all_tasks)}, Remaining to run: {len(remaining_tasks)}", flush=True)

    t0 = time.perf_counter()
    done_count = len(completed_rows)

    for i, t in enumerate(remaining_tasks, 1):
        d, sc, arm, seed, dev = t
        t_run0 = time.perf_counter()
        res = _worker_entry(t)
        t_run1 = time.perf_counter()

        completed_rows.append(res)
        done_count += 1
        r = res["row"]

        # Immediate flush to disk checkpoint
        with open(chk_path, "w") as f:
            json.dump(completed_rows, f)

        print(
            f"[{done_count:3d}/240] (+{i}/{len(remaining_tasks)}) {d:6s} {sc:15s} {arm:16s} "
            f"seed={seed:2d} auc={r['auc']:.4f} dpd={r['dpd_hard']:.4f} w_adv={r['w_adv']:.4f} "
            f"({t_run1 - t_run0:.1f}s)",
            flush=True,
        )

    t1 = time.perf_counter()
    print(f"\n[*] All runs completed in {t1 - t0:.1f}s. Assembling final artifact...", flush=True)

    # Assemble structured results
    results: dict = {"bail": {}}
    for sc in SCENARIOS_BAIL:
        results["bail"][sc] = {}
        for arm in ARMS_ALL:
            results["bail"][sc][arm] = {"per_seed": []}

    for item in completed_rows:
        d = item["dataset"]
        sc = item["scenario"]
        a = item["arm"]
        if d in results and sc in results[d] and a in results[d][sc]:
            results[d][sc][a]["per_seed"].append(item["row"])

    for sc in SCENARIOS_BAIL:
        for a in ARMS_ALL:
            per_seed = results["bail"][sc][a]["per_seed"]
            per_seed.sort(key=lambda x: x["seed"])
            if per_seed:
                aucs = [r["auc"] for r in per_seed]
                dpds = [r["dpd_hard"] for r in per_seed]
                w_advs = [r["w_adv"] for r in per_seed]
                walls = [r["wall_s"] for r in per_seed]
                results["bail"][sc][a]["auc"] = {
                    "mean": float(st.mean(aucs)),
                    "std": float(st.stdev(aucs)) if len(aucs) > 1 else 0.0,
                }
                results["bail"][sc][a]["dpd_hard"] = {
                    "mean": float(st.mean(dpds)),
                    "std": float(st.stdev(dpds)) if len(dpds) > 1 else 0.0,
                }
                results["bail"][sc][a]["w_adv"] = {
                    "mean": float(st.mean(w_advs)),
                    "std": float(st.stdev(w_advs)) if len(w_advs) > 1 else 0.0,
                }
                results["bail"][sc][a]["mean_wall_s"] = float(st.mean(walls))

    all_walls = [item["row"]["wall_s"] for item in completed_rows]
    mean_wall_per_run = float(st.mean(all_walls)) if all_walls else 0.0

    manifest = build_manifest(
        benchmark="fltrust_delta_grid_bail_s4",
        hardware={"device": "cuda", "workers": 1, "threads": 1},
        metrics={
            "total_runs": len(completed_rows),
            "mean_wall_per_run_s": mean_wall_per_run,
            "total_wall_s": t1 - t0,
        },
    )

    artifact = {
        "manifest": manifest,
        "config": {
            "seeds": SEEDS_S4,
            "datasets": ["bail"],
            "scenarios": SCENARIOS_BAIL,
            "arms": ARMS_ALL,
            "alpha": 0.1,
            "K": 5,
        },
        "results": results,
    }

    with open(final_out, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[SUCCESS] Final artifact with all {len(completed_rows)} runs saved to {final_out}", flush=True)


if __name__ == "__main__":
    main()
