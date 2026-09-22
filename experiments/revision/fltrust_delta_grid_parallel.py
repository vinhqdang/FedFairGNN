"""P5-X1 S4 parallel launcher — splits 30 seeds into N_WORKERS chunks.

Each worker runs fltrust_delta_grid.py on its subset of seeds,
outputs to results/revision/fltrust_delta_grid_shard_{i}.json,
then merge_shards() combines all shards into fltrust_delta_grid.json.

Usage: python fltrust_delta_grid_parallel.py --workers 5
Expected: 30 seeds / 5 workers x 8 arms x 68s = ~54 min (bail bottleneck)
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import statistics as st


def run_worker(worker_id: int, seeds: list[int], out_dir: str) -> str:
    """Run a fltrust_delta_grid subprocess for a subset of seeds."""
    shard_path = f"{out_dir}/fltrust_delta_grid_shard_{worker_id}.json"
    seed_args = " ".join(str(s) for s in seeds)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    # Patch the runner to accept --seeds override
    cmd = [
        sys.executable,
        "experiments/revision/fltrust_delta_grid.py",
        "--gate", "s4",
        "--out-json", shard_path,
        "--seeds", seed_args,
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"[W{worker_id}] {line}", end="", flush=True)
    proc.wait()
    return shard_path


def merge_shards(shard_paths: list[str], out_json: str) -> None:
    """Merge all shards into a single artifact."""
    merged_results: dict = {}
    all_manifests = []

    for path in shard_paths:
        with open(path) as f:
            shard = json.load(f)
        all_manifests.append(shard["manifest"])
        for dataset, arms in shard["results"].items():
            if dataset not in merged_results:
                merged_results[dataset] = {}
            for arm, data in arms.items():
                if arm not in merged_results[dataset]:
                    merged_results[dataset][arm] = {"per_seed": []}
                merged_results[dataset][arm]["per_seed"].extend(data["per_seed"])

    # Compute summary stats for merged
    for dataset, arms in merged_results.items():
        for arm, data in arms.items():
            aucs = [r["auc"] for r in data["per_seed"]]
            dpds = [r["dpd_hard"] for r in data["per_seed"]]
            w_advs = [r["w_adv"] for r in data["per_seed"]]
            data["auc"] = {"mean": st.mean(aucs), "std": st.stdev(aucs)}
            data["dpd_hard"] = {"mean": st.mean(dpds), "std": st.stdev(dpds)}
            data["w_adv"] = {"mean": st.mean(w_advs), "std": st.stdev(w_advs)}

    # Use first shard's manifest as base
    manifest = all_manifests[0].copy()
    manifest["note"] = f"Merged from {len(shard_paths)} parallel shards"
    manifest["total_seeds"] = sum(
        len(list(merged_results.values())[0][arm]["per_seed"])
        for arm in list(list(merged_results.values())[0].keys())[:1]
    )

    payload = {"manifest": manifest, "results": merged_results}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Merged {len(shard_paths)} shards -> {out_json}", flush=True)

    # Print 2x2 summary
    print("\n=== P5-X1 2x2 DELTA SUMMARY ===", flush=True)
    for dataset in merged_results:
        arms = merged_results[dataset]
        print(f"\n{dataset}:")
        for arm, data in arms.items():
            print(f"  {arm:25s}: AUC={data['auc']['mean']:.4f}±{data['auc']['std']:.4f} "
                  f"DPD={data['dpd_hard']['mean']:.4f}±{data['dpd_hard']['std']:.4f} "
                  f"w_adv={data['w_adv']['mean']:.4f}±{data['w_adv']['std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out-dir", default="results/revision")
    parser.add_argument("--out-json", default="results/revision/fltrust_delta_grid.json")
    args = parser.parse_args()

    seeds = list(range(42, 72))
    chunk_size = len(seeds) // args.workers + 1
    chunks = [seeds[i:i+chunk_size] for i in range(0, len(seeds), chunk_size)][:args.workers]

    print(f"=== P5-X1 S4 PARALLEL: {len(seeds)} seeds / {args.workers} workers ===", flush=True)
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: seeds {chunk[0]}..{chunk[-1]} ({len(chunk)} seeds)", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    shard_paths = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_worker, i, chunk, args.out_dir): i for i, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            wid = futures[fut]
            path = fut.result()
            shard_paths.append(path)
            print(f"[+] Worker {wid} DONE -> {path}", flush=True)

    merge_shards(sorted(shard_paths), args.out_json)
    total = time.perf_counter() - t0
    print(f"\n=== TOTAL PARALLEL TIME: {total:.1f}s ({total/240:.1f}x speedup vs serial) ===", flush=True)
