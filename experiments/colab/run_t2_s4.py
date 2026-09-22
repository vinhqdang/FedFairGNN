"""T2 S4 -- Full 30-seed metadata capture on Bail (K=5) on Colab GPU.

Gate S4: runs after S3 was approved.
Seeds: 42..71 (30 seeds), 10 rules x 3 arms = 900 total runs.
Expected time: ~118 min per S3 extrapolation (7.87s/run).
"""
import os
import sys
import time

os.chdir("/content/FedFairGNN")
sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.metadata_capture_endtoend import run  # noqa: E402

SEEDS = list(range(42, 72))  # 30 seeds

print(f"=== STARTING T2 S4: Full run ({len(SEEDS)} seeds x 10 rules x 3 arms = {len(SEEDS)*30} total) ===", flush=True)
print(f"Expected ~{int(len(SEEDS)*30*7.87/60)} min based on S3 probe (7.87s/run)", flush=True)
t0 = time.perf_counter()
run(
    out_json="/content/results/metadata_capture_bail.json",
    dataset="bail",
    num_clients=5,
    rounds=20,
    seeds=SEEDS,
)
t1 = time.perf_counter()
total_sec = t1 - t0
print(f"=== S4 COMPLETE in {total_sec:.1f}s ({total_sec/(len(SEEDS)*30):.2f}s/run) ===", flush=True)
print("[+] Artifact: /content/results/metadata_capture_bail.json", flush=True)
