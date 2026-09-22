"""Smoke test execution script for Colab.

Runs:
1. Repo setup & directory structure
2. Pre-flight FLTrust CPU bit-exact test
3. T12 (adv influence), T13 (task-neutral adversary with 3 theoretical invariant checks), T15 (skewed holdout) on CPU
4. T14 (FLTrust + EMA smoothing) on GPU
"""
import os
import shutil
import subprocess
import sys
import time

REPO = "/content/FedFairGNN"
DATA = "/content/data"
RESULTS = "/content/results/fairshare"

print("=" * 80)
print("  STARTING COMPREHENSIVE SMOKE TEST ON COLAB")
print("=" * 80)

# Unpack latest repo
shutil.rmtree(REPO, ignore_errors=True)
os.makedirs(REPO, exist_ok=True)
ret = subprocess.run(f"tar -xzf /content/fedfairgnn.tgz -C {REPO}", shell=True)
assert ret.returncode == 0, "Failed to unpack fedfairgnn.tgz"

# Symlinks
for d in (DATA, RESULTS):
    os.makedirs(d, exist_ok=True)

link_data = os.path.join(REPO, "data")
if os.path.islink(link_data) or os.path.exists(link_data):
    shutil.rmtree(link_data, ignore_errors=True)
    if os.path.islink(link_data):
        os.unlink(link_data)
os.symlink(DATA, link_data)

link_res = os.path.join(REPO, "results")
if os.path.islink(link_res) or os.path.exists(link_res):
    shutil.rmtree(link_res, ignore_errors=True)
    if os.path.islink(link_res):
        os.unlink(link_res)
os.symlink("/content/results", link_res)

os.chdir(REPO)
sys.path.insert(0, REPO)

def run(cmd):
    print(f"\n[EXEC] {cmd}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, shell=True, text=True)
    dt = time.perf_counter() - t0
    print(f"[STATUS] Exit {p.returncode} in {dt:.2f}s\n", flush=True)
    assert p.returncode == 0, f"Command failed: {cmd}"

# 1. Step 1: FLTrust Bit-Exact Check
print("\n>>> STEP 1: FLTrust Bit-Exact & CPU Hash Lock Check...")
run("python3 tests/test_fltrust_bitexact.py")

# 2. Step 2: CPU Tasks Smoke (T12, T13, T15)
print("\n>>> STEP 2: Running Smoke on T12, T13, T15 (CPU)...")
run("python3 experiments/colab/run_remaining_experiments.py --smoke --tasks t12 t13 t15")

# 3. Step 3: GPU Task Smoke (T14)
print("\n>>> STEP 3: Running Smoke on T14 (FLTrust + EMA, GPU)...")
run("python3 experiments/colab/run_remaining_experiments.py --smoke --tasks t14 --device cuda")

print("\n" + "=" * 80)
print("  ALL SMOKE TESTS AND THEORETICAL INVARIANTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
