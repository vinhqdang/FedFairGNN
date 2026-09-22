"""Full Run Orchestrator for T12, T13, T14, T15 on Colab.

Executes:
1. Unpack latest tarball & setup symlinks
2. Run full suite:
   - T12: adv_influence_instrumentation (German K=5, R=20, n=10, cpu)
   - T13: task_neutral_fairness_adversary (German K=5, R=20, n=20, cpu, 7 eps points x 2 alphas)
   - T14: fltrust_ema_benchmark (Pokec-z K=10, R=50, n=10, cuda, 3 arms)
   - T15: skewed_holdout_stress (German + Bail K=5, R=20, n=10, cpu, 4 regimes)
3. Post-run FLTrust CPU bit-exact hash verification
4. Evaluate 4 acceptance gates (G1, G2, G3, G4)
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
print("  LAUNCHING FULL EXPERIMENTAL CAMPAIGN (T12, T13, T14, T15)")
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
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
    dt = time.perf_counter() - t0
    print(f"\n[STATUS] Exit {p.returncode} in {dt:.2f}s\n", flush=True)
    assert p.returncode == 0, f"Command failed: {cmd}"

# Run all 4 tasks full suite
t0_campaign = time.perf_counter()
run("python3 experiments/colab/run_remaining_experiments.py --tasks t12 t13 t14 t15")
total_campaign_time = time.perf_counter() - t0_campaign

print("\n" + "=" * 80)
print(f"  FULL EXPERIMENTAL RUN FINISHED IN {total_campaign_time/60:.2f} MINUTES")
print("=" * 80)

# Post-run verification of bit-exact FLTrust hash
print("\n>>> POST-RUN: Re-checking FLTrust bit-exact hash...")
run("python3 tests/test_fltrust_bitexact.py")
