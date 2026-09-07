"""Execute a single harness script by index (1 to 6) on Colab GPU VM.

Usage:
  python 07_run_single_harness.py <idx>
"""
import sys
import os
import subprocess

if len(sys.argv) < 2:
    print("Error: missing script index")
    sys.exit(1)

idx = int(sys.argv[1])
REPO = "/content/FedFairGNN"
OUT_DIR = "/content/results/fairshare"
os.chdir(REPO)
os.makedirs(OUT_DIR, exist_ok=True)

env = dict(os.environ, FEDFAIR_DEVICE="cuda", PYTHONUNBUFFERED="1")

cmds = {
    1: ("derisk_phase3", "python -u -m experiments.derisk_phase3 --datasets german --seeds 1 --rounds 10 --alphas 0.1 --out /content/results/fairshare"),
    2: ("incentive_audit", "python -u -m experiments.incentive_audit --dataset german --rounds 10 --seeds 1 --num_clients 5 --num_byzantine 1 --out /content/results/fairshare"),
    3: ("exact_shapley_correlation", "python -u -m experiments.exact_shapley_correlation --dataset german --seed 0 --probe_at 10 --alpha 0.1 --game loss --normalize target_norm --out /content/results/fairshare"),
    4: ("ablation_val_size", "python -u -m experiments.ablation_val_size --dataset german --seed 0 --rounds 10 --alpha 0.1 --sizes 50 100 --out /content/results/fairshare"),
    5: ("topology_shapley_analysis", "python -u -m experiments.topology_shapley_analysis --dataset german --seed 0 --rounds 10 --alpha 0.3 --num_clients 5 --out /content/results/fairshare"),
    6: ("plot_shapley", "python -u -m experiments.plot_shapley --dataset german --seed 0 --rounds 10 --alpha 0.1 --out /content/results/fairshare")
}

if idx not in cmds:
    print(f"Invalid idx: {idx}")
    sys.exit(1)

name, cmd = cmds[idx]
print(f"=== Running Harness [{idx}/6]: {name} ===", flush=True)
r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, cwd=REPO)
print(r.stdout[-1500:], flush=True)
if r.returncode != 0:
    print(f"❌ {name} FAILED:", r.stderr[-1000:], flush=True)
    sys.exit(r.returncode)
else:
    print(f"✅ {name} PASSED", flush=True)
