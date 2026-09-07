"""Phase 1 & Phase 2 — Pipeline verification and 6-script Harness test on Colab GPU.

Run via:
  colab exec -s $S -f colab/06_phase1_2_harness.py --timeout 1800
"""
import json
import os
import glob
import pandas as pd
import subprocess

REPO = "/content/FedFairGNN"
OUT_DIR = "/content/results/fairshare"

os.chdir(REPO)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/content/data", exist_ok=True)

# 1. Symlink dataset cache
if not os.path.exists(f"{REPO}/data"):
    os.symlink("/content/data", f"{REPO}/data")

env = dict(os.environ, FEDFAIR_DEVICE="cuda", PYTHONUNBUFFERED="1")

report = {
    "phase1": {},
    "phase2": {},
    "gate1_pass": False,
    "gate2_pass": False
}

print("=== PHASE 1: End-to-End Pipeline Verification ===", flush=True)
cmd_p1 = (
    "python -u -m experiments.derisk_phase3 "
    "--datasets german --seeds 1 --rounds 50 --alphas 0.1 --out /content/results/fairshare"
)
r_p1 = subprocess.run(cmd_p1, shell=True, env=env, capture_output=True, text=True, cwd=REPO)
print(r_p1.stdout[-1500:], flush=True)
if r_p1.returncode != 0:
    print("STDERR:", r_p1.stderr[-1000:], flush=True)

p1_csv = f"{OUT_DIR}/derisk_D1_summary.csv"
if os.path.exists(p1_csv):
    df_p1 = pd.read_csv(p1_csv)
    print("Phase 1 CSV columns:", list(df_p1.columns), flush=True)
    no_nan = not df_p1[["auc_mean", "dpd_mean"]].isnull().any().any()
    auc_valid = df_p1["auc_mean"].between(0.55, 0.85).all()
    report["phase1"] = {
        "csv_exists": True,
        "no_nan": bool(no_nan),
        "auc_in_range": bool(auc_valid),
        "rows": len(df_p1)
    }
    report["gate1_pass"] = bool(no_nan and auc_valid)
else:
    report["phase1"] = {"csv_exists": False, "error": r_p1.stderr[-500:]}

print(f"GATE 1 Status: {'PASS' if report['gate1_pass'] else 'FAIL'}", flush=True)

print("\n=== PHASE 2: 6-Script Harness Verification ===", flush=True)
harness_scripts = [
    ("derisk_phase3", "python -u -m experiments.derisk_phase3 --datasets german --seeds 1 --rounds 10 --alphas 0.1 --out /content/results/fairshare"),
    ("incentive_audit", "python -u -m experiments.incentive_audit --dataset german --rounds 10 --seeds 1 --num_clients 5 --num_byzantine 1 --out /content/results/fairshare"),
    ("exact_shapley_correlation", "python -u -m experiments.exact_shapley_correlation --dataset german --seed 0 --probe_at 10 --alpha 0.1 --game loss --normalize target_norm --out /content/results/fairshare"),
    ("ablation_val_size", "python -u -m experiments.ablation_val_size --dataset german --seed 0 --rounds 10 --alpha 0.1 --sizes 50 100 --out /content/results/fairshare"),
    ("topology_shapley_analysis", "python -u -m experiments.topology_shapley_analysis --dataset german --seed 0 --rounds 10 --alpha 0.3 --num_clients 5 --out /content/results/fairshare"),
    ("plot_shapley", "python -u -m experiments.plot_shapley --dataset german --seed 0 --rounds 10 --alpha 0.1 --out /content/results/fairshare")
]

p2_results = {}
for name, cmd in harness_scripts:
    print(f"Running harness script: {name}...", flush=True)
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, cwd=REPO)
    success = (res.returncode == 0)
    p2_results[name] = {
        "returncode": res.returncode,
        "success": success,
        "stdout_tail": res.stdout[-400:] if success else res.stderr[-400:]
    }
    if not success:
        print(f"  ❌ {name} failed: {res.stderr[-400:]}", flush=True)

# Check CSV schemas and outputs
generated_files = glob.glob(f"{OUT_DIR}/*.csv")
report["phase2"] = {
    "scripts": p2_results,
    "generated_csv_count": len(generated_files),
    "csv_list": [os.path.basename(f) for f in generated_files]
}

# Check audit_traj__*.csv and incentive metrics
audit_trajs = glob.glob(f"{OUT_DIR}/audit_traj__*.csv")
has_audit_traj = len(audit_trajs) > 0
all_scripts_ok = all(v["success"] for v in p2_results.values())

report["gate2_pass"] = bool(all_scripts_ok and has_audit_traj)
print(f"GATE 2 Status: {'PASS' if report['gate2_pass'] else 'FAIL'}", flush=True)

report_json_path = f"{OUT_DIR}/phase1_2_report.json"
json.dump(report, open(report_json_path, "w"), indent=2)
print("\n" + json.dumps(report, indent=2))
