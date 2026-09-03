"""Phase 4 · Stage 4.5 — Resource-Efficient Q1 Ablation Studies Suite.

Executes:
- Part 1: Component-wise Ablations (M1–M7) across 3 seeds (42, 43, 44)
- Part 2: Sweep 1 — Fairness Weight Alpha (Pareto Frontier)
- Part 3: Sweep 2 — Server Holdout Size Sensitivity
- Part 4: Sweep 3 — DP Privacy Budget Epsilon
- Part 5: Sweep 4 — Non-IID Dirichlet Skew

Output: /content/stage4_5_report.json
"""
import json
import os
import subprocess
import sys

REPO = "/content/FedFairGNN"
os.chdir(REPO)

print("=" * 70)
print("🚀 [COLAB VM] RUNNING STAGE 4.5 ABLATION STUDIES SUITE")
print("=" * 70)

env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"

cmd = "python -u experiments/stage4_5_ablation_suite.py"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

print("--- STDOUT ---")
print(r.stdout)

if r.stderr:
    print("--- STDERR ---")
    print(r.stderr)

print(f"[*] Process exit code: {r.returncode}")

report = {
    "stage": "Stage 4.5",
    "exit_code": r.returncode,
    "status": "PASS" if r.returncode == 0 else "FAIL",
    "stdout": r.stdout,
    "stderr": r.stderr,
}

with open("/content/stage4_5_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n[*] Written report to /content/stage4_5_report.json")
if r.returncode != 0:
    sys.exit(r.returncode)
