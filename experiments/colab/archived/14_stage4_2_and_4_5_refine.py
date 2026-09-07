"""Phase 4 · Refinement Runner: Stage 4.2 Symmetric Matrix & Stage 4.5 Local Pareto Sweep.

Executes:
1. Stage 4.2: FedAvg & FairShare on German + FedAvg & FairShare on Bail + Exact correlation probe.
2. Stage 4.5: Targeted Local Pareto 2D Grid Sweep (fairness_weight x beta_init on German).

Output: /content/refine_report.json
"""
import json
import os
import subprocess
import sys

REPO = "/content/FedFairGNN"
os.chdir(REPO)

print("=" * 70)
print("🚀 [COLAB VM] RUNNING REFINED STAGE 4.2 MATRIX & STAGE 4.5 LOCAL PARETO SWEEP")
print("=" * 70)

env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"

# 1. Run Stage 4.2
print("\n[*] Running Stage 4.2 symmetric suite...")
cmd_4_2 = "python -u experiments/stage4_2_runner.py"
r_4_2 = subprocess.run(cmd_4_2, shell=True, capture_output=True, text=True, env=env)
print(r_4_2.stdout)
if r_4_2.stderr:
    print("--- 4.2 STDERR ---", r_4_2.stderr)

# 2. Run Stage 4.5 targeted pareto sweep
print("\n[*] Running Stage 4.5 targeted local Pareto sweep...")
cmd_4_5 = "python -u experiments/stage4_5_ablation_suite.py --targeted"
r_4_5 = subprocess.run(cmd_4_5, shell=True, capture_output=True, text=True, env=env)
print(r_4_5.stdout)
if r_4_5.stderr:
    print("--- 4.5 STDERR ---", r_4_5.stderr)

success = (r_4_2.returncode == 0) and (r_4_5.returncode == 0)
report = {
    "stage": "Stage 4.2 & 4.5 Refinement",
    "exit_code": 0 if success else 1,
    "status": "PASS" if success else "FAIL",
    "stage4_2_stdout": r_4_2.stdout,
    "stage4_5_stdout": r_4_5.stdout,
}

with open("/content/refine_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n[*] Written report to /content/refine_report.json")
if not success:
    sys.exit(1)
