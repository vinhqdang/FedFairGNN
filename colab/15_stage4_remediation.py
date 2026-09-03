"""Stage 4 Remediation Runner for Google Colab VM.

Executes:
1. Unit Tests Suite (41 tests): verifying all guards and fixes on the VM.
2. Stage 4 Remediation Suite (experiments/stage4_remediation_runner.py):
   - Part 1: Stage 4.2 Canonical 3-seed Matrix (German & Bail without leakage)
   - Part 2: FSER Ablation M1 vs M2 Beta Sweep across 3 seeds
   - Part 3: Two-Tier Defense Validation (M1 vs M6) under Byzantine Attacks across 3 seeds

Output: /content/remediation_report.json
"""
import json
import os
import subprocess
import sys

REPO = "/content/FedFairGNN"
os.chdir(REPO)

print("=" * 70, flush=True)
print("🚀 [COLAB VM] RUNNING STAGE 4 REMEDIATION SUITE (3 SEEDS CANONICAL)", flush=True)
print("=" * 70, flush=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
env["PYTHONUNBUFFERED"] = "1"

# 1. Run Unit Tests
print("\n[*] [1/2] Running full unit tests on VM...", flush=True)
r_test = subprocess.run(["python", "-m", "pytest", "-v", "tests/"], env=env)

# 2. Run Remediation Suite
print("\n[*] [2/2] Running Stage 4 remediation suite (3 seeds)...", flush=True)
r_rem = subprocess.run(["python", "-u", "experiments/stage4_remediation_runner.py"], env=env)

success = (r_test.returncode == 0) and (r_rem.returncode == 0)
report = {
    "stage": "Stage 4 Remediation",
    "exit_code": 0 if success else 1,
    "status": "PASS" if success else "FAIL",
}

with open("/content/remediation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n[*] Written report to /content/remediation_report.json", flush=True)
if not success:
    sys.exit(1)
