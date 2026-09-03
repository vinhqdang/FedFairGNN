"""Phase 4 · Stage 4.2 — Small Real Benchmarks (German Credit & Bail Recidivism).

Executes Stage 4.2 on the remote Colab VM.
Output: /content/stage4_2_report.json
"""
import json
import os
import subprocess
import sys

REPO = "/content/FedFairGNN"
os.chdir(REPO)

print("=" * 70)
print("🚀 [COLAB VM] RUNNING STAGE 4.2 SMALL BENCHMARKS & CORRELATION HARNESS")
print("=" * 70)

env = os.environ.copy()
env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"

cmd = "python -u experiments/stage4_2_runner.py"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

print("--- STDOUT ---")
print(r.stdout)

if r.stderr:
    print("--- STDERR ---")
    print(r.stderr)

print(f"[*] Process exit code: {r.returncode}")

report = {
    "stage": "Stage 4.2",
    "exit_code": r.returncode,
    "status": "PASS" if r.returncode == 0 else "FAIL",
    "stdout": r.stdout,
    "stderr": r.stderr,
}

with open("/content/stage4_2_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n[*] Written report to /content/stage4_2_report.json")
if r.returncode != 0:
    sys.exit(r.returncode)
