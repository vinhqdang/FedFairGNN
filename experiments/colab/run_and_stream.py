import os
import subprocess
import sys

cmd = "python3 experiments/colab/run_remaining_experiments.py --smoke --tasks t12 t13 t15"
print(f"[STREAMING EXEC] {cmd}", flush=True)
os.chdir("/content/FedFairGNN")

p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
print(f"\n[FINISHED] Exit code: {p.returncode}")
