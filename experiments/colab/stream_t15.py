import os
import subprocess
import sys

os.system("tar -xzf /content/fedfairgnn.tgz -C /content/FedFairGNN")
cmd = "python3 experiments/revision/skewed_holdout_stress.py --dataset german --num-clients 5 --rounds 2 --device cpu --output /content/results/fairshare/skewed_holdout_results.json --smoke"
print(f"[STREAMING EXEC T15] {cmd}", flush=True)
os.chdir("/content/FedFairGNN")

p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
print(f"\n[FINISHED T15] Exit code: {p.returncode}")
