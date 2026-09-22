import os
import sys
import json
import time

os.chdir("/content/FedFairGNN")
sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.metadata_capture_endtoend import run

print("=== STARTING S0: Smoke run on Colab GPU ===", flush=True)
t0 = time.perf_counter()
run(out_json="/content/results/t2_s0.json", dataset="bail", num_clients=5, rounds=20, seeds=[42], rules=["fairfed"])
t1 = time.perf_counter()
print(f"S0 complete in {t1-t0:.2f}s", flush=True)

with open("/content/results/t2_s0.json") as f:
    d0 = json.load(f)

m0 = d0["manifest"]
res0 = d0["results"]["fairfed"]["arms"]
print("S0 Manifest:", json.dumps(m0, indent=2), flush=True)
print("S0 Results clean AUC:", res0["clean"]["auc_mean"], flush=True)

print("\n=== STARTING S1: Reproducibility & CUDA same-seed jitter measurement ===", flush=True)
t0 = time.perf_counter()
run(out_json="/content/results/t2_s1.json", dataset="bail", num_clients=5, rounds=20, seeds=[42], rules=["fairfed"])
t1 = time.perf_counter()
print(f"S1 complete in {t1-t0:.2f}s", flush=True)

with open("/content/results/t2_s1.json") as f:
    d1 = json.load(f)

res1 = d1["results"]["fairfed"]["arms"]
print("S1 Results clean AUC:", res1["clean"]["auc_mean"], flush=True)

print("\n=== CUDA SAME-SEED JITTER REPORT (S0 vs S1) ===", flush=True)
for arm in ["clean", "poison_honest", "poison_lie"]:
    auc0 = res0[arm]["auc_mean"]
    auc1 = res1[arm]["auc_mean"]
    dpd0 = res0[arm]["dpd_hard_mean"]
    dpd1 = res1[arm]["dpd_hard_mean"]
    print(f"Arm {arm:<14}: S0 AUC={auc0:.8f}, S1 AUC={auc1:.8f}, delta_auc={abs(auc0-auc1):.8f}", flush=True)
    print(f"                S0 DPD={dpd0:.8f}, S1 DPD={dpd1:.8f}, delta_dpd={abs(dpd0-dpd1):.8f}", flush=True)
