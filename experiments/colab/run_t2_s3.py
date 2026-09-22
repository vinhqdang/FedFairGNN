import os
import sys
import json
import time

os.chdir("/content/FedFairGNN")
sys.path.insert(0, "/content/FedFairGNN")
os.environ["FEDFAIR_DEVICE"] = "cuda"

from experiments.revision.metadata_capture_endtoend import run

print("=== STARTING S3: Warmup run (3 seeds x 10 rules x 3 arms) on Colab GPU ===", flush=True)
t0 = time.perf_counter()
run(out_json="/content/results/t2_s3.json", dataset="bail", num_clients=5, rounds=20, seeds=[42, 43, 44])
t1 = time.perf_counter()
total_sec = t1 - t0
print(f"=== S3 COMPLETE in {total_sec:.1f}s ({total_sec/90:.2f}s/run) ===", flush=True)

with open("/content/results/t2_s3.json") as f:
    d = json.load(f)

results = d["results"]
print("\n=== S3 DELTAS & BUDGET EXTRAPOLATION ===", flush=True)
for rule, r_data in results.items():
    reads = r_data["reads"]
    arms = r_data["arms"]
    c_auc = arms.get("clean", {}).get("auc_mean", float("nan"))
    h_auc = arms.get("poison_honest", {}).get("auc_mean", float("nan"))
    l_auc = arms.get("poison_lie", {}).get("auc_mean", float("nan"))
    c_dpd = arms.get("clean", {}).get("dpd_hard_mean", float("nan"))
    h_dpd = arms.get("poison_honest", {}).get("dpd_hard_mean", float("nan"))
    l_dpd = arms.get("poison_lie", {}).get("dpd_hard_mean", float("nan"))
    delta_auc = l_auc - h_auc if (l_auc is not None and h_auc is not None) else float("nan")
    delta_dpd = l_dpd - h_dpd if (l_dpd is not None and h_dpd is not None) else float("nan")
    w_adv_lie = arms.get("poison_lie", {}).get("w_adv_mean", float("nan"))
    print(f"Rule: {rule:<18} | reads={reads:<25} | delta_auc={delta_auc:+.4f} | delta_dpd={delta_dpd:+.4f} | w_adv_lie={w_adv_lie}")

s4_projected_sec = (total_sec / 3) * 30
print(f"\nProjected S4 budget (30 seeds x 10 rules x 3 arms = 900 runs): {s4_projected_sec:.0f}s ({s4_projected_sec/60:.1f} min)", flush=True)
