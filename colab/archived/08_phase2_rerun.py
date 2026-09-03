"""Phase 2 re-run — 6-script harness, evidentiary edition.

Fixes the three reasons the previous Phase 2 could not be graded:
  B2  outputs go to results/fairshare/<run_id>/ so a run can never be merged
      into, or overwritten by, another lineage.
  B3  every script's stdout+stderr is written to <name>.log, and the machine
      verdict is written to phase2_report.json (the previous run used
      07_run_single_harness.py, which never wrote a report at all).
  #4  exact_shapley_correlation uses the runbook default --probe_at
      (8 12 16 20 24) instead of a single probe round.

Run via:
  colab exec -s $S -f colab/08_phase2_rerun.py --timeout 3600
"""
import json
import os
import glob
import subprocess
import time

REPO = "/content/FedFairGNN"
os.chdir(REPO)

# 00_pack.sh loại ./.git khỏi tarball, nên `git rev-parse` trên VM LUÔN rỗng --
# đó mới là nguyên nhân thật của "commit": "nogit", không phải cây bẩn. Đọc
# provenance từ manifest do 00_pack.sh sinh; thiếu nó thì DỪNG.
MANIFEST = "/content/manifest_local.json"
if not os.path.exists(MANIFEST):
    raise SystemExit("DỪNG: thiếu /content/manifest_local.json (chạy 00_pack.sh rồi upload).")
man = json.load(open(MANIFEST))
if man.get("dirty"):
    raise SystemExit(f"DỪNG: manifest ghi dirty=true (commit {man.get('commit_short')}).")
commit = man["commit_short"]
RUN_ID = f"{commit}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
OUT_DIR = f"/content/results/fairshare/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/content/data", exist_ok=True)
if not os.path.exists(f"{REPO}/data"):
    os.symlink("/content/data", f"{REPO}/data")

import torch
# Thiết bị canonical là CPU (chốt 2026-08-03): chỉ CPU mới tái lập bit-exact.
# Xem 1_2_phase0.md §2.5 (F22) và §2.6.
DEVICE = "cpu"
env = dict(os.environ, FEDFAIR_DEVICE="cpu", PYTHONUNBUFFERED="1")

print(f"RUN_ID  = {RUN_ID}")
print(f"OUT_DIR = {OUT_DIR}")
print(f"DEVICE  = {DEVICE}\n", flush=True)

O = f"--out {OUT_DIR}"
scripts = [
    ("derisk_phase3",
     f"python -u -m experiments.derisk_phase3 --datasets german --seeds 1 --rounds 10 --alphas 0.1 {O}"),
    ("incentive_audit",
     f"python -u -m experiments.incentive_audit --dataset german --rounds 10 --seeds 1 --num_clients 5 --num_byzantine 1 {O}"),
    # no --probe_at -> runbook default [8,12,16,20,24]; --rounds is ignored then.
    ("exact_shapley_correlation",
     f"python -u -m experiments.exact_shapley_correlation --dataset german --seed 0 --alpha 0.1 --game loss --normalize target_norm {O}"),
    ("ablation_val_size",
     f"python -u -m experiments.ablation_val_size --dataset german --seed 0 --rounds 10 --alpha 0.1 --sizes 50 100 {O}"),
    ("topology_shapley_analysis",
     f"python -u -m experiments.topology_shapley_analysis --dataset german --seed 0 --rounds 10 --alpha 0.3 --num_clients 5 {O}"),
    ("plot_shapley",
     f"python -u -m experiments.plot_shapley --dataset german --seed 0 --rounds 10 --alpha 0.1 {O}"),
]

results = {}
for name, cmd in scripts:
    print(f"--- {name} ---", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True, cwd=REPO)
    dt = round(time.time() - t0, 1)
    with open(f"{OUT_DIR}/{name}.log", "w") as fh:
        fh.write(f"$ {cmd}\n\n=== STDOUT ===\n{r.stdout}\n\n=== STDERR ===\n{r.stderr}\n")
    results[name] = {"cmd": cmd, "returncode": r.returncode,
                     "success": r.returncode == 0, "seconds": dt}
    print(f"{'OK ' if r.returncode == 0 else 'FAIL'} {name}  ({dt}s)", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1200:], flush=True)

csvs = sorted(os.path.basename(f) for f in glob.glob(f"{OUT_DIR}/*.csv"))
pngs = sorted(os.path.basename(f) for f in glob.glob(f"{OUT_DIR}/**/*.png", recursive=True))

report = {
    # Luôn False: guard ở đầu file đã DỪNG nếu manifest ghi dirty=true. Ghi vào
    # report để provenance tự mô tả được, không phải để suy luận.
    "run_id": RUN_ID, "commit": commit, "dirty": bool(man.get("dirty", False)),
    "device": DEVICE,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "torch": torch.__version__, "out_dir": OUT_DIR,
    "scripts": results, "csv_files": csvs, "png_files": pngs,
    "all_scripts_ok": all(v["success"] for v in results.values()),
    "has_audit_traj": len(glob.glob(f"{OUT_DIR}/audit_traj__*.csv")) > 0,
    "has_d1_verdict": os.path.exists(f"{OUT_DIR}/derisk_D1_verdict.json"),
}
report["gate2_pass"] = bool(report["all_scripts_ok"] and report["has_audit_traj"]
                            and report["has_d1_verdict"])
json.dump(report, open(f"{OUT_DIR}/phase2_report.json", "w"), indent=2)

print("\n" + "=" * 60)
print(f"GATE 2 (infra): {'PASS' if report['gate2_pass'] else 'FAIL'}")
print(f"CSV: {len(csvs)}  PNG: {len(pngs)}  logs: {len(results)}")
print(json.dumps({k: v for k, v in report.items() if k != "csv_files"}, indent=2))
