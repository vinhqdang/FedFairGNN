"""Phase 1 — GATE 1 (đường ống) + nghiệm thu DoD của SPEC §4.0.

    colab exec -s $S -f colab/09_phase1_rerun.py --timeout 5400

Thay `06_phase1_2_harness.py`, vốn không dùng lại được vì hai lý do:
  - dòng 35 (Phase 1, 50 round) và dòng 62 (Phase 2, 10 round) ghi **chung**
    `--out` ⇒ chung tên `derisk_D1_summary.csv` ⇒ bản 50 round bị bản 10 round
    đè. Đó là cách artifact `R2` biến mất.
  - `FEDFAIR_DEVICE="cuda"` — thiết bị canonical nay là **CPU**.

Script này làm HAI việc tách bạch, vì chúng nghiệm thu hai thứ khác nhau:

  (1) GATE 1 — đường ống. `derisk_phase3` german 50 round. Hỏi: chạy thông
      50 round không, có NaN/Inf không, AUC có nằm trong dải hợp lý không.

  (2) DoD của §4.0 — NaN-guard. **GATE 1 KHÔNG kiểm được cái này.**
      `derisk_phase3` chạy `attack="none"`, mà NaN chỉ phát sinh dưới
      `sign_flip`/`scaling` (gradient nổ vì `attack_intensity=10`). Chạy
      `incentive_audit` để ép đúng hai kịch bản đó và đòi `nan_round_frac = 0`.
      Không có bước này thì "Phase 1 xanh" **không nói gì** về bản vá §4.0.

Provenance: `git rev-parse` trên VM luôn trả rỗng vì `00_pack.sh` loại `./.git`
khỏi tarball — đó mới là nguyên nhân thật của `"commit": "nogit"` trong
`phase2_report.json`, không phải cây làm việc bẩn. Nên commit được đọc từ
`manifest_local.json` do `00_pack.sh` sinh; thiếu file đó thì DỪNG, vì một
artifact không truy được về commit là artifact không dùng làm bằng chứng.
"""
import json
import os
import subprocess
import time

# Chạy được ở CẢ HAI nơi (local / VM). Từ khi F22 chốt canonical là CPU, Colab
# không còn cấp gì mà máy local không có.
REPO = os.environ.get("FEDFAIR_REPO", "/content/FedFairGNN")
RESULTS = os.environ.get("FEDFAIR_RESULTS", "/content/results")
MANIFEST = os.environ.get("FEDFAIR_MANIFEST", "/content/manifest_local.json")

def provenance():
    """Commit hash phải định danh được thứ đang chạy — bằng đường nào cũng được.

    Trên VM: `00_pack.sh` loại `./.git` khỏi tarball nên `git rev-parse` luôn
    rỗng; đó là lý do `manifest_local.json` tồn tại. Ở local thì `.git` CÓ mặt
    nên đọc thẳng git là đúng nguồn. Inline (không import chung) vì
    `colab exec -f` chỉ gửi ĐÚNG một file lên kernel."""
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST))
        m["provenance_source"] = "manifest"
        return m
    git = os.path.join(REPO, ".git")
    if not os.path.exists(git):
        raise SystemExit(f"DỪNG: không có {MANIFEST} và cũng không có {git}.")
    g = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                                  text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"),
            "commit_short": g("rev-parse", "--short", "HEAD"),
            "branch": g("branch", "--show-current"),
            "dirty": bool(g("status", "--porcelain")),
            "tarball_sha256": None, "provenance_source": "git"}



# --- provenance: đọc từ manifest local, KHÔNG đoán ---------------------------
if not os.path.exists(MANIFEST):
    raise SystemExit(
        "DỪNG: thiếu /content/manifest_local.json.\n"
        "Chạy `SP=<scratchpad> bash colab/00_pack.sh` rồi upload cả tarball lẫn "
        "manifest lên VM. Không có nó thì run_id không truy được về commit.")

man = json.load(open(MANIFEST))
if man.get("dirty"):
    raise SystemExit(f"DỪNG: manifest ghi dirty=true (commit {man.get('commit_short')}). "
                     "Commit trước khi chạy — hash không định danh được thứ đang chạy.")

RUN_ID = f"{man['commit_short']}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
OUT_DIR = f"/content/results/fairshare/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/content/data", exist_ok=True)
os.chdir(REPO)
if not os.path.exists(f"{REPO}/data"):
    os.symlink("/content/data", f"{REPO}/data")

import torch
assert not os.environ.get("FEDFAIR_DEVICE") or os.environ["FEDFAIR_DEVICE"] == "cpu"
env = dict(os.environ, FEDFAIR_DEVICE="cpu", PYTHONUNBUFFERED="1")

report = {
    "run_id": RUN_ID, "device": "cpu",
    "commit": man["commit"], "commit_short": man["commit_short"],
    "branch": man.get("branch"), "tarball_sha256": man.get("tarball_sha256"),
    "torch": torch.__version__, "started_at": time.strftime("%FT%TZ", time.gmtime()),
    "gate1": {}, "spec_4_0_dod": {},
}
print(f"RUN_ID  = {RUN_ID}\nOUT_DIR = {OUT_DIR}\ncommit  = {man['commit_short']}\n", flush=True)


def run(name, cmd, budget=3600):
    t0 = time.time()
    r = subprocess.run(f"timeout {budget} {cmd}", shell=True, env=env,
                       capture_output=True, text=True, cwd=REPO)
    with open(f"{OUT_DIR}/{name}.log", "w") as fh:
        fh.write(r.stdout + "\n--- STDERR ---\n" + r.stderr)
    print(f"[{name}] rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1200:], flush=True)
    return r


# ===========================================================================
# (1) GATE 1 — đường ống 50 round
# ===========================================================================
print("=== GATE 1: đường ống, german 50 round ===", flush=True)
run("derisk_phase3_50r",
    f"python -u -m experiments.derisk_phase3 --datasets german --seeds 1 "
    f"--rounds 50 --alphas 0.1 --out {OUT_DIR}")

import csv
p1 = f"{OUT_DIR}/derisk_D1_summary.csv"
if not os.path.exists(p1):
    report["gate1"] = {"csv_exists": False}
else:
    rows = list(csv.DictReader(open(p1)))

    def fnum(x):
        try:
            v = float(x)
            return v if v == v and abs(v) != float("inf") else None
        except (TypeError, ValueError):
            return None

    aucs = [fnum(r.get("auc_mean")) for r in rows]
    dpds = [fnum(r.get("dpd_mean")) for r in rows]
    finite = all(v is not None for v in aucs + dpds)
    in_range = finite and all(0.55 <= v <= 0.85 for v in aucs)
    report["gate1"] = {
        "csv_exists": True, "rows": len(rows),
        "all_finite": finite, "auc_in_range": in_range,
        "verdict_json": os.path.exists(f"{OUT_DIR}/derisk_D1_verdict.json"),
        "auc_mean": aucs, "dpd_mean": dpds,
        "methods": [r.get("method") for r in rows],
    }
report["gate1"]["pass"] = bool(report["gate1"].get("all_finite")
                               and report["gate1"].get("auc_in_range")
                               and report["gate1"].get("verdict_json"))
print("GATE 1:", "PASS" if report["gate1"]["pass"] else "*** FAIL ***", flush=True)

# ===========================================================================
# (2) DoD §4.0 — NaN-guard dưới ĐÚNG hai kịch bản từng sinh NaN
# ===========================================================================
print("\n=== DoD §4.0: NaN-guard dưới sign_flip / scaling ===", flush=True)
run("incentive_audit_nanguard",
    f"python -u -m experiments.incentive_audit --dataset german --rounds 10 "
    f"--seeds 1 --num_clients 5 --num_byzantine 1 --out {OUT_DIR}")

pa = f"{OUT_DIR}/incentive_audit__german.csv"
if not os.path.exists(pa):
    report["spec_4_0_dod"] = {"csv_exists": False, "pass": False}
else:
    rows = list(csv.DictReader(open(pa)))
    bad = []
    for r in rows:
        try:
            nf = float(r.get("nan_round_frac", "nan"))
        except ValueError:
            nf = float("nan")
        if not (nf == 0.0):
            bad.append({"aggregator": r.get("aggregator"), "attack": r.get("attack"),
                        "nan_round_frac": r.get("nan_round_frac")})
    # Trước bản vá: fu_shapley/sign_flip = 0.8, /scaling = 0.4, robust_fu/sign_flip = 0.8.
    report["spec_4_0_dod"] = {
        "csv_exists": True, "rows": len(rows),
        "rows_with_nan": bad, "pass": len(bad) == 0,
        "table": [{k: r.get(k) for k in
                   ("aggregator", "attack", "auc", "dpd", "atk_w_mass",
                    "nan_round_frac", "degenerate_frac")} for r in rows],
    }
print("DoD §4.0:", "PASS" if report["spec_4_0_dod"].get("pass") else "*** FAIL ***", flush=True)
if report["spec_4_0_dod"].get("rows_with_nan"):
    print("  còn NaN ở:", json.dumps(report["spec_4_0_dod"]["rows_with_nan"], ensure_ascii=False), flush=True)

report["finished_at"] = time.strftime("%FT%TZ", time.gmtime())
report["phase1_pass"] = bool(report["gate1"]["pass"] and report["spec_4_0_dod"].get("pass"))
json.dump(report, open(f"{OUT_DIR}/phase1_report.json", "w"), indent=2)

print("\n" + "=" * 62)
print(json.dumps({k: report[k] for k in ("run_id", "commit_short", "device")}, indent=2))
print("GATE 1        :", "PASS" if report["gate1"]["pass"] else "FAIL")
print("DoD §4.0      :", "PASS" if report["spec_4_0_dod"].get("pass") else "FAIL")
print("PHASE 1 TỔNG  :", "PASS" if report["phase1_pass"] else "*** FAIL ***")
print(f"artifact      : {OUT_DIR}")
print("""
⚠️ GATE 1 xanh CHỈ chứng nhận đường ống chạy được. Nó không chứng nhận kết quả
   khoa học nào, và cấm suy diễn từ 1 seed (xem 1_3_phase1.md §3.3 và F11).
""")
