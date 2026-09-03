"""Phase 0 · bước 3 — GATE 0b: TÁI LẬP BIT-EXACT TRÊN CPU.

    VM   : colab exec -s $S -f colab/02_determinism_gate.py --timeout 2400
    local: FEDFAIR_REPO=<repo> FEDFAIR_RESULTS=<out> FEDFAIR_TMP=<tmp> \
           FEDFAIR_DATASETS=german python colab/02_determinism_gate.py

Thay thế cặp `02_gpu_gate.py` + `03_noise_floor.py` cũ. Hai script đó hỏi sai
câu hỏi, và cách hỏi sai đó đã trực tiếp sinh ra một phán quyết sai:

  - `03_noise_floor` đo "sàn nhiễu" = std qua 3 lần chạy cùng seed trên CUDA,
    rồi `02_gpu_gate` đem |ΔAUC| giữa CPU và GPU so với cái sàn đó. Nhưng lệch
    CPU↔GPU KHÔNG cùng loại với lệch run-to-run: dropout mask sinh trên device,
    CPU dùng MT19937 còn CUDA dùng Philox, nên cùng seed vẫn là hai luồng ngẫu
    nhiên khác nhau — tức tương đương đổi seed, không phải sai số float. So một
    hiệu ứng cross-stream với một sàn within-stream là lỗi loại, và `german`
    "FAIL GATE 0b" là hệ quả gần như tất yếu của phép so đó, không phải phát
    hiện về phần cứng.
  - Tệ hơn, bản thân construct "sàn nhiễu" mời gọi đúng cái lỗi ấy: hễ có một
    ngưỡng để vượt là sẽ có người vượt nó bằng nhiễu.

Dự án đã chốt CPU (xem 1_2_phase0.md §2.5 — F22). Trên CPU, xác định là thứ ĐẠT ĐƯỢC chứ
không phải thứ phải đo. Nên gate này nhị phân: chạy cùng cấu hình hai lần, băm
trọng số cuối, đòi TRÙNG KHÍT. Không ngưỡng, không diễn giải.

Nếu PASS: mọi khác biệt về sau giữa hai lần chạy là khác biệt CODE, không phải
phần cứng — và toàn bộ khái niệm "sàn nhiễu" bị loại khỏi dự án.
Nếu FAIL: TRUY NGUYÊN NHÂN rồi sửa (num_threads, use_deterministic_algorithms),
KHÔNG quay lại đo biên độ nhiễu.

Đồng thời trả lời một câu hỏi đang để ngỏ: comment ở `src/config.py:174`
(`use_deterministic_algorithms(False)  # some scatter ops lack deterministic
kernels`) là một GIẢ ĐỊNH CHƯA KIỂM. Script chạy cả hai chế độ để lấy bằng
chứng thay vì niềm tin.
"""
import json
import os
import subprocess

REPO = os.environ.get("FEDFAIR_REPO", "/content/FedFairGNN")
RESULTS = os.environ.get("FEDFAIR_RESULTS", "/content/results")
ROUNDS = 20
SEED = 0
# Tách theo dataset: `bail` trên CPU là ~13 s/round, nhân 2 lần chạy x 2 chế độ
# (và deterministic ép về 1 thread) thì một exec duy nhất sẽ vỡ timeout. Chạy
# `german` trước để lấy tín hiệu, rồi mới tới `bail`.
DATASETS = tuple(os.environ.get("FEDFAIR_DATASETS", "bail").split(","))
OUT = f"{RESULTS}/fairshare/phase0_determinism__{'_'.join(DATASETS)}.json"

RUNNER = r'''
import hashlib, json, os, sys, time, torch
sys.path.insert(0, os.environ.get("FEDFAIR_REPO", "/content/FedFairGNN"))

# Bật/tắt determinism TRƯỚC khi import bất cứ thứ gì dựng kernel.
if os.environ.get("FEDFAIR_DETERMINISTIC") == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from experiments.fairshare_common import make_trainer

ds, seed, rounds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

if os.environ.get("FEDFAIR_DETERMINISTIC") == "1":
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

t0 = time.time()
tr = make_trainer(dataset=ds, seed=seed, rounds=rounds, method="fairshare",
                  device="cpu")
for t in range(rounds):
    tr._round(t)
m = tr.evaluate_global()

# Băm chính trọng số toàn cục: mạnh hơn nhiều so với so hai số float làm tròn.
flat = tr.global_flat.detach().cpu().contiguous().numpy()
digest = hashlib.sha256(flat.tobytes()).hexdigest()

print("JSON " + json.dumps({
    "dataset": ds, "rounds": rounds, "seed": seed,
    "sec_per_round": (time.time() - t0) / rounds,
    "auc": float(m["auc"]), "dpd": float(m["dpd"]),
    "diverged": float(m.get("diverged", 0.0)),
    "weight_sha256": digest,
}), flush=True)
'''


def run(ds, deterministic):
    """Mỗi lần chạy một tiến trình con riêng: kernel giữ state giữa các exec, và
    state rò rỉ (RNG toàn cục, module đã import) là kẻ thù của tái lập. Chính
    điều này khiến 'chạy lại y hệt' trở thành một phép thử thật.

    Bỏ `timeout` phía trước: nó là coreutils, KHÔNG có sẵn trên macOS. Giữ lại
    thì gate sẽ chết ở local với một thông báo không liên quan gì tới xác định."""
    env = dict(os.environ, FEDFAIR_DEVICE="cpu", PYTHONUNBUFFERED="1",
               FEDFAIR_DETERMINISTIC="1" if deterministic else "0")
    r = subprocess.run(
        f"python -u {RUNNER_PATH} {ds} {SEED} {ROUNDS}",
        shell=True, capture_output=True, text=True, env=env, cwd=REPO)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON ")]
    if not line:
        return {"error": (r.stderr or r.stdout)[-1500:]}
    return json.loads(line[0][5:])


RUNNER_PATH = os.path.join(os.environ.get("FEDFAIR_TMP", "/content"), "_det_runner.py")
os.makedirs(os.path.dirname(RUNNER_PATH), exist_ok=True)
open(RUNNER_PATH, "w").write(RUNNER)
os.chdir(REPO)

report = {"rounds": ROUNDS, "seed": SEED, "device": "cpu", "modes": {}}

for mode, det in (("default", False), ("deterministic", True)):
    report["modes"][mode] = {}
    for ds in DATASETS:
        a = run(ds, det)
        if "error" in a:
            report["modes"][mode][ds] = {"error": a["error"]}
            print(f"[{mode}/{ds}] LỖI:\n{a['error']}", flush=True)
            continue
        b = run(ds, det)
        if "error" in b:
            report["modes"][mode][ds] = {"error": b["error"]}
            continue
        same = a["weight_sha256"] == b["weight_sha256"]
        report["modes"][mode][ds] = {
            "bit_identical": same,
            "weight_sha256": [a["weight_sha256"][:16], b["weight_sha256"][:16]],
            "auc": [a["auc"], b["auc"]], "dpd": [a["dpd"], b["dpd"]],
            "d_auc": abs(a["auc"] - b["auc"]), "d_dpd": abs(a["dpd"] - b["dpd"]),
            "diverged": [a["diverged"], b["diverged"]],
            "sec_per_round": round(a["sec_per_round"], 3),
        }
        print(f"[{mode}/{ds}] bit_identical={same}  "
              f"auc={a['auc']:.6f}/{b['auc']:.6f}  "
              f"{a['sec_per_round']:.2f}s/round", flush=True)

# --- phán quyết -------------------------------------------------------------
def all_ok(mode):
    ms = report["modes"].get(mode, {})
    return bool(ms) and all(isinstance(v, dict) and v.get("bit_identical")
                            for v in ms.values())

report["gate_0b_default"] = all_ok("default")
report["gate_0b_deterministic"] = all_ok("deterministic")
# Chỉ cần MỘT chế độ đạt bit-exact là dự án có một nền móng xác định để đứng.
report["gate_0b_pass"] = report["gate_0b_default"] or report["gate_0b_deterministic"]
report["recommended_config"] = (
    "default (use_deterministic_algorithms(False) là đủ trên CPU)"
    if report["gate_0b_default"] else
    "deterministic (bật use_deterministic_algorithms(True) + set_num_threads(1))"
    if report["gate_0b_deterministic"] else
    "KHÔNG CHẾ ĐỘ NÀO BIT-EXACT — phải truy nguyên nhân trước khi chạy Phase 1")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=2)
print(json.dumps(report["modes"], indent=2))
print("\nGATE 0b:", "PASS" if report["gate_0b_pass"] else "*** FAIL ***")
print("Cấu hình nên dùng:", report["recommended_config"])
print("""
→ PASS: xoá khái niệm "sàn nhiễu" khỏi dự án. Mọi |Δ| khác 0 giữa hai lần chạy
  cùng cấu hình từ nay là khác biệt CODE, phải truy, không được diễn giải là nhiễu.
→ FAIL: KHÔNG đo biên độ nhiễu để sống chung với nó. Truy nguyên nhân
  (num_threads, kernel thiếu bản xác định) và sửa.
""")
