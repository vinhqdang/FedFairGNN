"""Phase 1 · GATE 1-C — kiểm chuỗi toán §1.3–§1.4 trên đường code thật sự ship.

    colab exec -s $S -f colab/10_phase1_checkpoints.py --timeout 3600

Suy ra từ đâu (không phải chọn bừa)
-----------------------------------
`0_` §1.3–§1.4 định nghĩa một CHUỖI, mỗi mũi tên là một chỗ có thể hỏng:

    D_val ──► (g_task, g_fair) ──► g_target = g_task + α·g_fair
    g_k   ──► φ_k = ⟨g_k, g_target⟩ = φ_util + φ_fair
    φ_k   ──► φ̄_k  (EMA, β)
    φ̄_k   ──► w_k = ReLU(φ̄_k)/Σ ReLU(φ̄_j)
    w_k   ──► θ_{t+1} = θ_t − Σ w_k g_k ──► ŷ ──► AUC, DPD, EOD

`1_2` (Phase 0) đã chốt CPU **bit-exact**. Hệ quả trực tiếp lên thiết kế này:
một hàm xác định của đầu vào cố định chỉ cần đánh giá **một lần** — lặp lại
không thêm thông tin nào. Nhưng Phase 0 **không** nói gì về đường code thật
trên dữ liệu thật theo thời gian: unit test chạy đồ thị synthetic 160 node, và
`P0` đã từng cho thấy test khoá **nhầm** code path.

⇒ Phase 1 = kiểm chuỗi trên đường ship, theo trục thời gian.

Yêu cầu round/seed suy từ LOẠI tính chất, không đặt một hằng số chung
---------------------------------------------------------------------
  E — điều kiện tồn tại   đỏ ⇒ mọi claim phía sau vô nghĩa, kiểm TRƯỚC
  I — đồng nhất thức      đúng chính xác với mọi đầu vào ⇒ 1 lần đánh giá là đủ
  T — hợp đồng thời gian  liên quan đệ quy EMA ⇒ số round suy từ β
  P — tính chất phân bố   phụ thuộc phân hoạch Dirichlet ⇒ 1 seed = 0 bằng chứng

**Số round của nhóm T suy từ β, không phải chọn.** β = 0.9 ⇒ hằng số thời gian
τ = 1/(1−β) = 10 round. Bộ nhớ điều kiện đầu (φ̄₀ = 0) tắt theo β^t, còn 21% ở
t=15 và 5% ở t≈28. ⇒ mọi phát biểu về **hành vi** EMA cần **≥ 30 round (3τ)**;
riêng **đồng nhất thức** đệ quy chỉ cần 2 round liên tiếp.

**Số seed của nhóm P suy từ A3.** Giả định A3 nói non-IID là *điều kiện tồn tại*
của novelty ⇒ phân hoạch Dirichlet cố ý dị thể mạnh ⇒ hành vi gate đổi theo
từng lần bốc. Một seed là một lần bốc: không kết luận được gì. Nhóm P vì vậy
báo **tỉ lệ**, không báo nhị phân.

Cái Phase 1 KHÔNG làm: không so hiệu năng, không p-value, không xếp hạng
aggregator. A2 (dấu φ có đúng chiều đóng góp không) cần exact-SV làm chuẩn ⇒
D9/Phase 3. Ở đây chỉ kiểm **điều kiện cần** của A2: dấu φ có ổn định không.
"""
import json
import os
import subprocess
import time

# Chạy được ở CẢ HAI nơi. Từ khi F22 chốt thiết bị canonical là CPU, Colab không
# còn cấp gì mà máy local không có; giữ đường VM chỉ để không phá quy trình cũ.
REPO = os.environ.get("FEDFAIR_REPO", "/content/FedFairGNN")
RESULTS = os.environ.get("FEDFAIR_RESULTS", "/content/results")
MANIFEST = os.environ.get("FEDFAIR_MANIFEST", "/content/manifest_local.json")

DATASET = "german"
CLIENTS = 5
HOLDOUT = 100
BETA = 0.9          # phải khớp cfg.fu_ema_beta — checkpoint E0 ghim điều đó
TAU = 1.0 / (1.0 - BETA)
R_TEMPORAL = 30     # 3τ — xem docstring
R_CONVERGE = 50     # đủ dài để đường học chạm plateau (bằng GATE 1)
S_DIST = 5          # nhóm P
S_DIR = 3           # nhóm P kiểm hướng
TOL = 1e-5

def provenance():
    """Commit hash phải định danh được thứ đang chạy — bằng đường nào cũng được.

    Trên VM: `00_pack.sh` loại `./.git` khỏi tarball nên `git rev-parse` luôn
    rỗng; đó là lý do `manifest_local.json` tồn tại. Ở local thì `.git` CÓ mặt,
    nên đọc thẳng git là đúng nguồn và bớt một bước đồng bộ có thể lệch.
    Hàm này inline (không import chung) vì `colab exec -f` chỉ gửi ĐÚNG một file
    lên kernel — mọi import anh em sẽ chết trên VM."""
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST))
        m["provenance_source"] = "manifest"
        return m
    git = os.path.join(REPO, ".git")
    if not os.path.exists(git):
        raise SystemExit(
            f"DỪNG: không có {MANIFEST} và cũng không có {git}.\n"
            "Trên VM: chạy 00_pack.sh rồi upload manifest. "
            "Ở local: đặt FEDFAIR_REPO trỏ vào cây git của FedFairGNN.")
    g = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                                  text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"),
            "commit_short": g("rev-parse", "--short", "HEAD"),
            "branch": g("branch", "--show-current"),
            "dirty": bool(g("status", "--porcelain")),
            "tarball_sha256": None, "provenance_source": "git"}


man = provenance()
if man.get("dirty"):
    raise SystemExit(
        f"DỪNG: cây làm việc bẩn (commit {man.get('commit_short')}). Commit "
        "trước — một artifact không truy được về commit là artifact không dùng "
        "làm bằng chứng.")

RUN_ID = f"{man['commit_short']}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
OUT_DIR = f"{RESULTS}/fairshare/phase1c/{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO)
# Trên VM `data/` là symlink sang /content/data (đĩa VM). Ở local repo đã có sẵn
# `data/` của chính nó — không đụng vào.
if not os.path.exists(f"{REPO}/data") and os.path.isdir("/content"):
    os.makedirs("/content/data", exist_ok=True)
    os.symlink("/content/data", f"{REPO}/data")

import sys
sys.path.insert(0, REPO)
os.environ.setdefault("FEDFAIR_DEVICE", "cpu")

import torch
from experiments.fairshare_common import make_trainer
from src.federated.aggregation import aggregate
from src.federated.client import load_flat_state
from src.trust.incentive import get_server_target_gradients

import platform
ENV = {"machine": platform.machine(), "system": platform.system(),
       "python": platform.python_version(), "torch": torch.__version__,
       "threads": torch.get_num_threads()}
print(f"RUN_ID = {RUN_ID}\ncommit = {man['commit_short']} ({man['provenance_source']})")
print(f"env    = {ENV['system']}/{ENV['machine']} py{ENV['python']} torch{ENV['torch']}")
print(f"β = {BETA} ⇒ τ = {TAU:.0f} round ⇒ nhóm T dùng {R_TEMPORAL} round (3τ)\n", flush=True)

CHECKS = []


def check(cid, group, what, passed, requires, why, **detail):
    v = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
    CHECKS.append({"id": cid, "group": group, "what": what, "verdict": v,
                   "requires": requires, "why_it_matters": why, **detail})
    print(f"{ {'PASS': '✅', 'FAIL': '❌'}.get(v, '··') } {cid:<5} {what}", flush=True)


def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and x == x and abs(x) != float("inf")


def run(method, attack, rounds, seed, **over):
    tr = make_trainer(dataset=DATASET, seed=seed, num_clients=CLIENTS,
                      rounds=rounds, method=method, attack=attack,
                      num_byzantine=(0 if attack == "none" else 1),
                      fu_val_source="server_holdout", fu_holdout_size=HOLDOUT,
                      **over)
    return tr, tr.run(verbose=False)


def runs(method, attack, rounds, n_seeds, **over):
    out = []
    for s in range(n_seeds):
        t0 = time.time()
        tr, res = run(method, attack, rounds, s, **over)
        out.append((s, tr, res))
        print(f"   · {method:<18} {attack:<16} s{s} {rounds}r {time.time()-t0:5.1f}s", flush=True)
    return out


def col(hist, key):
    return [h.get(key) for h in hist]


# =============================================================================
# E — ĐIỀU KIỆN TỒN TẠI.  Đỏ ở đây ⇒ dừng, mọi checkpoint sau vô nghĩa.
# =============================================================================
print("--- E: điều kiện tồn tại ---", flush=True)
MAIN = runs("fairshare", "none", R_CONVERGE, S_DIST)
tr0 = MAIN[0][1]

check("E0", "E", f"cfg.fu_ema_beta khớp hằng số suy round ({BETA})",
      abs(float(tr0.cfg.fu_ema_beta) - BETA) < 1e-12,
      "1 lần đánh giá",
      "Cả R_TEMPORAL=30 lẫn checkpoint T1/T3 đều suy từ β. Lệch β ⇒ số round "
      "đang dùng là sai, và T sẽ đỏ vì sai hằng số chứ không vì sai code.",
      cfg_beta=float(tr0.cfg.fu_ema_beta), tau=TAU)

# E1 — mô hình có học, và có CHẠM PLATEAU chưa (không chỉ "có tăng")
learn = []
for s, tr, res in MAIN:
    a = [x for x in col(res["history"], "g_auc") if isnum(x)]
    if len(a) < 10:
        learn.append({"seed": s, "auc": None}); continue
    tail = a[int(0.8 * len(a)):]
    learn.append({"seed": s, "auc_first": round(a[0], 4), "auc_last": round(a[-1], 4),
                  "gain": round(max(a) - a[0], 4),
                  "tail_slope": round((tail[-1] - tail[0]) / max(1, len(tail)), 5)})
ok_learn = [d for d in learn if d.get("gain") is not None
            and d["gain"] > 0.02 and d["auc_last"] >= 0.60]
plateau = [d for d in learn if d.get("tail_slope") is not None
           and abs(d["tail_slope"]) < 0.002]
check("E1", "E", f"mô hình học VÀ chạm plateau trong {R_CONVERGE} round",
      len(ok_learn) >= 4 and len(plateau) >= 4,
      f"{R_CONVERGE} round (phải chạm plateau) × {S_DIST} seed",
      "GATE 1 chỉ kiểm AUC CUỐI ∈ [0.55,0.85] — không phân biệt 'đã học' với "
      "'khởi tạo may', và cũng không phân biệt 'đã hội tụ' với 'còn đang leo'. "
      "Nếu chưa plateau thì mọi so sánh sau này đang so hai mô hình chưa học xong.",
      per_seed=learn, n_learned=len(ok_learn), n_plateau=len(plateau))

# E2 — A3: cơ chế có thật sự làm gì khác FedAvg không
dev_uni = []
for s, tr, res in MAIN:
    ws = [h["agg_weights"] for h in res["history"][int(3 * TAU):] if h.get("agg_weights")]
    if not ws:
        continue
    d = [sum(abs(x - 1.0 / CLIENTS) for x in w) for w in ws]
    dev_uni.append({"seed": s, "mean_l1_dev": round(sum(d) / len(d), 4)})
check("E2", "E", "A3: w lệch khỏi uniform (cơ chế không phải no-op)",
      bool(dev_uni) and sum(d["mean_l1_dev"] for d in dev_uni) / len(dev_uni) > 0.05,
      f"sau 3τ={int(3*TAU)} round × {S_DIST} seed",
      "A3 nói non-IID là ĐIỀU KIỆN TỒN TẠI của novelty: nếu w ≈ 1/K thì "
      "FairShare = FedAvg và MỌI claim FS-* đều rỗng, bất kể chúng có 'thắng' "
      "hay không. Đo sau 3τ vì trước đó w còn mang điều kiện đầu của EMA.",
      per_seed=dev_uni, uniform=round(1.0 / CLIENTS, 4),
      dirichlet_alpha=float(tr0.cfg.dirichlet_alpha))


# --- probe một round: bắt nguyên liệu thô của chuỗi toán ---------------------
def capture(tr):
    """Chụp MỘT lần. Không được gọi `c.train()` hai lượt: dropout tiêu RNG khác
    nhau ⇒ gradient khác nhau ⇒ mọi so sánh sau đó đo nhầm nguồn khác biệt."""
    upd, metas = [], []
    for c in tr.clients:
        c.set_flat(tr.global_flat)
        c.train()
        upd.append(tr.global_flat - c.get_flat())
        metas.append(c.meta())
    load_flat_state(tr.ref_model, tr.global_flat.to(tr.device))
    tg = get_server_target_gradients(tr.ref_model, tr.server_holdout.to(tr.device),
                                     tr.cfg.fu_alpha,
                                     fair_surrogate=tr.cfg.fu_fair_surrogate)
    return upd, metas, tuple(g.cpu() for g in tg)


def agg_with(tr, upd, metas, tgt):
    gt, gk, gf = tgt
    return aggregate(tr.cfg.aggregator, upd, metas, state={},
                     g_target=gt, g_task=gk, g_fair=gf,
                     fu_alpha=tr.cfg.fu_alpha, fu_beta_ema=tr.cfg.fu_ema_beta,
                     fu_normalize=tr.cfg.fu_normalize, fu_score=tr.cfg.fu_score,
                     fu_grad_clip=tr.cfg.fu_grad_clip,
                     tau=tr.cfg.fairness_budget, krum_f=tr.cfg.krum_f)


tr_p, _ = run("fairshare", "none", int(TAU), 0)
UPD, METAS, TGT = capture(tr_p)
g_task, g_fair = TGT[1], TGT[2]
ratio = float((tr_p.cfg.fu_alpha * g_fair).norm() / (g_task.norm() + 1e-12))
check("E3", "E", "nhánh fairness của g_target không suy biến",
      ratio > 1e-3,
      "1 lần đánh giá",
      "g_target = g_task + α·g_fair. Nếu ‖α·g_fair‖/‖g_task‖ ≈ 0 thì φ_fair là "
      "nhiễu số học quanh 0: FS-1 mất chữ 'fairness-aware' và FS-3 (phân rã) "
      "mất luôn ý nghĩa — bất kể α được đặt bằng bao nhiêu.",
      alpha=float(tr_p.cfg.fu_alpha),
      norm_ratio=round(ratio, 6),
      g_task_norm=round(float(g_task.norm()), 6),
      alpha_g_fair_norm=round(float((tr_p.cfg.fu_alpha * g_fair).norm()), 6))

# =============================================================================
# I — ĐỒNG NHẤT THỨC.  Đúng chính xác ⇒ 1 lần đánh giá là đủ (Phase 0 bit-exact).
# =============================================================================
print("\n--- I: đồng nhất thức ---", flush=True)

# I1 — φ = φ_util + φ_fair (tuyến tính của tích vô hướng)
dec_err = 0.0
for h in MAIN[0][2]["history"]:
    pr, pu, pf = h.get("phi_raw"), h.get("phi_util"), h.get("phi_fair")
    if pr is None or pu is None or pf is None:
        continue
    for a, b, c in zip(pr, pu, pf):
        if isnum(float(a)) and isnum(float(b)) and isnum(float(c)):
            dec_err = max(dec_err, abs(float(a) - (float(b) + float(c))))
check("I1", "I", "FS-3: φ = φ_util + φ_fair",
      dec_err < TOL,
      "1 lần đánh giá (kiểm free trên mọi round có sẵn)",
      "Cộng tính là ĐIỀU KIỆN TỒN TẠI của phép phân rã, và F15 đã chốt nó "
      "KHÔNG đúng với cosine. Sai ⇒ hai cột phi_util/phi_fair là số trang trí.",
      max_abs_error=dec_err, tol=TOL, score=tr0.cfg.fu_score)

# I2 — w = ReLU(φ̄)/ΣReLU(φ̄): bất biến MỖI round
gate_bad, simplex_bad, fb_unlabelled = [], [], 0
for s, tr, res in MAIN:
    for h in res["history"]:
        w, ema = h.get("agg_weights"), h.get("phi_ema")
        if not w or ema is None:
            continue
        if h.get("fu_fallback"):
            if not str(h.get("fu_fallback")).strip():
                fb_unlabelled += 1
            continue
        if abs(sum(w) - 1.0) > 1e-5:
            simplex_bad.append({"seed": s, "round": h.get("round"), "sum": sum(w)})
        for k, (wk, ek) in enumerate(zip(w, [float(v) for v in ema])):
            if (ek <= 0) != (wk == 0.0):
                gate_bad.append({"seed": s, "round": h.get("round"), "client": k,
                                 "phi_ema": ek, "w": wk})
check("I2", "I", "FS-2 tầng 1: φ̄ ≤ 0 ⟺ w = 0, và Σw = 1",
      not gate_bad and not simplex_bad and fb_unlabelled == 0,
      "bất biến ⇒ kiểm MỌI round của mọi run có sẵn",
      "Nếu quan hệ này không đúng từng round thì 'gate' chỉ là một phép co "
      "giãn, và câu 'cơ chế đã zero attacker' là đọc sai cái đang chạy. "
      "Đ3 đòi fallback có nhãn — fallback không nhãn thì mọi thống kê về cơ "
      "chế đang trộn 'đo hỏng' với 'mọi client đóng góp âm'.",
      n_gate_violations=len(gate_bad), examples=gate_bad[:5],
      n_simplex_violations=len(simplex_bad), n_fallback_unlabelled=fb_unlabelled)

# I3 — θ_{t+1} = θ_t − Σ w_k g_k
g_agg, info_p = agg_with(tr_p, UPD, METAS, TGT)
w_p = info_p["weights"]
expect = sum(wk * u for wk, u in zip(w_p, UPD))
agg_err = float((g_agg - expect).abs().max())
check("I3", "I", "θ_{t+1} = θ_t − Σ w_k g_k (tổng hợp đúng trọng số)",
      agg_err < 1e-6,
      "1 lần đánh giá",
      "Mũi tên cuối của chuỗi §1.4. Nếu g_agg không đúng là tổ hợp lồi theo w "
      "thì mọi thứ đo được về w không nói gì về mô hình phát hành.",
      max_abs_error=agg_err)

# I4 — w độc lập metadata, KÈM đối chứng dương
def lie(ms):
    out = [dict(m) for m in ms]
    for m in out:
        m["dpd"], m["eod"], m["perf"], m["loss"] = 0.0, 0.0, 0.99, 0.01
    return out


w_honest = w_p                                    # cùng UPD/TGT đã chụp ở trên
w_lying = agg_with(tr_p, UPD, lie(METAS), TGT)[1]["weights"]
d_ours = max(abs(a - b) for a, b in zip(w_honest, w_lying))

tr_bf, _ = run("fedfairgnn-nodp", "none", int(TAU), 0)
U2, M2, T2 = capture(tr_bf)
d_bfwa = max(abs(a - b) for a, b in zip(
    agg_with(tr_bf, U2, M2, T2)[1]["weights"],
    agg_with(tr_bf, U2, lie(M2), T2)[1]["weights"]))

check("I4", "I", "P1/N1: w bỏ qua HOÀN TOÀN metadata client tự khai",
      d_ours == 0.0 and d_bfwa > 0.0,
      "1 lần đánh giá + 1 đối chứng dương",
      "Claim trung tâm của cả bài (C3): server không cần TIN self-report. "
      "Metadata không có trong công thức φ nên đây là đồng nhất thức, kiểm "
      "chính xác trong 1 round — không cần seed, không cần p-value. Đối chứng "
      "BFWA BẮT BUỘC phải lệch: nếu nó cũng bằng 0 thì phép thử không có sức "
      "phân biệt và I4 chưa chứng minh được gì.",
      max_weight_delta_ours=d_ours, max_weight_delta_bfwa=d_bfwa,
      control_has_power=bool(d_bfwa > 0.0),
      w_honest=[round(x, 6) for x in w_honest],
      w_lying=[round(x, 6) for x in w_lying])

# I5 — A1: server_holdout thật sự được carve ra
from src.data import load_dataset
_d = load_dataset(DATASET, root=tr0.cfg.data_root, seed=0)
n_total = int(_d.num_nodes)
n_ho = int(tr0.server_holdout.num_nodes) if tr0.server_holdout is not None else 0
n_cli = sum(int(d.num_nodes) for d in tr0.clients_data)
check("I5", "I", "A1: holdout được carve ra và không client nào giữ",
      n_ho > 0 and n_ho + n_cli == n_total,
      "1 lần đánh giá (kiểm tập hợp)",
      "`carve_server_holdout` trả None khi thiếu node val, và trainer khi đó "
      "ÂM THẦM rơi về pooled — đúng vòng tròn F10 (g_target dựng từ val của cả "
      "attacker), KHÔNG một dòng log nào. Đây là đường hỏng im lặng, và nó làm "
      "hỏng địa vị chứng cứ của mọi số Phase 1 chứ không chỉ của claim A1.",
      n_total=n_total, n_holdout=n_ho, n_clients_total=n_cli,
      requested=HOLDOUT,
      note="take = min(size, len(val_idx)//2) nên số thực nhận có thể nhỏ hơn yêu cầu")

# =============================================================================
# T — HỢP ĐỒNG THỜI GIAN.  Số round suy từ β (xem docstring).
# =============================================================================
print(f"\n--- T: hợp đồng thời gian ({R_TEMPORAL} round = 3τ) ---", flush=True)
SF = runs("fairshare", "sign_flip", R_TEMPORAL, S_DIST)


def ema_scan(hist, beta):
    rec_err, holds, hold_ok = 0.0, 0, 0
    prev = None
    for h in hist:
        ema, phn = h.get("phi_ema"), h.get("phi_norm")
        if ema is None or phn is None:
            continue
        ema = [float(v) for v in ema]
        phn = [float(v) for v in phn]
        if prev is not None:
            for k, (e, p) in enumerate(zip(ema, phn)):
                if isnum(p):
                    rec_err = max(rec_err, abs(e - (beta * prev[k] + (1 - beta) * p)))
                else:
                    holds += 1
                    hold_ok += int(e == prev[k])
        prev = ema
    return rec_err, holds, hold_ok


err_main = max(ema_scan(r[2]["history"], BETA)[0] for r in MAIN)
check("T1", "T", "FS-1b: đệ quy EMA khớp φ̄ₜ = β·φ̄ₜ₋₁ + (1−β)·φₜ",
      err_main < TOL,
      "2 round liên tiếp là đủ (đồng nhất thức) — kiểm trên toàn bộ run có sẵn",
      "Đây là ĐỒNG NHẤT THỨC nên không cần 3τ. Tách nó khỏi T3 là cố ý: sai đệ "
      "quy và sai hành vi làm mượt là hai lỗi khác nhau, cần hai phép kiểm khác nhau.",
      max_abs_error=err_main, tol=TOL)

holds_tot = hold_ok_tot = 0
for s, tr, res in SF:
    _, hh, hk = ema_scan(res["history"], BETA)
    holds_tot += hh; hold_ok_tot += hk
check("T2", "T", "Đ2: φ mất hữu hạn ⇒ EMA GIỮ giá trị round trước",
      (hold_ok_tot == holds_tot) if holds_tot else None,
      f"{R_TEMPORAL} round dưới sign_flip × {S_DIST} seed — cần sự kiện xảy ra",
      "NaN là ĐIỂM HÚT của đệ quy EMA (§1.3.1 Đ2): một lần dính thì mọi round "
      "sau đều NaN kể cả khi φ đã hồi phục, và vì `nan > 0` là False, cơ chế "
      "âm thầm rơi về FedAvg — trao attacker đúng 1/K, không báo fallback. "
      "Cần horizon dài vì phải quan sát cả GIỮ lẫn HỒI PHỤC.",
      n_nonfinite_coords=holds_tot, n_held_correctly=hold_ok_tot,
      note="holds=0 ⇒ sự kiện không xảy ra ⇒ N/A, KHÔNG phải PASS")

# T3 — EMA có thật sự làm mượt không (đây mới là claim giá trị của FS-1b)
smooth = []
for s, tr, res in MAIN:
    hs = res["history"][int(TAU):]          # bỏ 1τ đầu: còn mang điều kiện đầu
    for k in range(CLIENTS):
        raw = [float(h["phi_norm"][k]) for h in hs
               if h.get("phi_norm") and isnum(float(h["phi_norm"][k]))]
        sm = [float(h["phi_ema"][k]) for h in hs
              if h.get("phi_ema") and isnum(float(h["phi_ema"][k]))]
        if len(raw) > 5 and len(sm) > 5:
            var = lambda v: sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)
            smooth.append({"seed": s, "client": k,
                           "var_raw": round(var(raw), 8), "var_ema": round(var(sm), 8),
                           "smoothed": var(sm) < var(raw)})
check("T3", "T", "FS-1b: EMA thật sự làm mượt (var(φ̄) < var(φ))",
      bool(smooth) and sum(d["smoothed"] for d in smooth) >= 0.8 * len(smooth),
      f"≥3τ = {R_TEMPORAL}+ round, bỏ 1τ đầu × {S_DIST} seed",
      "T1 chỉ nói code chạy ĐÚNG công thức; T3 hỏi công thức có MUA được gì "
      "không. Đây là chỗ 15 round là sai: β^15 = 0.21 nên EMA còn giữ 21% điều "
      "kiện đầu, phương sai đo được lúc đó là phương sai của quá trình quá độ.",
      n_series=len(smooth), n_smoothed=sum(d["smoothed"] for d in smooth),
      examples=smooth[:5])

# T4 — điều kiện CẦN của A2: dấu φ có ổn định không
sign_stab = []
for s, tr, res in MAIN:
    hs = res["history"][int(TAU):]
    for k in range(CLIENTS):
        sg = [(1 if float(h["phi_norm"][k]) > 0 else -1) for h in hs
              if h.get("phi_norm") and isnum(float(h["phi_norm"][k]))]
        if len(sg) > 5:
            sign_stab.append({"seed": s, "client": k,
                              "stability": round(abs(sum(sg) / len(sg)), 4)})
med_stab = sorted(d["stability"] for d in sign_stab)[len(sign_stab) // 2] if sign_stab else None
check("T4", "T", "điều kiện CẦN của A2: dấu φ ổn định, không lật như tung đồng xu",
      (med_stab >= 0.5) if med_stab is not None else None,
      f"≥3τ = {R_TEMPORAL}+ round × {S_DIST} seed",
      "A2 (dấu φ phản ánh đúng CHIỀU đóng góp) chỉ phán quyết được bằng "
      "exact-SV làm chuẩn ⇒ D9/Phase 3. Nhưng ĐIỀU KIỆN CẦN thì kiểm được "
      "ngay, không cần chuẩn: nếu sign(φ_k) lật gần như ngẫu nhiên qua các "
      "round thì EMA đang làm mượt nhiễu và gate là tuỳ tiện — A2 sai mà "
      "không cần đợi Phase 3. 1.0 = ổn định tuyệt đối, 0.0 = tung đồng xu.",
      median_stability=med_stab, per_series=sign_stab[:10],
      note="Đây KHÔNG phải phép kiểm A2. Xanh không xác nhận A2; đỏ thì bác A2.")

# =============================================================================
# P — TÍNH CHẤT PHÂN BỐ.  Nhiều seed vì phụ thuộc phân hoạch Dirichlet (A3).
# =============================================================================
print(f"\n--- P: phân bố ({S_DIST} seed) ---", flush=True)

benign_rate = []
for s, tr, res in MAIN:
    hs = [h for h in res["history"][int(TAU):] if h.get("agg_weights")]
    z = sum(1 for h in hs if min(h["agg_weights"]) <= 1e-12)
    benign_rate.append({"seed": s, "rounds": len(hs), "zeroed_rounds": z,
                        "rate": round(z / len(hs), 4) if hs else None})
mean_rate = sum(d["rate"] for d in benign_rate if d["rate"] is not None) / max(1, len(benign_rate))
check("P1", "P", "đặc hiệu: không tấn công ⇒ gate hiếm khi zero client lương thiện",
      mean_rate <= 0.20,
      f"{R_CONVERGE} round × {S_DIST} seed — phụ thuộc phân hoạch nên báo TỈ LỆ",
      "Điều kiện tiên quyết để đọc được mọi con số 'attacker bị zero'. Gate "
      "zero ai đó mỗi round cũng cho w_atk = 0 — cùng con số, khác hẳn claim. "
      "Báo tỉ lệ chứ không nhị phân, vì A3 cố ý đẩy phân hoạch dị thể mạnh nên "
      "một client lương thiện CÓ THỂ chính đáng có φ<0 ở vài round.",
      mean_rate=round(mean_rate, 4), per_seed=benign_rate)

lat = []
for s, tr, res in SF:
    a = tr.byzantine_ids[0] if tr.byzantine_ids else None
    first, held = None, 0
    rounds_seen = 0
    for h in res["history"]:
        pr = h.get("phi_raw")
        if pr is None or a is None:
            continue
        v = [float(x) for x in pr]
        fi = [i for i, x in enumerate(v) if isnum(x)]
        if a not in fi:
            continue
        rounds_seen += 1
        if sorted(fi, key=lambda i: v[i]).index(a) == 0:
            held += 1
            if first is None:
                first = h.get("round")
    lat.append({"seed": s, "first_round_at_bottom": first,
                "frac_rounds_at_bottom": round(held / rounds_seen, 4) if rounds_seen else None})
early = [d for d in lat if d["first_round_at_bottom"] is not None
         and d["first_round_at_bottom"] <= 2]
check("P2", "P", "độ trễ phát hiện: φ đẩy attacker xuống đáy sớm VÀ giữ được",
      len(early) >= 4 and all((d["frac_rounds_at_bottom"] or 0) >= 0.8 for d in lat),
      f"{R_TEMPORAL} round × {S_DIST} seed — độ trễ phụ thuộc phân hoạch",
      "Learning point của cơ chế. F25 đã cho thấy phát hiện ĐÚNG mà MUỘN vẫn "
      "đủ để mô hình chết: gate bật ở round 6 trong khi gradient ×10 giết mô "
      "hình ở round 2. Nên phải đo CẢ độ trễ lẫn độ bền, không chỉ trạng thái cuối.",
      per_seed=lat, n_detected_by_round2=len(early))

print("\n--- P3/P4 ---", flush=True)
tier2 = {}
for atk in ("scaling", "alie"):
    row = {}
    for meth, lbl in (("fairshare", "tier1"), ("fairshare-robust", "tier2")):
        vals = []
        for s, tr, res in runs(meth, atk, R_TEMPORAL, S_DIR):
            ws = [h["agg_weights"] for h in res["history"] if h.get("agg_weights")]
            b = tr.byzantine_ids[0] if tr.byzantine_ids else 0
            if ws:
                vals.append(round(sum(w[b] for w in ws) / len(ws), 4))
        row[lbl] = vals
    row["tier2_lower_in_seeds"] = sum(
        1 for a, b in zip(row["tier1"], row["tier2"]) if b <= a)
    tier2[atk] = row
check("P3", "P", "FS-2 tầng 2 hạ trọng số attacker so với tầng 1 (scaling VÀ alie)",
      all(v["tier2_lower_in_seeds"] >= S_DIR - 1 for v in tier2.values()),
      f"{R_TEMPORAL} round × {S_DIR} seed × 2 attack",
      "F4 đã thu hẹp claim: ReLU-gate KHÔNG chặn scaling/ALIE (‖g_k‖ lớn đẩy "
      "điểm dương vượt gate) nên cần tầng 2. `alie` TRƯỚC NAY CHƯA TỪNG CHẠY "
      "(F26) ⇒ nửa phạm vi của claim hai tầng không có bằng chứng nào. Đây là "
      "kiểm HƯỚNG; độ lớn và ý nghĩa thống kê là Phase 3.",
      per_attack=tier2, fair_share=round(1.0 / CLIENTS, 4), n_seeds=S_DIR)

bf = []
for s, tr, res in runs("fedfairgnn-nodp", "fairness_poison", R_TEMPORAL, S_DIR):
    ws = [h["agg_weights"] for h in res["history"] if h.get("agg_weights")]
    b = tr.byzantine_ids[0] if tr.byzantine_ids else 0
    bf.append(round(sum(w[b] for w in ws) / len(ws), 4) if ws else None)
check("P4", "P", "đối chứng dương: BFWA BỊ fairness_poison bắt",
      sum(1 for x in bf if isnum(x) and x > 1.0 / CLIENTS) >= S_DIR - 1,
      f"{R_TEMPORAL} round × {S_DIR} seed",
      "FS-4 là negative result và là motivation của cả bài (P1/C3). Nó cũng là "
      "phép kiểm tra rằng THIẾT BỊ ĐO có khả năng báo đỏ: nếu không ô nào đỏ "
      "được thì các ô xanh không nói lên điều gì.",
      atk_w_mass_per_seed=bf, fair_share=round(1.0 / CLIENTS, 4))

# N6 cũ → quan sát ‖g_k‖ để hiệu chỉnh §4.0(b)
gmax = [float(h["g_norm_max"]) for _, _, r in (MAIN + SF)
        for h in r["history"] if isnum(h.get("g_norm_max"))]
thr = float(tr0.cfg.fu_grad_clip)
check("O1", "O", f"quan sát: fu_grad_clip={thr} có chạm ‖g_k‖ thực tế không",
      bool(gmax),
      "quan sát trên mọi run có sẵn",
      "`n_clipped = 0` MƠ HỒ: clip đúng cỡ trên round sạch cũng cho 0, ngưỡng "
      "sai đơn vị cũng cho 0. Checkpoint chỉ đòi ĐO ĐƯỢC; ACTIVE/INACTIVE là "
      "dữ liệu, và đổi hằng số là sửa ĐẶC TẢ §4.0(b) theo luồng 5 bước.",
      g_norm_max=round(max(gmax), 4) if gmax else None,
      g_norm_min=round(min(gmax), 4) if gmax else None, threshold=thr,
      status=("ACTIVE" if gmax and max(gmax) > thr else "INACTIVE — chưa từng chạm"))

# =============================================================================
report = {
    "run_id": RUN_ID, "commit": man["commit"], "commit_short": man["commit_short"],
    "branch": man.get("branch"), "tarball_sha256": man.get("tarball_sha256"),
    "provenance_source": man.get("provenance_source"),
    # Môi trường đi kèm MỌI artifact. Bài học 114×: hai môi trường không phân
    # biệt được bằng tên file là cách số của hai nơi bị bắc cầu thành một tỉ số
    # chưa từng tồn tại. arm64/macOS và x86/Linux là HAI môi trường.
    "device": "cpu", "env": ENV, "torch": torch.__version__,
    "design": {
        "beta": BETA, "tau_rounds": TAU,
        "rounds_temporal": R_TEMPORAL, "rounds_converge": R_CONVERGE,
        "seeds_distributional": S_DIST, "seeds_directional": S_DIR,
        "rationale": "Round của nhóm T suy từ β (3τ). Seed của nhóm P suy từ A3 "
                     "(phân hoạch Dirichlet dị thể ⇒ 1 seed = 0 bằng chứng). "
                     "Nhóm I chỉ cần 1 lần đánh giá vì Phase 0 đã cho bit-exact.",
    },
    "config": {"dataset": DATASET, "num_clients": CLIENTS,
               "fu_val_source": "server_holdout", "fu_holdout_size": HOLDOUT},
    "scope": "Chuỗi toán §1.3–§1.4 trên đường code ship. KHÔNG so hiệu năng, "
             "KHÔNG p-value, KHÔNG xếp hạng aggregator — đó là Phase 3.",
    "checks": CHECKS,
    "finished_at": time.strftime("%FT%TZ", time.gmtime()),
}
fails = [c for c in CHECKS if c["verdict"] == "FAIL"]
nas = [c for c in CHECKS if c["verdict"] == "N/A"]
report["gate1c_pass"] = not fails
json.dump(report, open(f"{OUT_DIR}/phase1_checkpoints.json", "w"),
          indent=2, ensure_ascii=False)

print("\n" + "=" * 78)
for g, name in (("E", "điều kiện tồn tại"), ("I", "đồng nhất thức"),
                ("T", "hợp đồng thời gian"), ("P", "phân bố"), ("O", "quan sát")):
    rows = [c for c in CHECKS if c["group"] == g]
    if not rows:
        continue
    print(f"\n[{g}] {name}")
    for c in rows:
        print(f"  {c['verdict']:>4}  {c['id']:<5} {c['what']}")
        print(f"        cần: {c['requires']}")
print("-" * 78)
if fails:
    print("ĐỎ:", ", ".join(c["id"] for c in fails))
if nas:
    print("KHÔNG KÍCH HOẠT (không phải PASS):", ", ".join(c["id"] for c in nas))
print("GATE 1-C:", "PASS" if report["gate1c_pass"] else "*** FAIL ***")
print(f"artifact: {OUT_DIR}/phase1_checkpoints.json")
print("""
⚠️ Đọc theo thứ tự phụ thuộc: E đỏ ⇒ DỪNG (mọi claim sau vô nghĩa). I đỏ ⇒ code
   lệch công thức. T đỏ ⇒ cơ chế thời gian sai. P đỏ ⇒ hành vi không như mô tả.
   GATE 1-C xanh KHÔNG nói cơ chế mạnh — mọi phát biểu về hiệu ứng cần Phase 3.
""")
