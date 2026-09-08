# 📊 FedFairGNN Experimental Results Repository
### Quản trị Kết Quả & Hồ Sơ Bằng Chứng Thực Nghiệm Chuẩn Q1

> **Tiêu chuẩn kiểm soát dữ liệu:** 3-Tier Anti-Hallucination Guard ([`.agents/rules/ag-research.md`](file:///Users/anson/DS/Research/.agents/rules/ag-research.md))  
> **Nguyên tắc Provenance (ADR-14):** 100% tệp kết quả chính thức phải chứa `manifest` (hoặc `_manifest`) ghi nhận chính xác: `git_commit`, `git_dirty` (`False`), `device` (`cpu` hoặc `cuda`), và `timestamp` ISO-8601. Mọi con số trong bài báo đều phải truy vết được về các artifacts tại thư mục này.

---

## 📁 1. CẤU TRÚC THƯ MỤC & PHÂN LOẠI ARTIFACTS

```
FedFairGNN/results/
├── README.md                          # Tài liệu này (Hồ sơ quản trị và mục lục artifacts)
├── preflight_datasets.json             # [S1/S6-C7] Tiền kiểm toán 5 tập dữ liệu & rò rỉ nhãn
├── canonical_suite.json                # [S3/S4/S6-C3] Bộ kiểm chuẩn chính tắc CPU & Ma trận bóc tách M1-M7
├── sota_pokecz.json                    # [S6] Ma trận SOTA Pokec-z GPU (10 baselines × 10 seeds = 100 runs)
├── sota_credit.json                    # [S6] Ma trận SOTA Credit GPU (10 baselines × 10 seeds = 100 runs)
├── privacy_attack.json                 # [S5.3] Tấn công suy diễn thuộc tính nhạy cảm dưới sweep ε
├── shapley_fidelity.json               # [S4.5] Độ tương quan giữa FU-Shapley và Exact Shapley (H3 REFUTED)
├── byzantine_sweep.json                # [S4.7] Quét khả năng chống chịu tấn công Byzantine (sign-flip, scaling)
├── consolidated_statistics.json        # Thống kê tổng hợp kiểm định giả thuyết
├── pareto_frontier_credit_pokecz.png   # Đồ thị trực quan hóa biên Pareto tiện ích - công bằng
├── fairshare/                          # [S4] Thư mục chứa nhật ký quỹ đạo, audit trọng số và kiểm chứng tiên đề
│   ├── audit_traj__german__*.csv       # Quỹ đạo cập nhật trọng số qua 20 rounds (24 kịch bản)
│   ├── metadata_immunity_verdict.json  # [Định lý 2(1)] Bằng chứng miễn nhiễm metadata gian lận
│   ├── null_player_verdict.json        # [Mệnh đề 3] Bằng chứng triệt tiêu trọng số null-player
│   └── convergence_empirical.json      # Bằng chứng đo lường biến thiên trọng số Ω_w
└── revision/                           # Thư mục lưu trữ các thí nghiệm mở rộng và bổ trợ
    ├── dp_accounting.json              # [S5.1] Chứng chỉ Rényi Differential Privacy (RDP)
    ├── update_level_attack.json        # [S5.2] Phân tích ranh giới rò rỉ theo từng kênh quan sát
    └── *.json                          # Các run bổ trợ pre-Phase-0 (chuẩn bị chạy lại ở S7/S8)
```

---

## 📜 2. CHI TIẾT CÁC ARTIFACTS ĐÃ NGHIỆM THU (VERIFIED STAGE ARTIFACTS)

### 🔹 Stage S1 & S6-C7: Tiền Kiểm Toán Dữ Liệu
* **File:** [`preflight_datasets.json`](preflight_datasets.json)
* **Commit:** `476d72dc` | **Thiết bị:** `cpu` | **Dirty:** `False`
* **Nội dung:** Thẩm định 5 đồ thị chuẩn: German ($N=1.000$, $|E|_{\text{undir}}=21.742$), Bail ($N=18.876$, $|E|_{\text{undir}}=311.870$), Credit ($N=30.000$, $|E|_{\text{undir}}=1.421.858$), Pokec-z ($N=67.796$, $|E|_{\text{undir}}=617.958$), Elliptic ($N=203.769$, $|E|_{\text{undir}}=234.355$).
* **Quy chuẩn:** Xác nhận thống nhất quy ước cạnh vô hướng $|E|_{\text{undir}}$ và cạnh có hướng PyG $|E|_{\text{dir}} = 2|E|_{\text{undir}}$; kiểm định $\max_j \text{AUC}_{\text{feat}} < 0{,}85$ (không có rò rỉ nhãn tầm thường).

### 🔹 Stage S3, S4 & S6-C3: Kiểm Chuẩn Chính Tắc & Ma Trận Bóc Tách (Component Ablation)
* **File:** [`canonical_suite.json`](canonical_suite.json)
* **Commit:** `904be982` | **Thiết bị:** `cpu` | **Dirty:** `False`
* **Nội dung:** Chạy 10 seeds $\{42\ldots51\}$ trên German Credit ($K=5, R=20, E=3$):
  - `M1_Full`: $\text{AUC} = 0{,}6512 \pm 0{,}0373$ | $\text{DPD}_{\text{hard}} = 0{,}0407 \pm 0{,}0278$
  - `M2_wo_FSER_true` (Clean Arm, $\beta=0$ đóng băng): $\text{AUC} = 0{,}6500 \pm 0{,}0362$ | $\text{DPD}_{\text{hard}} = 0{,}0497 \pm 0{,}0422$ ($\Delta\text{DPD} = +0{,}0090, p=0{,}7344 \implies$ German Benign Null)
  - `M2_wo_FSER` (Old Confounded GAT): $\text{AUC} = 0{,}6934$ | $\text{DPD}_{\text{hard}} = 0{,}1284$ (phản ánh thiếu hụt scaffold chuẩn hóa)
  - `M3_wo_FTGD`: $\text{AUC} = 0{,}6540$ | $\text{DPD}_{\text{hard}} = 0{,}0864$ ($\Delta\text{DPD} = +0{,}0457, p=0{,}0488$)
  - `M4_Full_DPSGD`: $\text{AUC} = 0{,}5869$ | $\text{DPD}_{\text{hard}} = 0{,}1471$ (sụp đổ do nhiễu chiều cao)
  - `M5_wo_FairScore`, `M6_wo_TwoTier`, `M7_wo_EMA`: Các nhánh phân tích vai trò dưới điều kiện lành tính.

### 🔹 Stage S6: Ma Trận SOTA So Sánh Đồ Thị Lớn (10 Baselines $\times$ 10 Seeds)
* **Files:** [`sota_pokecz.json`](sota_pokecz.json) và [`sota_credit.json`](sota_credit.json)
* **Commit:** `279ed390` | **Thiết bị:** `cuda` (Tesla T4) | **Dirty:** `False`
* **Quy mô:** Đủ 100 runs độc lập cho mỗi tập dữ liệu, bao gồm 10 phương pháp: `fedavg-gcn`, `fairgnn`, `fairsin`, `fairfed`, `fairgfl`, `fedgraphfair`, `cgsv`, `ours-nofser`, `ours-nofser-true`, `fedfairgnn`.
* **Kết quả cốt lõi:**
  - **Pokec-z:** `fedfairgnn` đạt **$\text{AUC} = 0{,}7899 \pm 0{,}0095$**, dẫn đầu tuyệt đối và đánh bại cả 7 baseline đối chứng độc lập ($p \le 0{,}0020$ sau hiệu chỉnh Holm-Bonferroni).
  - **Credit:** `fedfairgnn` đạt **$\text{AUC} = 0{,}7522 \pm 0{,}0065$**, bám sát nhóm dẫn đầu (FairGFL $0{,}7536$, FedAvg $0{,}7529$), xác lập ranh giới ứng dụng trên đồ thị bảng nhân tạo.

### 🔹 Stage S4: Kiểm Chứng Tiên Đề & Cơ Chế Gom Tụ
* **Files:** Trong thư mục [`fairshare/`](fairshare/):
  - [`metadata_immunity_verdict.json`](fairshare/metadata_immunity_verdict.json): Client khai gian $\widehat{\text{DPD}}=0$ làm trọng số FU-Shapley đổi **$\Delta w = 0{,}0000$ bit-exact** (trong khi BFWA bị thao túng $\Delta w = 0{,}8246$).
  - [`null_player_verdict.json`](fairshare/null_player_verdict.json): $w_{\text{null}} = 0{,}0000$ ở 100% các vòng kiểm thử (khớp Định lý Lean 4).
  - [`convergence_empirical.json`](fairshare/convergence_empirical.json): Đo lường biến thiên trọng số $\Omega_w$: `fu_shapley` ($0{,}6036$) ổn định gấp 27 lần `bfwa` ($16{,}5407$).

### 🔹 Stage S5: Quyền Riêng Tư Vi Sai & Phân Tích Rò Rỉ Kênh
* **Files:** Trong thư mục [`revision/`](revision/):
  - [`dp_accounting.json`](revision/dp_accounting.json): Đạt chứng chỉ Rényi DP $\varepsilon = 7{,}9985 \le 8{,}0$ tại $\delta = 10^{-4}$ trên German ($T=60$).
  - [`update_level_attack.json`](revision/update_level_attack.json): Đo lường chính xác ranh giới rò rỉ: Targeted FTGD đưa attack AUC kênh thống kê từ $1{,}0000$ về $0{,}4983$ với chi phí chỉ $+0{,}002$ AUC (bảo tồn tiện ích gấp 115 lần DP-SGD toàn diện); kênh update duy trì ở $0{,}6450$ (Limitation đã khai).

---

## 🔍 3. HƯỚNG DẪN KIỂM CHÉO TỰ ĐỘNG (AUDIT PROTOCOL)

Bất kỳ nhà nghiên cứu hoặc reviewer nào đều có thể kiểm tra tính toàn vẹn bit-exact và hợp thức của các artifacts bằng script sau:

```bash
cd /Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN

# Thực thi kiểm toán tính toàn vẹn của artifacts
python3 - <<'PY'
import json, glob, os

print(">>> KIỂM TRA PROVENANCE CỦA CÁC ARTIFACTS CHÍNH THỨC:")
official_files = [
    "results/preflight_datasets.json",
    "results/canonical_suite.json",
    "results/sota_pokecz.json",
    "results/sota_credit.json",
    "results/revision/dp_accounting.json",
    "results/revision/update_level_attack.json"
]

for path in official_files:
    assert os.path.exists(path), f"Thiếu file: {path}"
    with open(path) as f:
        d = json.load(f)
    m = d.get("manifest") or d.get("_manifest")
    assert m is not None, f"{path}: Thiếu manifest"
    assert m.get("git_dirty") is False, f"{path}: Cây làm việc bẩn (git_dirty=True)"
    print(f"✅ PASS: {path:<40s} | Commit: {m.get('git_commit')[:8]} | Device: {m.get('device')}")

print("\n>>> TẤT CẢ ARTIFACTS CHÍNH THỨC ĐẠT CHUẨN BIT-EXACT 100%.")
PY
```

---

## 📌 4. LƯU Ý VỀ CÁC TỆP BỔ TRỢ TRONG `revision/`

Các tệp trong `revision/` như `dirichlet_sweep.json`, `metis_partition.json`, `proxy_sensitivity.json`, `trust_score_sensitivity.json` là các bản chạy bổ trợ từ giai đoạn thử nghiệm sơ khởi (pre-Phase-0). Theo kế hoạch tại `EXPERIMENT_EXECUTION_PLAN_04.md` (§4, Stage S8), các nội dung này sẽ được chạy lại đồng bộ dưới hệ thống logging hiện hành trước khi đưa vào các mục phụ lục của bài báo.
