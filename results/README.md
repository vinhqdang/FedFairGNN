# 📊 FedFairGNN Experimental Results Repository & Data Provenance Guide

> **Mục tiêu:** Quản trị hồ sơ bằng chứng thực nghiệm chuẩn tạp chí Q1 (*Neurocomputing*).  
> **Sổ dữ liệu độc quyền hạt nhân:** [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md)  
> **Kế hoạch & Nhật ký thực thi:** [`../../docs/04_experiment_execution.md`](../../docs/04_experiment_execution.md)  
> **Tiêu chuẩn kiểm soát dữ liệu:** 3-Tier Anti-Hallucination Guard ([`.agents/rules/ag-research.md`](../../.agents/rules/ag-research.md)).  
> **Nguyên tắc Provenance (ADR-14):** 100% tệp kết quả chính thức chứa `manifest` (hoặc `_manifest`) ghi nhận chính xác: `git_commit`, `git_dirty` (`False`), `device` (`cpu` hoặc `cuda`), và `timestamp` ISO-8601. Mọi con số trong bài báo tại `manuscript_v2/` đều được truy vết tự động về các artifacts tại thư mục này, tuyệt đối không có số liệu gõ tay.

---

## 📁 1. CẤU TRÚC THƯ MỤC & MỤC LỤC ARTIFACTS

```
FedFairGNN/results/
├── README.md                          # Tài liệu này (Hồ sơ quản trị và mục lục artifacts)
├── preflight_datasets.json            # [Table 1] Tiền kiểm toán 5 tập dữ liệu & rò rỉ nhãn
├── canonical_suite.json               # [Table 4, 9, 11] Bộ kiểm chuẩn chính tắc CPU & Ma trận bóc tách M1-M7 & Two-Tier
├── sota_pokecz.json                   # [Table 7, 8] Ma trận SOTA Pokec-z GPU (10 baselines × 10 seeds = 100 runs)
├── sota_credit.json                   # [Table 7] Ma trận SOTA Credit GPU (10 baselines × 10 seeds = 100 runs)
├── byzantine_sweep.json               # [Figure 3] Quét Byzantine scaling đa mức f/K ∈ {0.1, 0.2, 0.3} × 5 seeds
├── convergence_bail.json              # Quá trình hội tụ qua 20 rounds với n=10 seeds trên Bail Recidivism
├── consolidated_statistics.json       # Thống kê tổng hợp kiểm định giả thuyết Wilcoxon & Holm-Bonferroni
├── pareto_frontier_credit_pokecz.png  # [Figure 5] Đồ thị biên Pareto tiện ích - công bằng
│
├── fairshare/                         # Thư mục kiểm chứng các tiên đề toán học & Shapley fidelity
│   ├── metadata_immunity_verdict.json # [Định lý 2] Bằng chứng miễn nhiễm metadata gian lận (Δw = 0.0000 bit-exact)
│   ├── null_player_verdict.json       # [Mệnh đề 3] Bằng chứng triệt tiêu trọng số null-player (w_null = 0)
│   ├── convergence_empirical.json     # [Table 8] Bằng chứng đo lường chi phí tính toán & wall-clock
│   └── exact_sv_corr*                 # [Table 14] Dữ liệu kiểm định độ tương quan với Exact Shapley
│
└── revision/                          # 25+ artifacts kiểm định phản biện chuyên sâu đã nghiệm thu 100%
    ├── aggregator_control_pokecz.json # [RUN-CTRL / Table 7] Đối chứng cùng backbone GAT (A0 FedAvg Scaffold Control, n=10)
    ├── fltrust_delta_grid_results.json# [RUN-DELTA-GRID / Table 10] Lưới nhân tử 2x2 (600 runs, n=30), chứng minh Mechanical Separability
    ├── fltrust_delta_grid_german_s4.json # Dữ liệu chi tiết lưới nhân tử German Credit seed set 4
    ├── metadata_capture_stats.json    # [RUN-META / Table 2] Thống kê Sign test độc lập trên 6 quy tắc SOTA (n=30)
    ├── metadata_capture_endtoend.json # Dữ liệu đầy đủ 144 runs kiểm toán kênh metadata tự khai
    ├── metadata_capture_bail.json     # Dữ liệu mở rộng kiểm toán metadata trên Bail Recidivism (900 runs)
    ├── adaptive_poisoner_results.json # [RUN-STEALTH / Table 5] Tấn công ngụy trang thích ứng (Omniscient Stealth Adversary)
    ├── adaptive_poisoner_results_breakdown_summary.json # Tóm tắt phân rã tấn công thích ứng
    ├── flame_adaptive_results.json    # [RUN-FLAME / Table 5] Đối chuẩn với thuật toán gom cụm khoảng cách cosine FLAME (USENIX'22)
    ├── flame_adaptive_results_breakdown_summary.json # Tóm tắt kết quả phân rã FLAME
    ├── test_flame_smoke_breakdown_summary.json # Smoke test kiểm tra tính toàn vẹn của module FLAME
    ├── alignment_adversary.json       # [Table 6] Tấn công hộp trắng toàn tri Kerckhoffs T1 tối ưu Adam (n=10)
    ├── rescale_median_ablation.json   # [Table 4] Bóc tách tương tác giữa Norm Rescaling và Coordinate Median
    ├── pokecz_adversarial.json        # [Table 7] Kiểm tra đối kháng quy mô lớn trên Pokec-z dưới scaling c=100
    ├── bfwa_slack.json                # [RUN-SLACK / Table 3] 1,000 mẫu Monte Carlo kiểm định độ chùng ràng buộc LDP
    ├── dp_accounting.json             # [Table 3] Bảng tính Rényi Differential Privacy (RDP) lý thuyết & thực nghiệm
    ├── metis_partition.json           # [RUN-PART / Table 12] So sánh phân vùng cộng đồng Metis đo ứng suất tô-pô đồ thị
    ├── dirichlet_sweep.json           # [RUN-DIR / Table 12] 48 runs quét Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0} đo độ lệch Holdout
    ├── noninferiority_test.json       # Kiểm định TOST tương đương sinh học hai phía (Two One-Sided Tests)
    ├── centralized_sanity.json        # Neo kiểm chứng tập trung vs phân tán (Δ_FL)
    ├── proxy_sensitivity.json         # Độ nhạy cảm nhóm đại diện (Topological vs Demographic)
    ├── trust_score_sensitivity.json   # [Table 13] 2,000 mẫu Monte Carlo nhiễu trọng số Composite Trust Score
    └── update_level_attack.json       # Ranh giới rò rỉ kênh thống kê (0.498) vs kênh update tham số (0.645)
```

---

## 🔗 2. BẢN ĐỒ ÁNH XẠ 1-1: ARTIFACT $\leftrightarrow$ BẢNG BÀI BÁO `manuscript_v2/`

Mọi bảng biểu và biểu đồ trong bản thảo [`manuscript_v2/`](../manuscript_v2/) được liên kết trực tiếp với các tệp dữ liệu nguồn:

| Bảng trong `manuscript_v2` | Tệp Bằng Chứng Nguồn (Artifact JSON) | Nội Dung Khoa Học & Phán Quyết Đối Chiếu |
|---|---|---|
| **Table 1** (`tab_datasets.tex`) | `preflight_datasets.json` | **Đặc Tính 5 Bộ Dữ Liệu Benchmark:** Kiểm toán homophily nhạy cảm $h_s$, zero label leakage, và phân hoạch holdout $D_{\mathrm{root}}$ ($\le 125$ nodes). |
| **Table 2** (`tab_metadata_capture.tex`) | `revision/metadata_capture_stats.json`<br>`revision/metadata_capture_endtoend.json`<br>`revision/metadata_capture_bail.json` | **Khảo Sát Lỗ Hổng Siêu Dữ Liệu Tự Khai:** 6/6 quy tắc gom tụ công bằng bị thao túng; Sign-test độc lập từng phương pháp ($F^2\text{GNN}$ 30/30, FairFed 28/28, BFWA 18/19); 12 seeds phân kỳ của $q$-FedAvg được ghi nhận minh bạch là Attack Success. Bảo chứng cú pháp $\Delta w = 0.0000$ (Theorem 2). |
| **Table 3** (`tab_ldp_barrier.tex`) | `revision/bfwa_slack.json`<br>`revision/dp_accounting.json` | **Rào Cản Bất Khả Của LDP:** Phép gập $| \cdot |$ biến nhiễu Gauss thành thiên lệch dương (Folded Normal Bổ đề 1); độ chùng thực tế dãn tới $+10{,}891\%$ ở $\varepsilon=0.5$; Cận Le Cam Minimax chứng minh sai số kiểm định tiến dần về $50\%$ (Theorem 4). |
| **Table 4** (`tab_two_tier.tex`) | `canonical_suite.json`<br>`revision/rescale_median_ablation.json` | **Cơ Chế Phòng Thủ Hai Tầng:** Norm Rescaling khống chế scaling thô bạo ($c=100$); Median screening backfire dưới stealth ($w_{\mathrm{adv}}: 0.2254 \to 0.2561, +13.6\%$); xác lập cấu hình sản xuất **Canonical FU-Alignment (Gating + Rescaling + EMA)** là mặc định. |
| **Table 5** (`tab_adaptive_poisoner.tex`) | `revision/adaptive_poisoner_results.json`<br>`revision/flame_adaptive_results.json` | **Đấu Trường Đối Kháng Thích Ứng & Nghịch Lý Median:** Dưới ngụy trang cự ly thích ứng trên Bail ($f/K \in [0.1, 0.4]$), Median screening phản tác dụng làm tăng $+82.9\%$ $w_{\mathrm{adv}}$; đối chuẩn với FLAME (USENIX'22) xác nhận ưu thế bảo vệ công bằng của mỏ neo tham chiếu. |
| **Table 6** (`tab_alignment_adversary.tex`)| `revision/alignment_adversary.json` | **Tấn Công Hộp Trắng Toàn Tri Kerckhoffs T1:** Kẻ địch tối ưu Adam chiếm đoạt $w_{\mathrm{adv}} = 0.8514$ (German) và $0.7461$ (Bail), xác lập ranh giới an ninh tự nhiên của cơ chế tham chiếu khi holdout bị lộ. |
| **Table 7** (`tab_sota_main.tex`) | `sota_pokecz.json`<br>`sota_credit.json`<br>`revision/aggregator_control_pokecz.json` | **SOTA Benchmark & Bóc tách Scaffold:** Pokec-z ($n=10$) & Credit ($n=10$). Margin $+0.0594$ là ưu thế cấp hệ thống; trên cùng backbone GAT (`RUN-CTRL`), TrustFedGNN tương đương lành tính với FedAvg ($\Delta = +0.0015, p=0.625$, TOST $\delta=0.0100$). |
| **Table 8** (`tab_cost.tex`) | `fairshare/convergence_empirical.json`<br>`sota_pokecz.json` | **Chi Phí Tính Toán & Quản Trị Vận Hành:** Đo lường trên GPU T4; phụ trội thời gian huấn luyện $2.06\times$ tại server; client chỉ thêm 8 bytes/round; 5.60 GFLOPs. |
| **Table 9** (`tab_weight_stability.tex`) | `canonical_suite.json`<br>`revision/aggregator_control_pokecz.json` | **Hòa Giải Đa Chế Độ Biến Thiên Trọng Số $\Omega_w$:** Trên Pokec-z EMA dập rung lắc gấp $30\times$ ($0.6845 \to 0.0227$), trên German dập $24.1\times$ ($1.4134 \to 0.0586$), trên BFWA dập $27.4\times$ ($16.5407 \to 0.6033$). |
| **Table 10** (`tab_factorial_2x2.tex`) | `revision/fltrust_delta_grid_results.json`<br>`revision/fltrust_delta_grid.json` | **Lưới Nhân Tử $2 \times 2$ (Alpha Grid):** $n=30$ seeds, 600 runs. FU-Gating dập tắt kẻ địch ($p < 10^{-5}$); hiệu ứng cột của $\alpha$ trơ dưới $\delta=0.0050$, xác nhận tính **Mechanical Separability (Không mất thuế kháng lỗi)**. |
| **Table 11** (`tab_ablation_suite.tex`) | `canonical_suite.json` | **Bộ Bóc Tách Thành Phần M1–M7:** German Credit ($n=10$). Làm rõ M2 sạch (FSER trơ trên đồ thị thuần, $p=0.541$); M7 xác nhận vai trò sống còn của EMA. |
| **Table 12** (`tab_topology_stress.tex`) | `revision/metis_partition.json`<br>`revision/dirichlet_sweep.json` | **Ứng Suất Dị Thể Tô-pô & Khảo Sát Holdout Skew:** Metis giảm $27\%$ cosine alignment; Dirichlet sweep 12 ô thừa nhận FedAvg dẫn trước utility khi $\alpha_{\mathrm{Dir}} \le 0.1$ ($p=0.00049$), phân định ranh giới vận hành minh bạch. |
| **Table 13** (`tab_trust_score_sensitivity.tex`)| `revision/trust_score_sensitivity.json` | **Độ Vững Chắc Điểm Tin Cậy Tổng Hợp:** 2,000 mẫu Monte Carlo xác nhận thứ hạng ổn định $\rho_s = 0.965$, Rank-1 duy trì $100\%$. |
| **Table 14** (`tab_trust_fidelity.tex`) | `results/fairshare/` artifacts | **Giới Hạn Xấp Xỉ Shapley Tổ Hợp:** 125 điểm thăm dò ($K=5, n=5$ seeds). $r = 0.7662 < 0.80$; định vị chính xác là heuristic phân rã tuyến tính bậc 1 $O(KP)$, không tuyên bố tương đương game theory. |

---

## 🔍 3. HƯỚNG DẪN KIỂM CHÉO TỰ ĐỘNG (PROVENANCE AUDIT SCRIPT)

Bất kỳ reviewer hoặc cộng tác viên nào đều có thể kiểm tra tính toàn vẹn bit-exact và hợp lệ của toàn bộ hồ sơ dữ liệu bằng lệnh:

```bash
cd /Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN

python3 - <<'PY'
import json, glob

files = sorted(glob.glob("results/*.json") + glob.glob("results/revision/*.json"))
print(f"Tổng số artifacts được quét: {len(files)}")
valid_count = 0

for f in files:
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
        manifest = d.get('manifest') or d.get('_manifest') or {}
        commit = (manifest.get('git_commit') or 'N/A')[:7]
        device = manifest.get('device') or 'unknown'
        dirty = manifest.get('git_dirty')
        print(f"✅ {f:<55} | Commit: {commit} | Device: {device:<5} | Dirty: {dirty}")
        valid_count += 1
    except Exception as e:
        print(f"❌ {f:<55} | LỖI: {e}")

print(f"\nKết quả kiểm toán: {valid_count}/{len(files)} artifacts hợp lệ bit-exact.")
PY
```

---

## 🛡️ 4. NGUYÊN TẮC BẢO VỆ DỮ LIỆU BẤT KHẢ XÂM PHẠM

1. **Tuyệt đối không sửa tay trong file JSON:** Mọi file kết quả đều mang mã hash cấu trúc; nếu cần sửa thuật toán, phải chạy lại runner tương ứng để sinh artifact mới kèm manifest mới.
2. **Không trích dẫn các file đã bị lưu trữ (`experiments/legacy/`):** Chỉ các artifacts được liệt kê ở Mục 1 và 2 mới được coi là căn cứ khoa học chính thức.
3. **Đồng bộ hóa 100% với Sổ Dữ Liệu Hạt Nhân:** Khi phát hiện bất kỳ sự bất đồng nào giữa tài liệu và code, sổ dữ liệu [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md) là **thẩm quyền tối cao**.
