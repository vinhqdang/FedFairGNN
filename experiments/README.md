# `experiments/` — Bộ Điều Khiển Thực Nghiệm TrustFedGNN

> **File điều khiển DUY NHẤT** của toàn bộ chuỗi `colab → runner → results → manuscript_v2`.  
> **Kế hoạch thực thi & Trạng thái stage:** [`../../docs/04_experiment_execution.md`](../../docs/04_experiment_execution.md).  
> **Sổ dữ liệu độc quyền & Kết quả bit-exact:** [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md).  
> **Nhật ký thay đổi & Lịch sử mã commit:** [`../../docs/CHANGELOG.md`](../../docs/CHANGELOG.md).  
> **Bản thảo đầu ra trực tiếp:** [`../manuscript_v2/`](../manuscript_v2/).

---

## 1. Cấu Hình Chuẩn Tắc (Canonical Configuration) — Nguồn Chân Lý Duy Nhất

Từ ngày 07-09-2026, toàn bộ dự án thống nhất dùng **duy nhất giao thức chuẩn tắc**:

```python
# src/config.py :: ExperimentConfig.canonical()   ◄── NGUỒN CHÂN LÝ DUY NHẤT
dataset="german", num_clients=5, rounds=20, dirichlet_alpha=0.3,
model="trustfedgnn", aggregator="fu_shapley",
fu_alpha=0.1, fu_ema_beta=0.9, fu_val_source="server_holdout",
fairness_weight=1.0, beta_init=0.5, fser_mode="sub",
dp_enabled=True, dp_mode="ftgd"
# Mặc định khác: local_epochs=3, dp_epsilon=8.0, dp_delta=1e-5, sampling=False
```

Các runner SOTA override cấu hình thành: `num_clients=10, rounds=50`, tập hạt giống `{42…51}`.  
**Mọi ablation bắt buộc phải khởi tạo từ `canonical()` rồi override tường minh** — tuyệt đối không dựng config thủ công.

---

## 2. Bố Cục Thư Mục `experiments/`

```
experiments/
├── README.md                          # Tài liệu này (Quy trình thực nghiệm & điều khiển)
├── colab/                             # Kịch bản điều phối máy ảo Google Colab GPU từ xa
│   ├── 00_pack.sh                     # [LOCAL] Đóng gói codebase + manifest
│   ├── 01_setup.py                    # [VM] Giải nén, symlink, pytest — GATE 0
│   ├── 11_smoke_test.py               # [VM] Kiểm thử luồng khép kín
│   ├── 15_canonical_suite.py          # [VM] Thực thi canonical suite
│   └── run_local.sh                   # [LOCAL] Chạy CPU cục bộ
│
├── run_smoke_test.py                  # Kiểm thử tính toàn vẹn đường ống huấn luyện (CPU)
├── run_canonical_suite.py             # Bộ kiểm chuẩn Canonical + Ablation M1–M7 + Two-Tier (CPU)
├── run_sota_pokecz.py                 # Ma trận SOTA Pokec-z & Đối chứng A0 Scaffold Control (GPU)
├── run_sota_credit.py                 # Ma trận SOTA Credit Default (GPU)
├── run_byzantine_sweep.py             # Quét tỷ lệ tấn công Byzantine đa mức f/K (GPU)
├── run_shapley_fidelity.py            # Kiểm định tương quan FU-Shapley vs Exact Shapley (CPU)
├── run_scalability_ogbn.py            # Thử nghiệm khả năng mở rộng ogbn-products 2.4M nút (CPU)
├── run_pareto_sweep.py                # Quét lưới biên Pareto đa mục tiêu (CPU)
│
├── incentive_audit.py                 # Kiểm toán lỗ hổng kênh metadata tự khai 6/6 SOTA
├── exact_shapley_correlation.py       # Tính toán đóng góp Shapley chính xác (K=4, 16 liên minh)
│
├── revision/                          # Các runner chuyên biệt cho chiến dịch phản biện AC
│   ├── adaptive_poisoner.py           # Tấn công ngụy trang cự ly thích ứng & FLAME baseline
│   ├── ablation_grid_runner.py        # Lưới nhân tử 2x2 (600 runs, n=30) khảo sát alpha
│   ├── bfwa_slack_analysis.py         # Monte Carlo kiểm định độ chùng ràng buộc LDP
│   ├── dirichlet_sweep.py             # Quét 12 ô Dirichlet non-IID khảo sát holdout skew
│   ├── metis_partition_experiment.py  # Phân vùng cộng đồng Metis đo ứng suất tô-pô
│   ├── dp_accounting_table.py         # Bảng kế toán Rényi DP giải tích
│   ├── update_level_attack.py         # Đòn tấn công suy diễn thuộc tính trên kênh update
│   └── trust_score_sensitivity.py     # Kiểm định độ vững của Composite Trust Score
│
├── methods.py                         # Đăng ký 16 baselines SOTA + FLAME + CGSV + biến thể Ours
├── make_manuscript_v2_figures.py      # Sinh toàn bộ 5 biểu đồ vector vào manuscript_v2/figures/
├── make_tables_c2.py                  # Sinh các bảng chứng cứ toán học vào manuscript_v2/tables/
├── make_stats.py                      # Tổng hợp thống kê Wilcoxon, Cohen d, Holm-Bonferroni
└── legacy/                            # ⛔ ĐÃ LƯU TRỮ — Không chạy, không trích dẫn
```

---

## 3. Ma Trận Chiến Dịch Thực Nghiệm & Ánh Xạ 14 Bảng `manuscript_v2`

Toàn bộ 14 bảng số liệu trong bản thảo `manuscript_v2/` được liên kết 1-1 với các kịch bản thực thi và hồ sơ dữ liệu nguồn:

| Mã Chiến Dịch | Mục Tiêu & Cơ Chế Kiểm Định | Script Thực Thi | Thiết Bị Bắt Buộc | Tệp Kết Quả Đầu Ra (`results/`) | Bảng trong `manuscript_v2` |
|---|---|---|:--:|---|---|
| **PREFLIGHT** | Tiền kiểm toán 5 tập benchmark, kiểm tra homophily nhạy cảm $h_s$ và rò rỉ nhãn | `experiments/revision/preflight_handoff.py` | **CPU** | `results/preflight_datasets.json` | Table 1 (`tab_datasets.tex`) |
| **RUN-META** | Khảo sát lỗ hổng siêu dữ liệu tự khai trên 6 quy tắc SOTA ($n=30$ seeds) | `experiments/incentive_audit.py` | **CPU / GPU** | `results/revision/metadata_capture_stats.json` | Table 2 (`tab_metadata_capture.tex`) |
| **RUN-SLACK** | 1,000 mẫu Monte Carlo kiểm định rào cản Folded Normal và Le Cam Minimax của LDP | `experiments/revision/bfwa_slack_analysis.py` | **CPU** | `results/revision/bfwa_slack.json`<br>`results/revision/dp_accounting.json` | Table 3 (`tab_ldp_barrier.tex`) |
| **RUN-RESCALE** | Bóc tách cơ chế phòng thủ 2 tầng: Norm Rescaling vs Coordinate Median | `experiments/run_canonical_suite.py` | **CPU** | `results/canonical_suite.json`<br>`results/revision/rescale_median_ablation.json` | Table 4 (`tab_two_tier.tex`) |
| **RUN-STEALTH / FLAME** | Tấn công ngụy trang cự ly thích ứng ($f/K \in [0.1, 0.4]$) & Đối chuẩn FLAME (USENIX'22) | `experiments/revision/adaptive_poisoner.py` | **GPU T4** | `results/revision/adaptive_poisoner_results.json`<br>`results/revision/flame_adaptive_results.json` | Table 5 (`tab_adaptive_poisoner.tex`) |
| **RUN-ALIGN** | Tấn công hộp trắng toàn tri Kerckhoffs T1 tối ưu Adam ($n=10$ seeds) | `experiments/revision/alignment_adversary.py` | **GPU T4** | `results/revision/alignment_adversary.json` | Table 6 (`tab_alignment_adversary.tex`) |
| **RUN-CTRL / SOTA** | Bóc tách Scaffold: Đối chứng `fedavg-gat` vs TrustFedGNN trên cùng backbone ($n=10$) | `experiments/run_sota_pokecz.py`<br>`experiments/run_sota_credit.py` | **GPU T4** | `results/sota_pokecz.json`<br>`results/sota_credit.json`<br>`results/revision/aggregator_control_pokecz.json` | Table 7 (`tab_sota_main.tex`) |
| **RUN-COST** | Đo lường thời gian huấn luyện wall-clock, GFLOPs và phụ trội tính toán tại server | `experiments/revision/convergence_audit.py` | **GPU T4** | `results/fairshare/convergence_empirical.json` | Table 8 (`tab_cost.tex`) |
| **RUN-STABILITY** | Hòa giải đa chế độ biến thiên trọng số $\Omega_w$ (EMA dập rung lắc tới $30\times$) | `experiments/run_canonical_suite.py` | **CPU** | `results/canonical_suite.json`<br>`results/revision/aggregator_control_pokecz.json` | Table 9 (`tab_weight_stability.tex`) |
| **RUN-DELTA-GRID** | Khảo sát lưới nhân tử $2 \times 2$ ($n=30$ seeds, 600 runs), chứng minh Mechanical Separability | `experiments/revision/ablation_grid_runner.py` | **GPU T4 / CPU** | `results/revision/fltrust_delta_grid_results.json` | Table 10 (`tab_factorial_2x2.tex`) |
| **RUN-ABLATION** | Bộ bóc tách thành phần M1–M7 trên German Credit ($n=10$ seeds) | `experiments/run_canonical_suite.py` | **CPU** | `results/canonical_suite.json` | Table 11 (`tab_ablation_suite.tex`) |
| **RUN-PART / DIR** | Quét 12 ô Dirichlet Skew & Phân vùng cộng đồng Metis trên Bail | `experiments/revision/dirichlet_sweep.py`<br>`experiments/revision/metis_partition_experiment.py` | **GPU T4 / CPU** | `results/revision/dirichlet_sweep.json`<br>`results/revision/metis_partition.json` | Table 12 (`tab_topology_stress.tex`) |
| **RUN-SENS** | 2,000 mẫu Monte Carlo kiểm định độ vững thứ hạng điểm tin cậy tổng hợp ($\rho_s = 0.965$) | `experiments/revision/trust_score_sensitivity.py` | **CPU** | `results/revision/trust_score_sensitivity.json` | Table 13 (`tab_trust_score_sensitivity.tex`) |
| **RUN-FIDELITY** | Kiểm định độ phân kỳ tiên đề và giới hạn xấp xỉ Shapley tổ hợp (125 điểm thăm dò) | `experiments/run_shapley_fidelity.py` | **CPU** | `results/fairshare/` artifacts | Table 14 (`tab_trust_fidelity.tex`) |

---

## 4. Quy Trình Xuất Bản Bảng Biểu & Biên Dịch Bản Thảo `manuscript_v2/`

Mọi con số đưa vào bản thảo `manuscript_v2` phải tuân thủ nghiêm ngặt pipeline tự động (**Zero Manual Typing**):

```bash
cd /Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN

# 1. Kiểm tra tính hợp lệ của toàn bộ artifacts JSON
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("results/*.json") + glob.glob("results/revision/*.json")):
    try:
        d = json.load(open(f))
        m = d.get('manifest') or d.get('_manifest') or {}
        print("OK: %-50s | commit=%s" % (f, (m.get('git_commit') or 'none')[:7]))
    except Exception as e:
        print("FAIL: %s (%s)" % (f, e))
PY

# 2. Sinh các bảng biểu vào manuscript_v2/tables/ (14 bảng biểu)
python3 experiments/make_tables_c2.py

# 3. Sinh các biểu đồ vector vào manuscript_v2/figures/ (5 biểu đồ vector)
python3 experiments/make_manuscript_v2_figures.py

# 4. Quét từ khóa cấm & kiểm toán tuyên bố học thuật (Zero Overclaim Guard)
python3 scripts/lint_manuscript_blacklist.py

# 5. Biên dịch PDF bản thảo hoàn chỉnh với chu trình chuẩn tắc (68 trang, 0 warnings)
cd manuscript_v2
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## 5. Nguyên Tắc Bất Di Bất Dịch Trong Thực Nghiệm

1. **Một Nguồn Chân Lý Duy Nhất (Single Source of Truth):** Số liệu chỉ được ghi vào artifact JSON; tài liệu và bản thảo chỉ đọc trực tiếp, tuyệt đối không gõ tay.
2. **Khai Báo Thiết Bị Minh Bạch Trong Manifest:** Kết quả CPU (tất định bit-exact) và GPU (CUDA non-deterministic) không bao giờ được trộn lẫn trong cùng một bảng đối chuẩn.
3. **Phân Định Minh Bạch Công Trình Tiền Nhiệm:** Mọi tài liệu và script phải làm rõ các thành phần kế thừa từ bài báo hội nghị trước (`dang2026fedfairgnn`, PMLR v319: FSER, FTGD, BFWA) so với các đóng góp mới độc quyền của tạp chí (FU-Alignment, Le Cam Minimax, 4 chứng chỉ Lean 4, 14 bảng thực nghiệm).
4. **Chuẩn Mực Học Thuật & Kiểm Soát Tuyên Bố:** Các công cụ chứng minh hình thức (Lean 4) được trình bày như chứng chỉ kiểm toán tính chất đại số/hình học, không dùng làm highlight thực nghiệm hay tuyên bố phóng đại.
5. **Tuân Thủ Bộ Quy Chuẩn Viết Lách:** Mọi phân tích kết quả thực nghiệm phải bám sát [`../../docs/review/KE_HOACH_VIET_LAI_MANUSCRIPT.md`](../../docs/review/KE_HOACH_VIET_LAI_MANUSCRIPT.md).

