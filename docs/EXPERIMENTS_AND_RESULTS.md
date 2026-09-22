# Hướng Dẫn Tái Lập Thực Nghiệm & Sổ Tra Cứu Kết Quả (Reproduction Guide)

> **Tài liệu điều phối hạt nhân:** [`../../docs/04_experiment_execution.md`](../../docs/04_experiment_execution.md)  
> **Sổ dữ liệu độc quyền:** [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md)  
> **Mục tiêu:** Cung cấp hướng dẫn từng bước để tái lập 100% các kết quả thực nghiệm trong bản thảo `manuscript_v2/` với độ chính xác bit-exact trên CPU và kiểm định phân phối trên GPU.

---

## 1. Lệnh Tái Lập Nhanh (Quick Reproduction Commands)

### A. Chạy Một Thực Nghiệm Đơn Lẻ (Single Experiment)
```bash
# Chạy TrustFedGNN (Canonical) trên German Credit với seed 42
python3 -m experiments.run_experiment --method fedfairgnn --dataset german --seed 42

# Chạy Baseline FLAME (USENIX Security 2022) trên Bail Recidivism
python3 -m experiments.run_experiment --method flame --dataset bail --seed 42

# Chạy Đối chứng Cùng Backbone GAT (A0 FedAvg Scaffold Control)
python3 -m experiments.run_experiment --method fedavg-gat --dataset pokec_z --seed 42
```

### B. Chạy Bộ Kiểm Chuẩn Chính Tắc (Canonical Suite)
```bash
# Chạy bộ kiểm chuẩn CPU: Canonical + Ablation M1–M7 + Two-Tier Defense
python3 experiments/run_canonical_suite.py
```

---

## 2. Danh Mục Các Chiến Dịch Thực Nghiệm Phản Biện (`experiments/revision/`)

Toàn bộ các yêu cầu kiểm định chuyên sâu từ quá trình phản biện AC-reviewers được tổ chức thành các chiến dịch thực nghiệm độc lập:

| Mã Chiến Dịch | Mục Tiêu Khoa Học | Script Thực Thi | Artifact Đầu Ra (`results/revision/`) | Bảng trong `manuscript_v2` |
|---|---|---|---|---|
| **PREFLIGHT** | Tiền kiểm toán 5 tập dữ liệu benchmark, tỷ lệ đồng chất nhạy cảm $h_s$ | `experiments/revision/preflight_handoff.py` | `results/preflight_datasets.json` | Table 1 (`tab_datasets.tex`) |
| **RUN-META** | Khảo sát lỗ hổng kênh metadata tự khai trên 6 quy tắc SOTA ($n=30$ seeds) | `experiments/incentive_audit.py` | `metadata_capture_stats.json` | Table 2 (`tab_metadata_capture.tex`) |
| **RUN-SLACK** | Kiểm chứng thực nghiệm rào cản LDP (Folded Normal bias & Le Cam minimax) | `experiments/revision/bfwa_slack_analysis.py` | `bfwa_slack.json`, `dp_accounting.json` | Table 3 (`tab_ldp_barrier.tex`) |
| **RUN-RESCALE** | Bóc tách cơ chế phòng thủ 2 tầng: Norm Rescaling vs Coordinate Median | `experiments/run_canonical_suite.py` | `rescale_median_ablation.json` | Table 4 (`tab_two_tier.tex`) |
| **RUN-STEALTH** | Tấn công ngụy trang cự ly thích ứng & Phản tác dụng của Median ($f/K \in [0.1, 0.4]$) | `experiments/revision/adaptive_poisoner.py` | `adaptive_poisoner_results.json`, `flame_adaptive_results.json` | Table 5 (`tab_adaptive_poisoner.tex`) |
| **RUN-ALIGN** | Tấn công hộp trắng toàn tri Kerckhoffs T1 tối ưu Adam ($n=10$ seeds) | `experiments/revision/alignment_adversary.py` | `alignment_adversary.json` | Table 6 (`tab_alignment_adversary.tex`) |
| **RUN-CTRL** | Bóc tách Scaffold: Đối chứng FedAvg vs TrustFedGNN trên cùng backbone GAT ($n=10$) | `experiments/run_sota_pokecz.py` | `sota_pokecz.json`, `aggregator_control_pokecz.json` | Table 7 (`tab_sota_main.tex`) |
| **RUN-COST** | Đo lường thời gian huấn luyện wall-clock, GFLOPs và phụ trội tính toán tại server | `experiments/revision/convergence_empirical.py` | `convergence_empirical.json` | Table 8 (`tab_cost.tex`) |
| **RUN-STABILITY** | Hòa giải đa chế độ biến thiên trọng số $\Omega_w$ (EMA dập rung lắc tới $30\times$) | `experiments/run_canonical_suite.py` | `canonical_suite.json`, `aggregator_control_pokecz.json` | Table 9 (`tab_weight_stability.tex`) |
| **RUN-DELTA-GRID** | Khảo sát lưới nhân tử $2 \times 2$ ($\alpha \in \{0, 0.1\}$), chứng minh Mechanical Separability | `experiments/revision/ablation_grid_runner.py` | `fltrust_delta_grid_results.json` | Table 10 (`tab_factorial_2x2.tex`) |
| **RUN-ABLATION** | Bộ bóc tách thành phần M1–M7 trên German Credit ($n=10$ seeds) | `experiments/run_canonical_suite.py` | `canonical_suite.json` | Table 11 (`tab_ablation_suite.tex`) |
| **RUN-DIR/PART** | Quét 12 ô Dirichlet Skew ($\alpha \in [0.1, 1.0]$) & Phân vùng cộng đồng Metis trên Bail | `experiments/revision/dirichlet_sweep.py` | `dirichlet_sweep.json`, `metis_partition.json` | Table 12 (`tab_topology_stress.tex`) |
| **RUN-SENS** | 2,000 mẫu Monte Carlo kiểm tra độ vững thứ hạng điểm tin cậy tổng hợp ($\rho_s = 0.965$) | `experiments/revision/trust_score_sensitivity.py` | `trust_score_sensitivity.json` | Table 13 (`tab_trust_score_sensitivity.tex`) |
| **RUN-FIDELITY** | Kiểm định độ phân kỳ tiên đề và giới hạn xấp xỉ Shapley tổ hợp (125 điểm thăm dò) | `experiments/run_shapley_fidelity.py` | `results/fairshare/` artifacts | Table 14 (`tab_trust_fidelity.tex`) |

---

## 3. Quy Trình Tự Động Sinh Bảng Biểu & Hình Vẽ Bản Thảo

Dự án áp dụng nguyên tắc **Zero Manual Typing**: toàn bộ bảng biểu và biểu đồ được sinh tự động từ các tệp JSON chính thức:

```bash
# 1. Sinh các bảng biểu chứng cứ toán học & phòng thủ vào manuscript_v2/tables/
python3 experiments/make_tables_c2.py

# 2. Sinh các biểu đồ vector (PDF/PNG) vào manuscript_v2/figures/
python3 experiments/make_manuscript_v2_figures.py

# 3. Biên dịch kiểm thử toàn bộ bản thảo LaTeX
cd manuscript_v2 && pdflatex -interaction=nonstopmode main.tex
```

Mọi dữ liệu chi tiết, các giá trị p-value và khoảng tin cậy 95% được ghi nhận đầy đủ tại [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md).
