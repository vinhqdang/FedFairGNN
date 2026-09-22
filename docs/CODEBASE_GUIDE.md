# TrustFedGNN Codebase Architecture & Engineering Guide

> **Phiên bản:** Revision Q1 (Chuẩn hóa cấu trúc 5 phần `manuscript_v2`)  
> **Tài liệu tham chiếu:** [`../../docs/03_ast_and_codebase_mapping.md`](../../docs/03_ast_and_codebase_mapping.md) · [`../../docs/02_mathematical_formulation_and_formal_proofs.md`](../../docs/02_mathematical_formulation_and_formal_proofs.md)

Tài liệu này cung cấp cái nhìn tổng quan ở cấp độ kỹ thuật về codebase `FedFairGNN`, trách nhiệm của các module, các cấu trúc dữ liệu chính, giao thức kết nối và hướng dẫn kiểm thử tự động.

---

## 1. Cấu Trúc Thư Mục (Directory Layout)

```
FedFairGNN/
├── src/                          # Mã nguồn cốt lõi
│   ├── config.py                 # Dataclass ExperimentConfig & Thiết lập hạt giống tất định
│   ├── models/                   # Kiến trúc mạng nơ-ron đồ thị (GNN)
│   │   ├── gnn.py                # Lớp FSERLayer / TrustFedGNN (GAT 2 lớp + BN + Skip) & baselines
│   │   └── baselines.py          # Kiến trúc chuyên biệt của FairGNN và FairSIN
│   ├── federated/                # Giao thức huấn luyện & thuật toán tổng hợp liên đoàn
│   │   ├── client.py             # Vòng đời Client, bước chiếu trực giao FTGD, soft-DPD, weighted BCE
│   │   ├── trainer.py            # FederatedTrainer điều phối vòng lặp cross-silo
│   │   ├── aggregation.py        # FU-Shapley, FLAME, FLTrust, BFWA, Krum, Median, Trimmed Mean
│   │   ├── attacks.py            # Đòn tấn công (Gaussian, Sign-flip, Scaling, Stealth, Fairness-poison)
│   │   └── server.py             # API máy chủ điều phối trung tâm
│   ├── trust/                    # Các module quản trị và độ tin cậy
│   │   ├── privacy.py            # Kế toán RDP PrivacyAccountant & Tính toán nhiễu Gauss
│   │   ├── trust_score.py        # Tính toán Composite Trust Index (Độ tin cậy 5 chiều)
│   │   ├── uncertainty.py        # MC-Dropout epistemic uncertainty & Hiệu chuẩn ECE
│   │   ├── compliance.py         # Kiểm tra tuân thủ kỹ thuật EU AI Act & NIST AI RMF
│   │   └── explain.py            # Giải thích trọng số chú ý GNN
│   ├── data/                     # Tải & phân vùng dữ liệu đồ thị
│   │   ├── datasets.py           # 5 benchmarks (German, Credit, Bail, Pokec-z, Elliptic)
│   │   ├── partition.py          # Phân vùng Non-IID Dirichlet & Cụm cộng đồng Louvain/Metis
│   │   └── sampler.py            # SimpleNeighborLoader cho suy luận đồ thị mini-batch
│   └── utils/                    # Tiện ích dùng chung
│       ├── metrics.py            # AUC-ROC, AP, F1-macro, DPD_hard, EOD, độ rung lắc trọng số Ω_w
│       └── logging_utils.py      # Ghi log JSONL và serialize artifacts chuẩn
├── experiments/                  # Kịch bản thực thi & điều phối thực nghiệm
│   ├── run_experiment.py         # Điểm vào thực thi một thực nghiệm đơn lẻ
│   ├── methods.py                # Đăng ký 16 baselines SOTA & biến thể Ours
│   ├── make_manuscript_v2_figures.py # Sinh biểu đồ tự động cho bản thảo manuscript_v2
│   ├── make_tables_c2.py         # Sinh bảng biểu tự động từ kết quả JSON
│   └── revision/                 # Các kịch bản chuyên biệt phục vụ chiến dịch phản biện
├── results/                      # Hồ sơ dữ liệu gốc JSON có chữ ký manifest
│   └── revision/                 # 40+ artifacts kiểm định phản biện (RUN-CTRL, RUN-DELTA-GRID...)
├── manuscript_v2/                # Mã nguồn LaTeX bài báo bản thảo v2 (68 trang, 0 warnings)
│   ├── sections/                 # 00_preamble, 01_intro, 02_related, 03_method, 04_results, 05_conclusion
│   ├── tables/                   # 14 bảng biểu (10 bảng chính + 4 bảng phụ lục/kiểm định)
│   └── figures/                  # 5 biểu đồ vector chính thức (PDF/PNG)
├── scripts/                      # Công cụ kiểm toán (lint_manuscript_blacklist.py)
├── tests/                        # Bộ kiểm thử pytest khóa các bất biến toán học và thuật toán
└── docs/                         # Tài liệu kỹ thuật nội bộ codebase
```

---

## 2. Các Module Cốt Lõi (Core Implementation)

### A. Phía Client: Huấn luyện Cục bộ & Phép Chiếu Trực Giao FTGD (`src/federated/client.py`)
- **`Client._ftgd_step(model, optimizer, batch, config)`**:
  1. Tính toán gradient nhiệm vụ $g_{\text{task}}$ và gradient công bằng $g_{\text{fair}} = \nabla_\theta (\lambda \mathcal{L}_{\text{fair}})$.
  2. Thực hiện phép chiếu trực giao loại bỏ xung đột mục tiêu (Định lý 1):
     $$g_{\text{task}}^\perp = g_{\text{total}} - \frac{\langle g_{\text{total}}, g_{\text{fair}}\rangle}{\|g_{\text{fair}}\|^2 + \varepsilon} g_{\text{fair}}$$
  3. Đo lường thống kê chênh lệch nhóm 2 chiều $(\mu_0, \mu_1)$. Khi bật chế độ DP, tiêm nhiễu Gauss $\mathcal{N}(0, \sigma_{\text{DP}}^2)$ với độ nhạy giới hạn $\Delta \le \sqrt{2}/n_{\min}$.
  4. Phát hành thống kê đã bảo vệ $\widetilde{\text{DPD}}_k = |\tilde{\mu}_0 - \tilde{\mu}_1|$ lên máy chủ.

### B. Phía Server: Cơ Chế Tổng Hợp Ba Tầng & Phòng Thủ Đa Lớp (`src/federated/aggregation.py`)
Thuật toán `fu_shapley` / `robust_fu_shapley` được thiết kế theo kiến trúc module phân rã:

1. **Tầng 1 — Cổng Phi Tuyến FU-Gating (Tier-1 Directional Filter):**
   - Đo lường góc cosine giữa gradient client $\theta_k$ và gradient mỏ neo $\gtarg = g_{\mathrm{task}} + \alpha g_{\mathrm{fair}}$ tính trên tập kiểm chuẩn sạch $\Droot$:
     $$\text{sim}_k = \cos(\theta_k, \gtarg) = \frac{\langle \theta_k, \gtarg \rangle}{\|\theta_k\| \cdot \|\gtarg\| + \varepsilon}$$
   - Lọc bỏ các cập nhật đi ngược hướng mục tiêu bằng hàm chỉnh lưu $\text{ReLU}(\text{sim}_k)$.
2. **Tầng 2 — Khống Chế Biên Độ (Tier-2 Norm Rescaling):**
   - Chuẩn hóa chặn biên độ gradient của client theo chuẩn vector mỏ neo sạch:
     $$\theta_k \leftarrow \theta_k \cdot \min\left(1, \frac{\|\gtarg\|}{\|\theta_k\| + \varepsilon}\right)$$
   - Ngăn chặn hoàn toàn các đòn tấn công phóng đại trọng số (Scaling $c=100$).
3. **Tầng 3 — Làm Mượt Động Lực Học (Tier-3 Reference EMA):**
   - Cập nhật vector mỏ neo qua Exponential Moving Average $\gtarg^{(t)} = \beta \gtarg^{(t-1)} + (1-\beta) g_{\Droot}^{(t)}$ giúp giảm rung lắc trọng số $\Omega_w$ từ $24.1\times$ (German Credit) tới $30\times$ (Pokec-z, Bảng 9).
4. **Cơ Chế Dự Phòng Thích Ứng (Static Defense-in-Depth Variant):**
   - Biến thể `robust_fu_shapley` cung cấp thêm bộ lọc trung vị tọa độ (Coordinate Median screening) được cấu hình tĩnh cho môi trường đe dọa cao (High-threat deployments), trong khi cấu hình sản xuất mặc định là **Canonical FU-Alignment (Gating + Norm Rescaling + EMA)**.

### C. Bộ Thuật Toán Tổng Hợp Đối Chuẩn (Benchmark Aggregators)
Trong `src/federated/aggregation.py`, codebase hỗ trợ đầy đủ các bộ gom tụ:
- **`fedavg`**: Trung bình có trọng số cổ điển (McMahan et al., 2017).
- **`fltrust`**: Gom tụ mỏ neo tin cậy đơn mục tiêu (Cao et al., NDSS 2021).
- **`flame`**: Gom tụ phân cụm khoảng cách cosine kết hợp dynamic norm clipping (Nguyen et al., USENIX Security 2022).
- **`krum` / `multikrum`**: Lọc khoảng cách Euclid loại bỏ ngoại lai.
- **`trimmed_mean` / `median`**: Lọc thống kê tọa độ không mỏ neo.
- **`fairfed` / `f2gnn` / `fedgraphfair` / `popets_fairfed`**: Các thuật toán công bằng phụ thuộc siêu dữ liệu tự khai của client.

---

## 3. Quy Trình Kiểm Thử & Đảm Bảo Chất Lượng (QA & Testing)

Codebase duy trì bộ kiểm thử tự động gồm 53 bài test khóa cứng toàn bộ các bất biến thuật toán:

```bash
# 1. Kiểm tra từ khóa cấm trong bản thảo (Zero Overclaim Guard)
python3 scripts/lint_manuscript_blacklist.py

# 2. Chạy toàn bộ kiểm thử đơn vị & kiểm tra bất biến
pytest tests/ -v

# 3. Kiểm tra riêng biệt thuật toán FLAME mới bổ sung
pytest tests/test_flame_aggregator.py -v
```

Mọi kết quả kiểm thử và số liệu đối chiếu chi tiết xem tại [`EXPERIMENTS_AND_RESULTS.md`](EXPERIMENTS_AND_RESULTS.md) và [`../../docs/05_data_and_results.md`](../../docs/05_data_and_results.md).
