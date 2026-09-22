# 📚 FedFairGNN Documentation Hub & Master Index

> **Vị trí trong dự án:** Thư mục tài liệu kỹ thuật nội bộ của codebase `FedFairGNN`.  
> **Tài liệu nguồn gốc cấp cao:** Hệ thống tài liệu quản trị nghiên cứu hạt nhân tại [`../../docs/`](../../docs/).  
> **Nguyên tắc quản trị:** Đồng nhất 100% giữa tài liệu kỹ thuật của repo và hồ sơ khoa học của bài báo (3-Tier Anti-Hallucination Guard).

---

## 🗺️ 1. BẢN ĐỒ LIÊN KẾT ĐỒNG BỘ VỚI THƯ MỤC GỐC `TrustFedGNN/docs/`

Mọi đặc tả toán học, phân loại y văn, phân rã AST và sổ dữ liệu kết quả chính thức được lưu trữ tập trung tại `1_Paper/01.GNN/TrustFedGNN/docs/`. Bảng dưới đây ánh xạ vai trò của từng tài liệu hạt nhân:

| Tài Liệu Hạt Nhân (`TrustFedGNN/docs/`) | Nội Dung Khoa Học & Kỹ Thuật | Vai Trò Đối Với Codebase `FedFairGNN` |
|---|---|---|
| [`01_sota_taxonomy_and_gap_analysis.md`](../../docs/01_sota_taxonomy_and_gap_analysis.md) | Phân loại 16 baselines SOTA, 4 trường phái & 3 khoảng trống tri thức lớn | Chuẩn hóa danh mục thuật toán trong `src/federated/aggregation.py` và `experiments/methods.py`. |
| [`02_mathematical_formulation_and_formal_proofs.md`](../../docs/02_mathematical_formulation_and_formal_proofs.md) | Khung toán học hoàn chỉnh: 5 Định lý, Bổ đề Folded Normal, 4 chứng chỉ Lean 4 | Neo toán học cho phép chiếu FTGD (`client.py`) và phân rã Shapley (`aggregation.py`). |
| [`03_ast_and_codebase_mapping.md`](../../docs/03_ast_and_codebase_mapping.md) | Bản đồ AST chi tiết, cây gọi hàm, cấu trúc dữ liệu và điểm vào thực thi | Hướng dẫn định vị mã nguồn và truy vết luồng dữ liệu tensor giữa Server và Client. |
| [`04_1_novelty_advantages.md`](../../docs/04_1_novelty_advantages.md) | 12 luận điểm bảo vệ tính mới & ưu thế cạnh tranh trước AC-reviewers | Định vị giá trị thực chất của các module phòng thủ và bóc tách scaffold. |
| [`04_experiment_execution.md`](../../docs/04_experiment_execution.md) | Kế hoạch thực thi thực nghiệm, nhật ký chạy GPU/Colab và cổng nghiệm thu | Kịch bản điều khiển chuỗi thực nghiệm trong `experiments/` và quản lý phiên tính toán. |
| [`05_data_and_results.md`](../../docs/05_data_and_results.md) | **Sổ dữ liệu độc quyền**: 100% con số bit-exact trích xuất từ `results/` JSON | Nguồn chân lý số liệu (Source of Truth) cho toàn bộ bảng biểu tại `manuscript_v2/tables/`. |
| [`CHANGELOG.md`](../../docs/CHANGELOG.md) | Toàn văn nhật ký kỹ thuật từ S1 đến S10, mã commit và timestamps | Lịch sử tiến hóa của codebase, các bản vá bảo mật và kiểm định mô hình. |
| [`review/ac_review_p06_implementation_plan.md`](../../docs/review/ac_review_p06_implementation_plan.md) | Ma trận đề xuất điều chỉnh & kế hoạch thực hiện phản biện 3 AC-Reviewers | Khung hành động ưu tiên (P0-P3) trực tiếp giải quyết các yêu cầu sửa đổi của AC. |
| [`review/KE_HOACH_VIET_LAI_MANUSCRIPT.md`](../../docs/review/KE_HOACH_VIET_LAI_MANUSCRIPT.md) | Quy chuẩn viết lách ngược (Bottom-Up), 5 tiền đề sống còn & 5 quy tắc vàng | Định hướng chấp bút bản thảo `manuscript_v2/`, bảng từ cấm và quy tắc "Nhận và Tiến". |

---

## 📂 2. CẤU TRÚC TÀI LIỆU NỘI BỘ TRONG `FedFairGNN/docs/`

```
FedFairGNN/docs/
├── README.md                      # Tài liệu này (Master Navigation & Sync Hub)
├── CODEBASE_GUIDE.md              # Cẩm nang kiến trúc mã nguồn & kỹ thuật chi tiết
├── EXPERIMENTS_AND_RESULTS.md     # Hướng dẫn tái lập thực nghiệm & ánh xạ 14 bảng biểu
├── BASELINES_AND_SOURCES.md       # Bảng độ trung thực tái lập 16 baselines & nguồn bài báo
├── COLAB_WORKFLOW.md              # Quy trình vận hành & đồng bộ tính toán đám mây Google Colab
└── proofs/                        # Mã nguồn kiểm chứng hình thức các đẳng thức đại số bằng Lean 4
    ├── SimplexProperties.lean     # Tính chất hình học của Simplex (Bảo toàn trọng số)
    ├── NullPlayer.lean            # Bằng chứng triệt tiêu trọng số null-player
    ├── OrthogonalProjection.lean  # Tính trực giao của phép chiếu FTGD
    └── LinearDecomposition.lean   # Phân rã tuyến tính đóng góp bậc 1 O(KP)
```

---

## ⚡ 3. NGUYÊN TẮC BẤT BIẾN KHI CẬP NHẬT CODEBASE & TÀI LIỆU

1. **Nguyên tắc Provenance Dữ liệu:**  
   Tuyệt đối không gõ tay bất kỳ con số nào vào tài liệu hoặc bài báo. Mọi kết quả phải được trích xuất tự động từ các file JSON trong `FedFairGNN/results/` hoặc script `experiments/make_*.py`.
2. **Khóa Mỏ Neo Thực Chứng (Section 4 Anchor):**  
   Mọi cập nhật trong tài liệu phải phản ánh đúng hiện trạng thực nghiệm đã khóa:
   - Margin $+0.0594$ là ưu thế cấp hệ thống (system-level advantage); trên cùng backbone GAT, isolated aggregator duy trì tương đương lành tính $\Delta\mathrm{AUC} = +0.0015$ (`aggregator_control_pokecz.json`).
   - Cổng phi tuyến FU-Gating là lá chắn lọc độc hại chính; $\alpha g_{\text{fair}}$ là guardrail dự phòng dài hạn đạt tính tách biệt cơ chế (**Mechanical Separability**, TOST $\delta = 0.0050$, `fltrust_delta_grid_results.json`).
   - Cấu hình mặc định sản xuất là **Canonical FU-Alignment (Gating + Norm Rescaling + EMA)**; Coordinate Median là cơ chế phòng thủ chuyên sâu chỉ kích hoạt khi cấu hình tĩnh cho môi trường rủi ro cao.
3. **Phân Định Minh Bạch Công Trình Tiền Nhiệm:**  
   Mọi tài liệu phải phân định rõ ràng các thành phần kế thừa từ công trình hội nghị trước của nhóm (`dang2026fedfairgnn`, PMLR v319, pp. 74–86, IndabaX 2026: FSER, FTGD, BFWA) so với các đóng góp mới 100% của bản thảo hiện tại (FU-Alignment, Le Cam Minimax, 4 chứng chỉ Lean 4, 14 bảng thực nghiệm).
4. **Đồng Bộ Hóa Đích Đến:**  
   Mọi liên kết bản thảo LaTeX hiện tại đều trỏ trực tiếp vào `manuscript_v2/` (bản thảo 68 trang, 14 bảng, 5 biểu đồ, 0 cảnh báo), toàn bộ tài liệu cũ trong `manuscript_neurocomputing/` đã được lưu trữ vào thư mục lưu trữ (`_archive/`).
