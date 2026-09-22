# Baseline Reimplementation Fidelity & Academic Sources

> **Tài liệu tham chiếu hạt nhân:** [`../../docs/01_sota_taxonomy_and_gap_analysis.md`](../../docs/01_sota_taxonomy_and_gap_analysis.md)

Tài liệu này ghi nhận đầy đủ nguồn gốc, mức độ trung thực khi tái lập (reimplementation fidelity) và các giả định xấp xỉ của 16 baselines SOTA được tích hợp trong codebase `FedFairGNN`.

---

## 1. Các Thuật Toán Tái Lập Trung Thực Tuyệt Đối (Exact / Faithful Core)

| Phương Pháp | Hội Nghị / Tạp Chí | Cơ Chế Tái Lập Trong Codebase | Mức Độ Trung Thực |
|---|---|---|---|
| **FedAvg-GCN / GAT** | AISTATS 2017 | Khung kiến trúc GCN/GAT tiêu chuẩn kết hợp trung bình có trọng số theo kích thước mẫu | **Exact** |
| **FLTrust** | NDSS 2021 | Gom tụ định hướng mỏ neo sạch $g_{\text{target}}$ từ server holdout $\Droot$ + Norm Rescaling | **Exact** |
| **FLAME** | USENIX Security 2022 | Ma trận khoảng cách cosine pairwise + Phân cụm liên kết trung bình + Cắt tỉa chuẩn động vị | **Exact** |
| **FairGNN** | WSDM 2021 | Khử thiên vị đối kháng (Minimax giữa GNN encoder và bộ phân biệt nhạy cảm) | **Faithful core** (Không dùng sensitive-estimator do thuộc tính $S$ được quan sát đầy đủ) |
| **FairSIN** | AAAI 2024 | FairSIN-F: Tăng cường đặc trưng láng giềng dị thể kết hợp bộ ước lượng MLP | **Faithful** (Phiên bản FairSIN-F tiêu chuẩn) |
| **FairFed** | AAAI 2023 | Cập nhật trọng số tổng hợp dựa trên độ lệch công bằng cục bộ của client | **Exact** (Công thức trọng số hàm mũ nguyên bản) |
| **q-FedAvg** | ICLR 2020 | Tái cân bằng trọng số theo tổn thất client nâng lũy thừa $q$ | **Exact** |
| **F$^2$GNN** | IEEE ICDM 2023 | Trọng số công bằng hàm softmax kết hợp cân bằng nhóm (pp. 980--985) | **Faithful** |
| **BFWA** | IndabaX 2026 | Tối ưu đối ngẫu Frank-Wolfe Lagrange dựa trên scalar telemetry tự khai (PMLR v319) | **Exact** (Công trình hội nghị tiền nhiệm của nhóm tác giả, đưa vào kiểm toán đối kháng trên cùng hệ quy chiếu) |
| **DP-FedAvg** | ICLR 2018 | Cắt tỉa gradient toàn phần + Nhiễu Gauss (Kế toán RDP) | **Exact** (Đóng vai trò đối chứng bảo mật cho FTGD) |
| **FaVGNN** | Info. Fusion 2026 | Biến thể ngang hóa (horizontal adaptation) của hợp nhất đặc trưng dị thể và đối kháng | **Faithful adaptation** |
| **FDP-Fair** | arXiv 2026 | DP-SGD + Dịch chuyển phân vị nhóm Demographic Parity ở bước hậu xử lý | **Exact** |
| **Krum / Multi-Krum** | NeurIPS 2017 | Lựa chọn gradient có tổng khoảng cách Euclid nhỏ nhất tới láng giềng | **Exact** |
| **Coordinate Median** | ICML 2018 | Lấy trung vị độc lập trên từng tọa độ tham số | **Exact** |
| **Trimmed Mean** | ICML 2018 | Cắt bỏ $\beta$-phân vị cao nhất và thấp nhất trước khi lấy trung bình | **Exact** |

---

## 2. Các Phương Pháp Tái Lập Có Điều Chỉnh Phạm Vi (Scoped / Partial Adaptation)

Các phương pháp dưới đây được thiết kế cho các bối cảnh đặc thù (như đồ thị nhiều thành phần tách rời hoặc mật mã FHE); khi chuyển vào bối cảnh đồ thị phân tán non-IID của bài báo, phần lõi thuật toán được giữ nguyên với các điều chỉnh minh bạch sau:

| Phương Pháp | Xuất Bản | Cơ Chế Gốc | Phần Được Giữ Lại Trong Codebase | Phần Được Điều Chỉnh / Bỏ Qua |
|---|---|---|---|---|
| **FairGFL** | IEEE TPDS 2026 | Trọng số $w_i \propto 1/(1+O_i)$ với $O_i$ là tỷ lệ chồng lấn cạnh đồ thị giữa các client | Trọng số nghịch đảo độ lệch mẫu chuẩn hóa (aggregator `fairgfl`) | Bài báo gốc giả định nhiều đồ thị tách rời có chồng lấn cạnh; trong bối cảnh đồ thị đơn Dirichlet phân vùng của bài báo, tỷ lệ này được xấp xỉ bằng độ lệch mất cân bằng dữ liệu |
| **FedGraph-Fair** | Info. Sci. 2026 | Mô hình cá nhân hóa qua đồ thị tương đồng client + Nhân tử Lagrange DRO $\lambda$ | Phần lõi DRO: Chiếu simplex nhân tử $\lambda$ cho các client có loss vượt trần (aggregator `fedgraphfair`) | Lớp cá nhân hóa và trộn đồ thị tương đồng phi tập trung được bỏ qua để giữ một mô hình toàn cục chung |
| **PUFFLE** | ECAI 2024 | DP-SGD + Bộ điều khiển phản hồi xung lượng tự điều chỉnh $\lambda \in [0, 1]$ | Bộ điều khiển tự động điều chỉnh $\lambda$ cục bộ dựa trên khoảng cách disparity (hàm `_puffle_step`) | Bỏ qua kênh chia sẻ số lượng nhóm nhạy cảm thứ ba vì mọi client trong thực nghiệm đều quan sát đủ cả hai nhóm |
| **FedFACT** | NeurIPS 2025 | Tối ưu hóa ràng buộc công bằng Bayes toàn cục + cục bộ | Công thức nghiệm giải tích dạng đóng cho bài toán Demographic Parity nhị phân | Không triển khai thuật toán dual-ascent đa lớp tổng quát do bài toán thực nghiệm là nhị phân |
| **PoPETs** | PoPETs 2025 | FairFed đa thức bậc 2 chạy trên giao thức mật mã ngưỡng CKKS kết hợp LDP | Phần lõi thống kê: Trọng số FairFed đa thức bậc 2 (aggregator `popets_fairfed`) | Bỏ qua hạ tầng mật mã đồng cấu CKKS (vốn chỉ đóng vai trò bảo mật truyền thông, không làm thay đổi giá trị số học sau giải mã) |

---

## 3. Phân Loại 5 Trường Phái SOTA Đối Chuẩn (Theo Section 2 `manuscript_v2`)

Toàn bộ 16 baselines trên được phân bổ chuẩn xác theo 5 trường phái nghiên cứu:

1. **Paradigm I — Huấn luyện Cục bộ Khử Thiên Vị In-Processing (Client-Side Debiasing):** FairGNN, FairSIN, FairGB, FairInv. *(Tử huyệt: Hoàn toàn mù tại biên gom tụ máy chủ; gradient đối kháng vô hiệu hóa nỗ lực cục bộ).*
2. **Paradigm II — Điều phối Trọng số Dựa vào Siêu Dữ Liệu Tự Khai (Client-Reported Metric Coordination):** FairFed, FairGFL, FedGraph-Fair, $q$-FedAvg, BFWA, PUFFLE, FedFACT. *(Tử huyệt: Thất bại trước đòn tấn công khai man metadata; kẻ địch chiếm tới 97.4% quyền kiểm soát mô hình — Theorem 2 & Table 2).*
3. **Paradigm III — Vi Phân Riêng Tư & Rào Cản Lý Thuyết Thông Tin (DP & Information Barriers):** DP-FedAvg, FDP-Fair, FedFDP. *(Tử huyệt: Bổ đề Folded Normal gây thiên lệch dương vĩnh viễn; Cận dưới Le Cam Minimax chứng minh DP triệt tiêu tính quan sát được của ràng buộc).*
4. **Paradigm IV — Bộ Lọc Hình Học Không Mỏ Neo (Geometric Byzantine Filtering):** Coordinate Median, Trimmed Mean, Krum, Multi-Krum, FLAME. *(Tử huyệt: Mù không gian con công bằng — Theorem 5; gây nghịch lý Median Backfire làm tăng tới +82.9% trọng số kẻ địch trên đồ thị non-IID).*
5. **Paradigm V — Gom Tụ Mỏ Neo Tham Chiếu & Sổ Cái Đóng Góp (Server-Anchored Alignment & Governance):** FLTrust (đơn mục tiêu), CGSV (mỏ neo nội sinh dễ tổn thương), GuardFed (sàng lọc ngưỡng), và **TrustFedGNN (Đề xuất)**. *(Đột phá: Mỏ neo ngoại sinh $\Droot$, miễn nhiễm cú pháp đại số $\Delta w \equiv 0$, chuẩn hóa norm, và sổ cái đóng góp tuyến tính $O(KP)$).*
