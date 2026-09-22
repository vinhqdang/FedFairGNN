# `results/archived/` — artifact đã rút khỏi lưu thông

⛔ **KHÔNG trích dẫn bất kỳ file nào trong thư mục này** vào bản thảo hay `docs/`.

Chuyển vào đây ngày **14-09-2026** sau kiểm toán phủ hai chiều
(`docs/CHANGELOG.md` mục `[14-09-2026c]`). Tiêu chí: **không script nào đọc,
không tài liệu nào dẫn** — tức không đỡ một tuyên bố nào trong bản thảo hiện tại.

| File | Nguồn gốc | Vì sao rút |
|---|---|---|
| `fairshare/ablation_holdout_size__german__s0__manifest.json` | `experiments/ablation_holdout_size.py`, giai đoạn FairShare | Thay bởi `revision/holdout_sweep.json` (10 seed, có tiền đăng ký) |
| `fairshare/ablation_warmup__german__manifest.json` | `experiments/ablation_warmup.py` | Ablation warm-up không vào bài |
| `fairshare/exact_sv_corr__german__s0__loss__manifest.json` | `experiments/exact_shapley_correlation.py` | Thay bởi `shapley_fidelity.json` |
| `fairshare/incentive_audit__german__manifest.json` | `experiments/incentive_audit.py` | Thăm dò một seed, không đỡ bảng nào |
| `fairshare/holdout/incentive_audit__german__manifest.json` | như trên, biến thể holdout | như trên |
| `revision/bfwa_slack_bail_2seed.json` | `bfwa_slack_analysis.py` trên Bail, **2 seed** | Thay bởi `revision/bfwa_slack.json` (German, 7 mức ε × 200 vòng) |

Giữ lại thay vì xoá: chúng là dấu vết của các thăm dò đã thực sự chạy, và việc
một artifact tồn tại rồi bị rút là thông tin đáng lưu.
