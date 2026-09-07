# `legacy_family_a/` — ĐÃ NGỪNG SỬ DỤNG (07-09-2026)

Bốn script trong thư mục này thuộc **Họ A**, cấu hình thực nghiệm cũ đã bị **loại bỏ khỏi
bài báo**. Giữ lại chỉ để đối chiếu lịch sử. **Không chạy. Không trích dẫn số từ chúng.**

| Script | Vì sao ngừng |
|---|---|
| `run_matrix.py` | $K/R/E$ khác Họ B (`local_epochs=2`, seeds `[0,1,2]`); ghi ra `summary.jsonl` **không có** `git_commit`/`device`/`timestamp` |
| `run_strong.py`, `run_strong2.py` | biến thể của `run_matrix.py`, cùng khiếm khuyết |
| `report.py` | chỉ đọc `results/summary.jsonl` — file **chưa từng được commit và không tồn tại** |

## Hệ quả phải xử lý

`report.py` từng là nguồn duy nhất sinh **5 hình** của bản thảo
(`pareto.pdf`, `privacy_bail.pdf`, `privacy_attack.pdf`, `robustness_byz.pdf`, `convergence.pdf`)
và 7 bảng (`main_auc/dpd/eod.tex`, `robustness.tex`, `efficiency.tex`, `privacy_attack.tex`,
`large_scale.tex`). Cả 12 artifact đó **mất nguồn** khi Họ A bị loại.

⇒ Phải viết `experiments/make_figures.py` đọc artifact **Họ B**. Xem STAGE 9 trong
`docs/04_experiment_execution.md`.

## Cấu hình Họ A (để đối chiếu, không dùng lại)

```python
ROUNDS  = {"german": 60, "bail": 50, "credit": 40, "elliptic": 25, "pokec_z": 30}
CLIENTS = {"german":  3, "bail":  5, "credit":  5, "elliptic": 10, "pokec_z": 10}
local_epochs = 2
SEEDS = [0, 1, 2]      # credit/elliptic/pokec_z chỉ [0, 1]
```

**Cấu hình chuẩn hiện hành (Họ B)** — `src/config.py :: ExperimentConfig.canonical()`:
`K=10`, `R=50`, `local_epochs=3`, seeds `{42…51}`, `aggregator="fu_shapley"`.
