# `experiments/` — Bộ điều khiển thực nghiệm TrustFedGNN

> **File điều khiển DUY NHẤT** của toàn bộ chuỗi `colab → runner → results → tables`.
> Trạng thái từng stage, cổng kiểm toán, và phán quyết khoa học nằm ở
> [`docs/04_experiment_execution.md`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/docs/04_experiment_execution.md).
> Nhật ký thay đổi và thời gian chạy nằm ở
> [`docs/CHANGELOG.md`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/docs/CHANGELOG.md).

---

## 1. Cấu hình chuẩn — chỉ còn **một** họ

Từ 07-09-2026, dự án dùng **duy nhất Họ B**. Họ A đã bị loại bỏ hoàn toàn
(xem [`legacy_family_a/README.md`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN/experiments/legacy_family_a/README.md)).

```python
# src/config.py :: ExperimentConfig.canonical()   ◄── NGUỒN CHÂN LÝ DUY NHẤT
dataset="german", num_clients=5, rounds=20, dirichlet_alpha=0.3,
model="trustfedgnn", aggregator="fu_shapley",
fu_alpha=0.1, fu_ema_beta=0.9, fu_val_source="server_holdout",
fairness_weight=1.0, beta_init=0.5, fser_mode="sub",
dp_enabled=True, dp_mode="ftgd"
# mặc định khác: local_epochs=3, dp_epsilon=8.0, dp_delta=1e-5, sampling=False
```

Các runner SOTA override thành `num_clients=10, rounds=50`, seeds `{42…51}`.
**Mọi ablation phải khởi tạo từ `canonical()` rồi override tường minh** — không dựng config thủ công.

---

## 2. Bố cục thư mục

```
experiments/
├── README.md                      ◄── file này
├── colab/                         điều phối VM từ xa  (§3)
│   ├── 00_pack.sh                 [LOCAL]  đóng gói repo + manifest
│   ├── 01_setup.py                [VM]     giải nén, symlink, pytest — GATE 0
│   ├── 11_stage4_1_smoke.py       [VM]     smoke test — STAGE 2
│   ├── 15_stage4_remediation.py   [VM]     canonical + ablation — STAGE 3
│   ├── run_local.sh               [LOCAL]  chạy CPU không cần Colab
│   └── archived/                  script các pha cũ + legacy_results/
│
├── stage4_1_smoke_test.py         STAGE 2  · CPU
├── stage4_remediation_runner.py   STAGE 3  · CPU hoặc GPU
├── stage4_3_pokecz_runner.py      STAGE 4  · GPU
├── stage4_3_credit_runner.py      STAGE 4  · GPU
├── stage4_3_byzantine_sweep_runner.py  STAGE 5 · GPU
├── stage4_3_shapley_runner.py     STAGE 6  · CPU
├── run_large_scale.py             STAGE 8  · CPU (ogbn-products, NeighborLoader)
│
├── revision/                      STAGE 5–7 · bộ thực nghiệm phản biện
├── methods.py                     đăng ký 16 baseline + CGSV + biến thể Ours
├── fairshare_common.py            tiện ích dùng chung (make_trainer, homophily)
├── run_experiment.py              `run_one()` — vòng huấn luyện đơn
│
├── generate_publication_stats.py  results → consolidated_statistics.json
├── generate_latex_publication_tables.py   results → 6 bảng .tex
├── plot_pareto_frontier.py        results → pareto_frontier PNG
│
└── legacy_family_a/               ⛔ ĐÃ NGỪNG — không chạy, không trích dẫn
```

---

## 3. Chạy ở CPU hay GPU? — quy tắc dứt khoát

**Nguyên tắc nền:** trên CUDA, `scatter`/`index_add` dùng atomics ⇒ hai lần chạy **cùng seed**
lệch tới **0,103 AUC**. Trên CPU, `sha256(global_flat)` trùng khít.

| Loại công việc | Thiết bị | Vì sao |
|---|:--:|---|
| Kiểm tra tái lập bit-exact, đối chứng hash | **CPU bắt buộc** | GPU không tất định — hash vô nghĩa |
| Smoke test, unit test, script tất định (DP accounting, trust score) | **CPU** | rẻ, vài giây–vài phút |
| Ablation $n{=}3$ seed trên German/Bail ($\le 19$k nút) | **CPU đủ** | GPU không rút ngắn đáng kể ở quy mô này |
| SOTA matrix Pokec-z / Credit / Elliptic (30k–204k nút, 10 seed) | **GPU T4** | CPU mất hàng chục giờ |
| Byzantine sweep, robustness đa seed | **GPU T4** | nhiều run ngắn ⇒ tổng lớn |
| ogbn-products 2,4M nút (NeighborLoader) | **CPU chấp nhận được** | nghẽn ở lấy mẫu láng giềng, không ở GEMM |

> **Thiết bị là một phần của manifest.** `FEDFAIR_DEVICE` mặc định `cpu`
> (`src/config.py`). Mỗi artifact ghi `device` vào `manifest` — kết quả CPU và GPU
> **không được trộn trong cùng một bảng**.
>
> **Trên CPU**, $|\Delta| \ne 0$ giữa hai lần cùng cấu hình là **khác biệt CODE, phải truy**.
> **Trên GPU**, đừng dùng hash làm cổng kiểm tra.

---

## 4. Quy trình A→Z

### Bước 0 — Cổng mã nguồn (LOCAL, CPU, ~1 phút)

```bash
cd /Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN
git status -sb                                    # phải sạch
git log -1 --oneline
/Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/.venv-local/bin/python -m pytest -q
#   CỔNG: 120 passed. FAIL bất kỳ ⇒ DỪNG.
```

### Bước 1 — Đóng gói & dựng VM

```bash
P=/Users/anson/.colab-profiles/gnn          # profile đã kiểm chứng
S=rerun
REPO=/Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN
C=$REPO/experiments/colab
SP=<thư mục scratchpad>

SP=$SP bash $C/00_pack.sh                   # → fedfairgnn.tgz + manifest_local.json
HOME=$P colab new     -s $S --gpu T4
HOME=$P colab install -s $S torch_geometric # VM MỚI KHÔNG CÓ SẴN
HOME=$P colab upload  -s $S $SP/fedfairgnn.tgz      /content/fedfairgnn.tgz
HOME=$P colab upload  -s $S $SP/manifest_local.json /content/manifest_local.json
HOME=$P colab exec    -s $S -f $C/01_setup.py --timeout 900     # GATE 0 trên VM
```

**Bốn cạm bẫy đã trả giá:**
1. VM mới **không có** `torch_geometric` — phải `colab install` trước.
2. VM nhàn rỗi bị thu hồi sau **~90 phút** (không phải 24h) ⇒ job dài chạy nền + polling.
3. `colab log -s <name>` chỉ khôi phục **cell đã hoàn tất** — không theo dõi được job đang chạy.
   Mọi runner phải ghi xuống `results/` rồi `download`, không dựa vào stdout.
4. `jupyter-kernel-client` phải ghim **`==0.9.0`**; 1.0.0 làm hỏng mọi `colab exec`.

### Bước 2 — Chạy theo thứ tự stage

| # | Lệnh trên VM | Stage | Thiết bị | Artifact | Phút |
|:--:|---|:--:|:--:|---|:--:|
| 1 | `python experiments/preflight_dataset_audit.py` ¹ | 1 | CPU | `preflight_dataset_audit.json` | 2 |
| 2 | `python experiments/stage4_3_pokecz_runner.py` | 4 | GPU | `stage4_3_pokecz_results.json` | 50 |
| 3 | `python experiments/stage4_3_credit_runner.py` | 4 | GPU | `stage4_3_credit_results.json` | 60 |
| 4 | `python experiments/stage4_4_elliptic_runner.py` ¹ | 4 | GPU | `stage4_4_elliptic_results.json` | 30 |
| 5 | `python experiments/stage4_3_byzantine_sweep_runner.py` | 5 | GPU | `stage4_3_byzantine_results.json` | 13 |
| 6 | `python experiments/revision/robustness_multiseed.py` ² | 5 | GPU | `revision/robustness_multiseed.json` | 25 |
| 7 | `python experiments/revision/adaptive_poisoner.py` ² | 5 | GPU | `revision/adaptive_poisoner_results.json` | 15 |
| 8 | `python experiments/stage4_3_shapley_runner.py` | 6 | CPU | `stage4_3_shapley_results.json` | 2 |
| 9 | `python experiments/revision/ablation_grid_runner.py` | 7 | GPU | `revision/ablation_grid_results.json` | 30 |
| 10 | `python experiments/revision/fser_beta_analysis.py` | 7 | GPU | `revision/fser_beta_analysis.json` | 10 |

¹ chưa tồn tại — phải viết trước (xem §6) · ² phải sửa danh sách aggregator trước (xem §6)

**STAGE 3 không chạy lại** — `stage4_remediation_results.json` đã ở commit hậu-fix `73a251e`.
Tổng ≈ **4 giờ**; ngân sách **6 giờ** tính dự phòng session death.

### Bước 3 — Tải về & nghiệm thu

```bash
for f in preflight_dataset_audit stage4_3_pokecz_results stage4_3_credit_results \
         stage4_4_elliptic_results stage4_3_byzantine_results stage4_3_shapley_results; do
  HOME=$P colab download -s $S /content/FedFairGNN/results/$f.json $REPO/results/$f.json
done
HOME=$P colab download -s $S /content/FedFairGNN/results/revision $REPO/results/revision
HOME=$P colab stop -s $S        # BẮT BUỘC — không gì tự thu hồi VM ngoài giới hạn 24h

bash experiments/verify_artifacts.sh      # §5
```

### Bước 4 — Sinh bảng & hình (LOCAL, CPU)

```bash
python experiments/generate_publication_stats.py        # → consolidated_statistics.json
python experiments/generate_latex_publication_tables.py # → 6 bảng .tex
python experiments/revision/dp_accounting_table.py      # → dp_accounting.{json,tex}
python experiments/plot_pareto_frontier.py              # → pareto PNG
git add -f results/ manuscript/tables/
git commit -m "data: post-fix re-run (Họ B)"
```

---

## 5. Cổng nghiệm thu artifact

Chạy sau **mỗi** runner. Đây là **cổng**, không phải gợi ý.

```bash
cd /Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN
HEAD=$(git rev-parse HEAD)
for f in results/*.json results/revision/*.json; do
  python3 - "$f" "$HEAD" <<'PY'
import json, sys
f, head = sys.argv[1], sys.argv[2]
try: d = json.load(open(f))
except Exception as e: print("  ?? %s: %s" % (f, e)); raise SystemExit
if isinstance(d, list): print("  -- %s: list, không manifest" % f); raise SystemExit
m = d.get('manifest') or d.get('_manifest') or {}
c = (m.get('git_commit') or '')[:7]
stale = '_STALENESS_NOTICE' in d
dirty = m.get('git_dirty', None)
ok = (c == head[:7]) and (not stale) and (dirty is False)
print("  %s%-52s commit=%-8s stale=%-5s dirty=%s" % ('OK ' if ok else 'XX ', f, c or '-', stale, dirty))
PY
done
```

Vì sao cần: commit `73a251e` vá sự cố **"false completion"** — `_load_checkpoint` từng coi mọi
`results/*.json` có sẵn là tiến độ của chính nó, khiến một lần deploy "hoàn tất" trong ~15 giây
và suýt phát hành lại 100% số liệu tiền-fix dưới nhãn "vừa chạy xong".

---

## 6. Việc phải làm trước khi mở VM (LOCAL, CPU, ~1 giờ)

| # | Việc | File |
|:--:|---|---|
| P1 | `ResultLogger.save()` phải ghi `git_commit`, `git_dirty`, `device`, `timestamp` | `src/utils/logging_utils.py` |
| P2 | Viết `preflight_dataset_audit.py`: đo $h_s$ + zero-leakage cho **6** bộ dữ liệu, ghi manifest | `experiments/` |
| P3 | `datasets.tex` phải **đọc** artifact P2 thay vì chuỗi cứng | `generate_latex_publication_tables.py:315-333` |
| P4 | `DATASET_SPECS` đặt theo **Họ B** ($K{=}10$, $E{=}3$, $R{=}50$) | `revision/dp_accounting_table.py` |
| P5 | Thêm `fu_shapley`, `robust_fu_shapley`, `cgsv` vào danh sách aggregator quét | `revision/robustness_multiseed.py`, `revision/adaptive_poisoner.py` |
| P6 | Viết `stage4_4_elliptic_runner.py` theo khuôn `stage4_3_pokecz_runner.py` | `experiments/` |
| P7 | Viết `make_figures.py` sinh 5 hình từ artifact Họ B (thay `report.py` đã loại) | `experiments/` |

---

## 7. Đăng ký phương pháp

`methods.py` là nơi **duy nhất** ánh xạ tên → cấu hình:

```python
"fedfairgnn":      dict(model="trustfedgnn", aggregator="fu_shapley",        dp_enabled=True,  dp_mode="ftgd")
"fedfairgnn-nodp": dict(model="trustfedgnn", aggregator="fu_shapley",        dp_enabled=False, dp_mode="ftgd")
"ours-robust":     dict(model="trustfedgnn", aggregator="robust_fu_shapley", dp_enabled=True,  dp_mode="ftgd")
"ours-nobfwa":     dict(model="trustfedgnn", aggregator="fedavg",            dp_enabled=False)   # ⚠️ tên gọi sai lịch sử
```

> ⚠️ **`ours-nobfwa` là tên gọi nhầm lẫn lịch sử** — nó tắt FU-Shapley (về FedAvg), không liên
> quan gì tới BFWA. Đây chính là gốc của việc bản thảo mô tả BFWA trong khi mã chạy FU-Shapley.
> Giữ khoá cũ để không mất kết quả đã cache; **đừng suy ra ý nghĩa từ tên**.

`ROBUST_AGGREGATORS` đã gồm `fu_shapley`, `robust_fu_shapley`, `cgsv` (dòng 120–126).

---

## 8. Nguyên tắc bất di bất dịch

1. **Số chỉ ghi một lần.** Mỗi kết quả vào đúng một artifact; bảng và tài liệu **trỏ tới**, không chép lại.
2. **Thay đổi đi một chiều:** `docs/02` (toán) → `docs/03` (ánh xạ AST) → code → test → artifact → bảng → `docs/04` + `docs/CHANGELOG.md`.
3. **Không con số nào vào bản thảo** nếu artifact tương ứng không qua cổng §5.
4. **Thiết bị phải khai báo** trong manifest; không trộn CPU/GPU trong một bảng.
5. **`legacy_family_a/` không bao giờ được chạy lại.**
