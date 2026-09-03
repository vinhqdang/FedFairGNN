# `colab/` — Bộ Công Cụ & Script Thực Thi Google Colab VM

Thư mục lưu trữ các script điều phối từ xa (*remote orchestration*) trên Google Colab GPU/CPU theo quy chuẩn **TrustFedGNN (Q1 ML Protocol)**.

---

## 🚀 1. CÁC SCRIPT ĐANG HOẠT ĐỘNG TRONG QUY TRÌNH MỚI (ACTIVE PIPELINE)

| File | Môi Trường | Mục Đích Thực Thi | Trạng Thái / Gate |
|---|:---:|---|:---:|
| [`00_pack.sh`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/00_pack.sh) | **Local** | Đóng gói repo `FedFairGNN` thành tarball + sinh manifest provenance | ✅ Active |
| [`01_setup.py`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/01_setup.py) | **Colab VM** | Giải nén repo, liên kết dữ liệu, chạy 41/41 unit tests trên VM | **GATE 0a (PASS)** |
| [`11_stage4_1_smoke.py`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/11_stage4_1_smoke.py) | **Colab VM** | Chạy Fast Smoke Test trên đồ thị tổng hợp để xác thực đường ống | **STAGE 4.1 PASS** |
| [`15_stage4_remediation.py`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/15_stage4_remediation.py) | **Colab GPU** | Chạy bộ thực nghiệm Canonical 3 seeds (German, Bail no-leakage, FSER sweep, Byzantine defense) | **STAGE 4.2 & 4.5 PASS** |
| [`run_local.sh`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/run_local.sh) | **Local** | Chạy kiểm thử nội bộ trên máy local | ✅ Active |

---

## 📦 2. THƯ MỤC LƯU TRỮ LỊCH SỬ (`archived/`)

Toàn bộ các script và kết quả thử nghiệm từ các pha trước (Phase 0, Phase 1, Phase 2, các script refine cũ) đã được gom vào [`colab/archived/`](file:///Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab/archived):
- `02_determinism_gate.py`, `04_manifest.py`, `05_fetch.py`, `06_phase1_2_harness.py`, `07_run_single_harness.py`, `08_phase2_rerun.py`, `09_phase1_rerun.py`, `10_phase1_checkpoints.py`
- `12_stage4_2_small_benchmarks.py`, `13_stage4_5_ablations.py`, `14_stage4_2_and_4_5_refine.py`, `check_vm_results.py`
- `legacy_results/` (Chứa các folder kết quả cũ: `4edfbd4...`, `de5a6d1...`, `phase1c`, v.v.)

---

## 🛠️ 3. QUY TRÌNH CHẠY CANONICAL TRÊN COLAB GPU T4

```bash
P=/Users/anson/.colab-profiles/gnn
S=stage4_gpu
REPO=/Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN
SP=/Users/anson/.gemini/antigravity-ide/brain/5b909278-b945-4704-8f79-513c211c7d29/scratch
C=/Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/colab

# 1. Đóng gói mã nguồn
SP=$SP bash $C/00_pack.sh

# 2. Khởi tạo máy ảo GPU T4
HOME=$P colab new -s $S --gpu T4
HOME=$P colab install -s $S torch_geometric
HOME=$P colab upload -s $S $SP/fedfairgnn.tgz /content/fedfairgnn.tgz
HOME=$P colab upload -s $S $SP/manifest_local.json /content/manifest_local.json

# 3. Chạy kiểm thử môi trường (GATE 0a)
HOME=$P colab exec -s $S -f $C/01_setup.py --timeout 300

# 4. Chạy bộ thực nghiệm Canonical (Stage 4 Remediation)
HOME=$P colab exec -s $S -f $C/15_stage4_remediation.py --timeout 1800

# 5. Tải kết quả về local
HOME=$P colab download -s $S /content/FedFairGNN/results/stage4_remediation_results.json $REPO/results/stage4_remediation_results.json

# 6. Dọn dẹp máy ảo
HOME=$P colab stop -s $S
```
