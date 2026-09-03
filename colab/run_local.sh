#!/usr/bin/env bash
# Chạy Phase 0b / Phase 1 NGAY TRÊN MÁY LOCAL, không cần Colab.
#
#   bash colab/run_local.sh setup     # dựng venv + cài torch/PyG (một lần)
#   bash colab/run_local.sh gate0     # GATE 0a + 0b — BẮT BUỘC trước khi tin số local
#   bash colab/run_local.sh phase1    # GATE 1 + GATE 1-C
#
# Vì sao chạy local là hợp lệ, và vì sao `gate0` là bắt buộc
# ----------------------------------------------------------
# `F22` đã chốt thiết bị canonical là **CPU**. Kể từ đó Colab không còn cấp thứ
# gì mà máy này không có — nó chỉ là một CPU Linux ở xa. Lý do tồn tại của cả
# runbook (mượn T4) đã biến mất cùng F22.
#
# NHƯNG `1_2_phase0.md` §2.6.1 nói rõ: bit-exact đã chứng minh là
# **within-machine, within-session**. Tái lập cross-machine chưa từng được
# khẳng định, và **không nên** khẳng định: máy này là arm64/macOS (Accelerate
# BLAS), VM là x86_64/Linux (MKL/OpenBLAS). Đó là HAI môi trường.
#
# ⇒ Muốn lấy số ở local thì phải chạy lại GATE 0 **ở local** để chứng minh
#   bit-exact **ở đây**. Không được kế thừa phán quyết của VM.
# ⇒ Và tuyệt đối KHÔNG trộn artifact hai nơi. Đó đúng là cơ chế đã đẻ ra con số
#   `114×` (tử số một môi trường, mẫu số môi trường kia).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/FedFairGNN"
VENV="$ROOT/.venv-local"
RESULTS="$REPO/results/local"
TMP="${TMPDIR:-/tmp}/fedfair"
PY="$VENV/bin/python"

export FEDFAIR_REPO="$REPO"
export FEDFAIR_RESULTS="$RESULTS"
export FEDFAIR_TMP="$TMP"
export FEDFAIR_DEVICE=cpu
export PYTHONUNBUFFERED=1
mkdir -p "$RESULTS" "$TMP"

case "${1:-}" in
setup)
  command -v uv >/dev/null || { echo "❌ cần uv (https://astral.sh/uv)" >&2; exit 1; }
  uv venv --python 3.12 "$VENV"
  # torch_geometric 2.x dùng scatter thuần torch ⇒ KHÔNG cần torch-scatter/
  # torch-sparse (hai gói đó phải biên dịch và là chỗ hỏng thường gặp nhất trên
  # arm64). Cài đúng hai gói dưới đây là đủ cho đường code CPU.
  VIRTUAL_ENV="$VENV" uv pip install torch torch_geometric \
    numpy scipy scikit-learn pandas pytest
  "$PY" -c "import torch,torch_geometric as g;print('torch',torch.__version__,'| pyg',g.__version__,'| threads',torch.get_num_threads())"
  ;;

gate0)
  echo "=== GATE 0a — test suite (phán quyết bằng EXIT CODE) ==="
  ( cd "$REPO" && "$PY" -m pytest tests/ -q )
  echo "GATE 0a: PASS (exit 0)"
  echo
  echo "=== GATE 0b — bit-exact TRÊN MÁY NÀY ==="
  for ds in german bail; do
    echo "--- $ds ---"
    FEDFAIR_DATASETS="$ds" "$PY" "$ROOT/colab/02_determinism_gate.py"
  done
  echo
  echo "⚠️  GATE 0b xanh ở đây CHỈ chứng nhận bit-exact TRÊN MÁY NÀY."
  echo "    Hash sẽ KHÁC hash của VM (arm64 vs x86) — đó là bình thường, và là"
  echo "    lý do không được trộn artifact hai nơi."
  ;;

phase1)
  test -f "$RESULTS/fairshare/phase0_determinism__german.json" || {
    echo "❌ Chưa có artifact GATE 0b local. Chạy 'gate0' trước." >&2
    echo "   Kế thừa phán quyết bit-exact của VM sang máy này là không hợp lệ." >&2
    exit 1; }
  echo "=== GATE 1 — đường ống ==="
  "$PY" "$ROOT/colab/09_phase1_rerun.py"
  echo
  echo "=== GATE 1-C — chuỗi toán §1.3–§1.4 ==="
  "$PY" "$ROOT/colab/10_phase1_checkpoints.py"
  ;;

*)
  echo "dùng: bash colab/run_local.sh {setup|gate0|phase1}" >&2; exit 1;;
esac
