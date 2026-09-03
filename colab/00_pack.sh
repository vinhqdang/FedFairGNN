#!/usr/bin/env bash
# Phase 0 · bước 1 — đóng gói repo + sinh nửa manifest phía LOCAL.
#
# Chạy trên máy local. Không đụng tới Colab.
# Sinh ra 2 file trong $SP:  fedfairgnn.tgz  và  manifest_local.json
#
#   SP=<scratchpad> bash colab/00_pack.sh
#
set -euo pipefail

REPO="${REPO:-/Users/anson/DS/Research/1_Paper/01.GNN/TrustFedGNN/FedFairGNN}"
: "${SP:?Phải đặt SP=<thư mục scratchpad>}"

cd "$REPO"

# --- provenance: phải lấy TRƯỚC khi đóng gói -------------------------------
COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="$(git rev-parse --short HEAD)"
BRANCH="$(git branch --show-current)"
if [ -n "$(git status --porcelain)" ]; then DIRTY=true; else DIRTY=false; fi

# Cây làm việc bẩn ⇒ commit hash KHÔNG định danh được thứ sắp chạy.
# Dừng lại thay vì ghi một manifest nói dối.
if [ "$DIRTY" = true ]; then
  echo "❌ Cây làm việc bẩn. Commit trước, hoặc đặt ALLOW_DIRTY=1 nếu cố ý." >&2
  git status --short >&2
  [ "${ALLOW_DIRTY:-0}" = "1" ] || exit 1
  echo "⚠️  ALLOW_DIRTY=1 — manifest sẽ ghi dirty:true; sha256 của tarball là" >&2
  echo "    định danh DUY NHẤT đáng tin của lần chạy này." >&2
fi

# --- đóng gói --------------------------------------------------------------
# ./data/raw chứ KHÔNG phải data: pattern 'data' khớp mọi thành phần đường dẫn
# và sẽ nuốt luôn src/data/ (đã cắn một lần → ModuleNotFoundError: src.data).
# COPYFILE_DISABLE=1: chặn AppleDouble ._* của macOS.
COPYFILE_DISABLE=1 tar \
  --exclude='./.git' --exclude='./data/raw' --exclude='./data/processed' \
  --exclude='./results/*' --exclude='./manuscript' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' --exclude='.DS_Store' \
  -czf "$SP/fedfairgnn.tgz" .

# --- kiểm tra tarball trước khi tin nó -------------------------------------
n_srcdata=$(tar -tzf "$SP/fedfairgnn.tgz" | grep -c 'src/data' || true)
n_rawdata=$(tar -tzf "$SP/fedfairgnn.tgz" | grep -c 'data/raw' || true)
[ "$n_srcdata" -gt 0 ] || { echo "❌ tarball thiếu src/data" >&2; exit 1; }
[ "$n_rawdata" -eq 0 ] || { echo "❌ tarball lỡ mang theo data/raw" >&2; exit 1; }

SHA="$(shasum -a 256 "$SP/fedfairgnn.tgz" | cut -d' ' -f1)"

cat > "$SP/manifest_local.json" <<JSON
{
  "commit": "$COMMIT",
  "commit_short": "$COMMIT_SHORT",
  "branch": "$BRANCH",
  "dirty": $DIRTY,
  "tarball_sha256": "$SHA",
  "packed_at": "$(date -u +%FT%TZ)"
}
JSON

echo "✅ $SP/fedfairgnn.tgz  ($(du -h "$SP/fedfairgnn.tgz" | cut -f1))"
echo "   commit $COMMIT_SHORT ($BRANCH) · dirty=$DIRTY"
echo "   sha256 $SHA"
