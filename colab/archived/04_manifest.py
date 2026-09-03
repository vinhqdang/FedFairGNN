"""Phase 0 · bước 5 — manifest môi trường (nguyên tắc §0.2-6).

    colab upload $SP/manifest_local.json /content/manifest_local.json
    colab exec -s $S -f colab/04_manifest.py --timeout 300

Ghép nửa provenance phía local (commit, dirty, sha256 tarball — do 00_pack.sh
sinh) với nửa môi trường phía VM (GPU, torch, PyG, CUDA).

Vì sao cần sha256 tarball chứ không chỉ commit hash: tarball đóng gói CÂY LÀM
VIỆC, không phải commit. Cây bẩn thì commit hash là một con số SAI chứ không
phải thiếu — người đọc sẽ tin nó. sha256 định danh đúng thứ đã chạy trong
mọi trường hợp.
"""
import json
import os
import platform
import subprocess

OUT = "/content/results/fairshare/manifest.json"
LOCAL = "/content/manifest_local.json"

import torch  # noqa: E402

m = {
    "created_at": subprocess.run("date -u +%FT%TZ", shell=True,
                                 capture_output=True, text=True).stdout.strip(),
    "host": "colab",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_used": os.environ.get("FEDFAIR_DEVICE", "cpu"),
}
try:
    import torch_geometric
    m["pyg"] = torch_geometric.__version__
except ImportError:
    m["pyg"] = None

for pkg in ("numpy", "scipy", "sklearn", "pandas"):
    try:
        m[pkg] = __import__(pkg).__version__
    except Exception:
        m[pkg] = None

if os.path.exists(LOCAL):
    m.update(json.load(open(LOCAL)))
else:
    m["provenance_warning"] = ("thiếu manifest_local.json — KHÔNG có commit hash "
                               "và KHÔNG có sha256. Upload nó rồi chạy lại.")

# sàn nhiễu, nếu bước 4 đã chạy: manifest phải mang theo con số này, vì mọi
# diễn giải về sau đều bị nó chặn trên.
nf = "/content/results/fairshare/phase0_noise_floor.json"
if os.path.exists(nf):
    m["noise_floor"] = json.load(open(nf)).get("noise_floor")

json.dump(m, open(OUT, "w"), indent=2)
print(json.dumps(m, indent=2))
