"""Phase 0 · bước 2 — giải nén repo trên VM, dựng cây thư mục, chạy test suite.

    colab exec -s $S -f colab/01_setup.py --timeout 900

Cổng ra: phải thấy `26 passed`. Chưa xanh thì DỪNG, không chạy gì tiếp.

Lưu ý bố cục: dataset cache và results nằm NGOÀI thư mục repo, vì mỗi lần
deploy code lại là xoá sạch repo. Không có symlink thì mỗi lần re-upload là
tải lại 20 MB bail / 120 MB credit.
"""
import os
import shutil
import subprocess

REPO = "/content/FedFairGNN"
DATA = "/content/data"
RESULTS = "/content/results/fairshare"
LOGS = "/content/logs"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


# --- cây thư mục bền, không bị deploy xoá ----------------------------------
for d in (DATA, RESULTS, LOGS):
    os.makedirs(d, exist_ok=True)

# --- deploy code ------------------------------------------------------------
shutil.rmtree(REPO, ignore_errors=True)
os.makedirs(REPO, exist_ok=True)
r = sh(f"tar -xzf /content/fedfairgnn.tgz -C {REPO}")
assert r.returncode == 0, r.stderr[-2000:]

# data/ trong repo -> trỏ ra cache bền. Không script nào nhận --data_root nên
# symlink là cách rẻ nhất, và không đụng một dòng mã nguồn nào.
link = os.path.join(REPO, "data")
if os.path.islink(link) or os.path.exists(link):
    shutil.rmtree(link, ignore_errors=True)
    if os.path.islink(link):
        os.unlink(link)
os.symlink(DATA, link)

os.chdir(REPO)
print("repo:", sorted(os.listdir("."))[:12])
print("data ->", os.path.realpath(link))
print("dataset đã cache:", sorted(os.listdir(DATA)) or "(trống, sẽ tự tải)")

# --- GATE 0a: test suite ----------------------------------------------------
# Phán quyết bằng EXIT CODE của pytest, không bằng khớp chuỗi trên stdout.
# Hai lần đã sai vì chuỗi: hard-code "26 passed" hoá đỏ ngay khi thêm test đúng,
# rồi tìm "error" lại bắt nhầm chữ trong một UserWarning. Exit code là hợp đồng
# duy nhất pytest thực sự bảo đảm.
r = sh("python -u -m pytest tests/ -q")
tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-12:])
print(tail)
print("pytest exit code:", r.returncode)
print("GATE 0a:", "PASS" if r.returncode == 0 else "*** FAIL — DỪNG LẠI ***")
