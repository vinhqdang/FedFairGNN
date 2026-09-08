"""Phase 0 · bước 2 — giải nén repo trên VM, dựng cây thư mục, chạy test suite.

    colab exec -s $S -f experiments/colab/01_setup.py --timeout 900

Cổng ra: pytest phải trả exit code 0 (hiện tại 120 passed). Chưa xanh thì DỪNG.
Không khớp chuỗi số test — xem ghi chú ở dòng 50.

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

# data/ và results/ trong repo -> trỏ ra cache bền.
link = os.path.join(REPO, "data")
if os.path.islink(link) or os.path.exists(link):
    shutil.rmtree(link, ignore_errors=True)
    if os.path.islink(link):
        os.unlink(link)
os.symlink(DATA, link)

rlink = os.path.join(REPO, "results")
if os.path.islink(rlink) or os.path.exists(rlink):
    shutil.rmtree(rlink, ignore_errors=True)
    if os.path.islink(rlink):
        os.unlink(rlink)
os.makedirs("/content/results/fairshare", exist_ok=True)
os.symlink("/content/results", rlink)

# Nếu có data_raw.tgz được tải lên, bung ra vào /content
if os.path.exists("/content/data_raw.tgz"):
    sh("tar -xzf /content/data_raw.tgz -C /content/")
    credit_zip = "/content/data/raw/credit/credit_edges.txt.zip"
    if os.path.exists(credit_zip) and not os.path.exists("/content/data/raw/credit/credit_edges.txt"):
        sh(f"unzip -o -q {credit_zip} -d /content/data/raw/credit")

os.chdir(REPO)
print("repo:", sorted(os.listdir("."))[:12])
print("data ->", os.path.realpath(link))
print("results ->", os.path.realpath(rlink))
raw_dir = os.path.join(DATA, "raw")
cached = sorted(os.listdir(raw_dir)) if os.path.exists(raw_dir) else []
print("dataset đã cache:", cached or "(trống, sẽ tự tải)")

# --- GATE 0a: test suite ----------------------------------------------------
# Stream output trực tiếp ra stdout để tránh timeout websocket khi test chạy lâu.
print(">>> Running test suite (streaming output)...", flush=True)
p = subprocess.Popen("python -u -m pytest tests/ -q", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
lines = []
for line in p.stdout:
    lines.append(line)
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
print("\npytest exit code:", p.returncode, flush=True)
print("GATE 0a:", "PASS" if p.returncode == 0 else "*** FAIL — DỪNG LẠI ***", flush=True)
assert p.returncode == 0, f"GATE 0a failed with exit code {p.returncode}"
