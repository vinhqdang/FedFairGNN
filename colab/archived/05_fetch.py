"""Phase 0 · bước 6 — gói kết quả trên VM để kéo về.

    colab exec -s $S -f colab/05_fetch.py --timeout 300
    colab download -s $S /content/out.tgz $SP/out.tgz
    tar -xzf $SP/out.tgz -C $REPO/          # đổ vào results/ local

VM có thể bị thu hồi bất cứ lúc nào (nhàn rỗi ~90 phút, hoặc GPU bị rút).
Kết quả chưa tải về là kết quả đã mất — chạy bước này sau MỖI bước, không dồn.
"""
import subprocess

print(subprocess.run(
    "tar -czf /content/out.tgz -C /content results && "
    "ls -la /content/out.tgz && "
    "echo '--- nội dung ---' && tar -tzf /content/out.tgz | head -40",
    shell=True, capture_output=True, text=True).stdout)
