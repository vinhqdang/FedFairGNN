"""Statistical testing and reporting module for Q1 publications.

Implements paired Wilcoxon signed-rank test, Cohen's d_z effect size,
bootstrap 95% confidence intervals, and family-wise Holm-Bonferroni correction.
"""
from typing import Dict, List, Tuple
import numpy as np
import math

from scipy.stats import wilcoxon


def paired_report(ours: List[float], base: List[float], lower_is_better: bool = False) -> Dict:
    """So sánh bắt cặp theo seed. Trả về đủ 4 đại lượng bắt buộc cho bảng Q1."""
    o, b = np.asarray(ours, float), np.asarray(base, float)
    assert o.shape == b.shape, "phải bắt cặp theo cùng bộ seed"
    d = o - b
    
    # Wilcoxon signed-rank test
    if np.all(d == 0):
        stat, p = 0.0, 1.0
    else:
        try:
            stat, p = wilcoxon(o, b)
        except Exception:
            stat, p = 0.0, 1.0

    wins = int(np.sum(d < 0) if lower_is_better else np.sum(d > 0))
    
    # Cohen's dz bắt cặp
    std_d = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    dz = float(d.mean() / std_d) if std_d > 0 else 0.0
    
    # Bootstrap CI 95% cho hiệu trung bình
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(10_000)]
    ci95 = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    
    return {
        "mean_diff": float(d.mean()),
        "ci95": ci95,
        "cohens_dz": dz,
        "wins": f"{wins}/{len(d)}",
        "p_wilcoxon": float(p),
    }


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    """Hiệu chỉnh đa so sánh khi đối sánh Ours với N baselines theo từng họ (family).

    Một p-value không hữu hạn (nan/inf) làm hỏng TOÀN BỘ họ, không chỉ mục của nó:
    so sánh với nan luôn trả về False nên sorted() cho thứ tự tuỳ tiện, và vì `reject`
    là biến tích luỹ, một phần tử sai thứ tự lật kết quả của mọi phần tử sau nó.
    Quan sát thực tế (RUN-E2, docs/04 §11.3.6): một nan đã biến p=0.0020 thành
    "không có ý nghĩa" trong khi p=0.0039 lại thành "có". Nên chặn ở cửa.
    """
    bad = {k: v for k, v in pvals.items() if not math.isfinite(v)}
    if bad:
        raise ValueError(
            f"holm_bonferroni nhận p-value không hữu hạn: {bad}. "
            "Hãy loại bỏ (và khai báo) các contrast không tính được TRƯỚC khi hiệu chỉnh, "
            "đừng để chúng đi vào họ kiểm định."
        )
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    reject = True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = reject and (p <= thresh)
        out[k] = reject
    return out
