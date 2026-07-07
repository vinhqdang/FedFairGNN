"""Performance and fairness metrics.

Performance
    AUC-ROC, Average Precision (AP), F1, accuracy, FPR@k%TPR.
Group fairness (binary sensitive attribute S in {0,1})
    - Demographic Parity Difference (DPD)
    - Equal Opportunity Difference (EOD)
    - Equalized Odds (EO)
All fairness metrics accept *soft* probabilities and an optional decision
threshold; DPD is reported on soft scores (mean predicted probability gap),
matching the differentiable training objective, while EOD/EO use hard
predictions at the threshold, matching their textbook definitions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _np(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _scores(x):
    """Sanitised probability scores: a diverged (NaN/inf) model -> 0.5 so that
    metrics report degraded performance instead of raising (important for the
    robustness study, where attacks can blow up predictions)."""
    return np.nan_to_num(_np(x).ravel().astype(float), nan=0.5, posinf=1.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def auc_roc(y_true, y_score) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def average_precision(y_true, y_score) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float(y_true.mean())


def f1(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    y_pred = (y_score >= threshold).astype(int)
    try:
        return float(f1_score(y_true, y_pred))
    except ValueError:
        return 0.0


def accuracy(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    return float(((y_score >= threshold).astype(int) == y_true).mean())


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.8) -> float:
    """False-positive rate at a target true-positive rate (operational metric)."""
    y_true, y_score = _np(y_true).ravel(), _scores(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def demographic_parity_difference(y_score, sensitive, threshold=None) -> float:
    """|E[score | S=0] - E[score | S=1]| (soft) or rate gap (if threshold set)."""
    y_score, s = _scores(y_score), _np(sensitive).ravel()
    v = (y_score >= threshold).astype(float) if threshold is not None else y_score
    m0, m1 = s == 0, s == 1
    if m0.sum() == 0 or m1.sum() == 0:
        return 0.0
    return float(abs(v[m0].mean() - v[m1].mean()))


def equal_opportunity_difference(y_true, y_score, sensitive, threshold=0.5) -> float:
    """|TPR(S=0) - TPR(S=1)|  (fairness on the positive class only)."""
    y_true, y_score, s = _np(y_true).ravel(), _scores(y_score), _np(sensitive).ravel()
    y_pred = (y_score >= threshold).astype(int)

    def tpr(group):
        pos = (y_true == 1) & group
        return y_pred[pos].mean() if pos.sum() > 0 else 0.0

    return float(abs(tpr(s == 0) - tpr(s == 1)))


def equalized_odds(y_true, y_score, sensitive, threshold=0.5) -> float:
    """max(|TPR gap|, |FPR gap|) across groups."""
    y_true, y_score, s = _np(y_true).ravel(), _scores(y_score), _np(sensitive).ravel()
    y_pred = (y_score >= threshold).astype(int)

    def rate(group, label, pred):
        m = (y_true == label) & group
        return (y_pred[m] == pred).mean() if m.sum() > 0 else 0.0

    tpr_gap = abs(rate(s == 0, 1, 1) - rate(s == 1, 1, 1))
    fpr_gap = abs(rate(s == 0, 0, 1) - rate(s == 1, 0, 1))
    return float(max(tpr_gap, fpr_gap))


def all_metrics(y_true, y_score, sensitive, threshold: float = 0.5) -> Dict[str, float]:
    """Convenience: compute the full metric suite in one call."""
    return {
        "auc": auc_roc(y_true, y_score),
        "ap": average_precision(y_true, y_score),
        "f1": f1(y_true, y_score, threshold),
        "acc": accuracy(y_true, y_score, threshold),
        "fpr@80tpr": fpr_at_tpr(y_true, y_score, 0.8),
        "dpd": demographic_parity_difference(y_score, sensitive),
        "eod": equal_opportunity_difference(y_true, y_score, sensitive, threshold),
        "eo": equalized_odds(y_true, y_score, sensitive, threshold),
    }


# Backwards-compatible aliases (older code / notebooks referenced these names)
calculate_auc = auc_roc
calculate_f1 = f1


def calculate_dpd(y_score, sensitive) -> float:
    return demographic_parity_difference(y_score, sensitive)
