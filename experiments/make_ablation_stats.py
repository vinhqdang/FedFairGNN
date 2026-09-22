"""Recompute every ablation p-value from the per-seed arrays, with Holm correction.

Until this script existed the ablation statistics quoted in the manuscript were
computed by hand and stored nowhere. One of them was wrong: the M3 contrast was
printed as p = 0.0020, which is the value from the adjacent M4 row -- the true
exact-Wilcoxon value is p = 0.0488, recoverable by any reader from the released
`canonical_suite.json`. Every ablation number in the paper must now come from
the artifact this writes, so that a reader recomputing it gets what we printed.

Holm-Bonferroni is applied per metric across the family of arms compared against
M1, which is the same correction the SOTA tables already use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments"))

from stats import holm_bonferroni, paired_report  # noqa: E402

from src.utils.provenance import build_manifest  # noqa: E402

REFERENCE_ARM = "M1_Full"

# metric -> lower_is_better
METRICS = {
    "auc": False,
    "dpd_soft": True,
    "dpd_hard": True,
    "eod": True,
    "omega_w": True,
}


def _column(arm_record: dict, metric: str) -> list:
    return [float(r[metric]) for r in arm_record["per_seed"]]


def compute(matrix: dict) -> dict:
    if REFERENCE_ARM not in matrix:
        raise KeyError(f"reference arm {REFERENCE_ARM!r} missing from ablation matrix")

    arms = [a for a in matrix if a != REFERENCE_ARM]
    comparisons: dict = {a: {} for a in arms}

    for metric, lower_is_better in METRICS.items():
        ref = _column(matrix[REFERENCE_ARM], metric)
        pvals = {}
        for arm in arms:
            other = _column(matrix[arm], metric)
            if len(other) != len(ref):
                raise ValueError(
                    f"{arm} has {len(other)} seeds but {REFERENCE_ARM} has {len(ref)}; "
                    "paired tests require the same seed set"
                )
            rep = paired_report(ref, other, lower_is_better=lower_is_better)
            rep["n_seeds"] = len(ref)
            rep["reference_mean"] = sum(ref) / len(ref)
            rep["arm_mean"] = sum(other) / len(other)
            comparisons[arm][metric] = rep
            pvals[arm] = rep["p_wilcoxon"]

        for arm, survives in holm_bonferroni(pvals).items():
            comparisons[arm][metric]["holm_bonferroni_sig"] = bool(survives)

    return comparisons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="results/canonical_suite.json")
    ap.add_argument("--out-json", default="results/ablation_statistics.json")
    args = ap.parse_args()

    with open(args.suite) as f:
        suite = json.load(f)
    matrix = suite["component_ablation_matrix"]
    comparisons = compute(matrix)

    payload = {
        "manifest": build_manifest(
            experiment="ablation_statistics",
            args={
                "suite": args.suite,
                "reference_arm": REFERENCE_ARM,
                "metrics": list(METRICS),
                "correction": "holm_bonferroni per metric across the arm family",
            },
        ),
        "reference_arm": REFERENCE_ARM,
        "comparisons": comparisons,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")

    for arm in sorted(comparisons):
        for metric in ("auc", "dpd_hard", "eod"):
            r = comparisons[arm][metric]
            flag = "holm-sig" if r["holm_bonferroni_sig"] else "not-holm-sig"
            print(
                f"{REFERENCE_ARM} vs {arm:<18s} {metric:<9s} "
                f"delta={r['mean_diff']:+.5f}  p={r['p_wilcoxon']:.4f}  "
                f"wins={r['wins']}  {flag}"
            )
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
