"""Step 4.2: Empirical verification of Theorem 2(1) -- Metadata Immunity.

Validates that FU-Shapley and Robust FU-Shapley weights are bit-exact independent
of client self-reported metadata (demographic disparity, performance, sample counts),
while the baseline BFWA aggregator is vulnerable to fabricated metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import torch

sys.path.insert(0, os.path.abspath("."))
from src.federated.aggregation import aggregate
from src.utils.provenance import build_manifest


def run_metadata_immunity_audit(out_dir: str = "results/fairshare"):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    K, P = 5, 64
    updates = [torch.randn(P) for _ in range(K)]
    g_target = torch.randn(P)

    meta_honest = [
        {"perf": 0.72, "dpd": 0.15, "eod": 0.10, "n": 100},
        {"perf": 0.68, "dpd": 0.08, "eod": 0.05, "n": 120},
        {"perf": 0.75, "dpd": 0.18, "eod": 0.12, "n": 90},
        {"perf": 0.70, "dpd": 0.11, "eod": 0.07, "n": 110},
        {"perf": 0.65, "dpd": 0.05, "eod": 0.03, "n": 80},
    ]
    meta_lying = deepcopy(meta_honest)
    # Adversary lies aggressively: claims 99% accuracy and 0 demographic disparity
    meta_lying[0].update(perf=0.99, dpd=0.0, eod=0.0)

    # 1. FU-Shapley
    _, info_fu_h = aggregate("fu_shapley", updates, meta_honest, g_target=g_target)
    _, info_fu_l = aggregate("fu_shapley", updates, meta_lying, g_target=g_target)
    w_fu_h = [float(x) for x in info_fu_h["weights"]]
    w_fu_l = [float(x) for x in info_fu_l["weights"]]
    max_diff_fu = max(abs(a - b) for a, b in zip(w_fu_h, w_fu_l))
    assert max_diff_fu == 0.0, f"FU-Shapley weights differed by {max_diff_fu}"

    # 2. Robust FU-Shapley
    _, info_rfu_h = aggregate("robust_fu_shapley", updates, meta_honest, g_target=g_target)
    _, info_rfu_l = aggregate("robust_fu_shapley", updates, meta_lying, g_target=g_target)
    w_rfu_h = [float(x) for x in info_rfu_h["weights"]]
    w_rfu_l = [float(x) for x in info_rfu_l["weights"]]
    max_diff_rfu = max(abs(a - b) for a, b in zip(w_rfu_h, w_rfu_l))
    assert max_diff_rfu == 0.0, f"Robust FU-Shapley weights differed by {max_diff_rfu}"

    # 3. FLTrust: the closest structural relative. Also server-referenced, so it
    # is expected to be metadata-immune too -- we run it precisely to show that
    # our immunity claim is not novel against FLTrust, only against the
    # fairness-aware aggregators that reintroduce the self-reported channel.
    # It scores against g_task (task-only root), not the bi-objective g_target.
    _, info_flt_h = aggregate("fltrust", updates, meta_honest, g_task=g_target)
    _, info_flt_l = aggregate("fltrust", updates, meta_lying, g_task=g_target)
    w_flt_h = [float(x) for x in info_flt_h["weights"]]
    w_flt_l = [float(x) for x in info_flt_l["weights"]]
    max_diff_flt = max(abs(a - b) for a, b in zip(w_flt_h, w_flt_l))

    # 4. BFWA Negative Control
    _, info_bfwa_h = aggregate("bfwa", updates, meta_honest)
    _, info_bfwa_l = aggregate("bfwa", updates, meta_lying)
    w_bfwa_h = [float(x) for x in info_bfwa_h["weights"]]
    w_bfwa_l = [float(x) for x in info_bfwa_l["weights"]]
    max_diff_bfwa = max(abs(a - b) for a, b in zip(w_bfwa_h, w_bfwa_l))
    assert max_diff_bfwa > 1e-4, "BFWA weights must differ under fabricated metadata"

    verdict = {
        "theorem_2_1_verified": True,
        "fu_shapley_bit_exact": bool(max_diff_fu == 0.0),
        "robust_fu_shapley_bit_exact": bool(max_diff_rfu == 0.0),
        "bfwa_vulnerable": bool(max_diff_bfwa > 1e-4),
        "max_diff_fu": max_diff_fu,
        "max_diff_robust_fu": max_diff_rfu,
        "max_diff_bfwa": max_diff_bfwa,
        "fltrust_bit_exact": bool(max_diff_flt == 0.0),
        "max_diff_fltrust": max_diff_flt,
        "weights_fltrust_honest": w_flt_h,
        "weights_fltrust_lying": w_flt_l,
        "weights_fu_honest": w_fu_h,
        "weights_fu_lying": w_fu_l,
        "weights_bfwa_honest": w_bfwa_h,
        "weights_bfwa_lying": w_bfwa_l,
    }

    manifest = build_manifest(experiment="metadata_immunity_audit", args={"out_dir": out_dir})
    payload = {"manifest": manifest, "verdict": verdict}
    out_file = os.path.join(out_dir, "metadata_immunity_verdict.json")
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 60)
    print("METADATA IMMUNITY: BIT-EXACT PASS")
    print(f"FU-Shapley max diff: {max_diff_fu}")
    print(f"Robust FU-Shapley max diff: {max_diff_rfu}")
    print(f"BFWA max diff (negative control): {max_diff_bfwa:.4f}")
    print(f"Wrote verdict to {out_file}")
    print("=" * 60)
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/fairshare")
    args = p.parse_args()
    run_metadata_immunity_audit(args.out_dir)
