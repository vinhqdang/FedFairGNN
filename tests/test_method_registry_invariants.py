import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.methods import METHODS


ABLATION_EXPECTED = {
    "ours-nofser": {"model"},            # strictly only differs by model (gat vs trustfedgnn)
    "fedfairgnn-nodp": {"dp_enabled"},   # strictly only differs by dp_enabled
}

BASELINE_MUST_NOT_USE_OUR_BACKBONE = {
    "cgsv", "fedavg-gcn", "fairfed", "fairgfl", "fedgraphfair", "fairgnn", "fairsin"
}


def test_ablation_arms_isolate_one_factor():
    """Verify that every ablation arm isolates EXACTLY the single knob in its name."""
    ours = METHODS["fedfairgnn"]
    for name, expected in ABLATION_EXPECTED.items():
        assert name in METHODS, f"Missing ablation arm {name} in METHODS"
        diff = {k for k in set(ours) | set(METHODS[name])
                if ours.get(k) != METHODS[name].get(k)}
        assert diff == expected, f"Ablation arm '{name}' diverges by {diff}, expected strictly {expected}"


def test_baselines_do_not_borrow_our_backbone():
    """Baseline SOTA models must NOT borrow the trustfedgnn backbone (to avoid misattribution)."""
    for name in BASELINE_MUST_NOT_USE_OUR_BACKBONE:
        assert name in METHODS, f"Missing baseline {name} in METHODS"
        assert METHODS[name].get("model") != "trustfedgnn", \
            f"Baseline '{name}' is mistakenly using proprietary backbone 'trustfedgnn' -> ablation mislabeled as baseline!"


if __name__ == "__main__":
    test_ablation_arms_isolate_one_factor()
    print("[*] test_ablation_arms_isolate_one_factor: PASS")
    test_baselines_do_not_borrow_our_backbone()
    print("[*] test_baselines_do_not_borrow_our_backbone: PASS")
    print("\n[ALL INVARIANT TESTS PASSED 100%]")
