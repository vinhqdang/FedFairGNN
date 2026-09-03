import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.methods import METHODS, ROBUST_AGGREGATORS
from src.federated.aggregation import ALL_METHODS, ROBUST_METHODS


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


def test_robustness_sweep_covers_every_defence():
    """The Byzantine study must sweep EVERY aggregator the library calls robust.

    Without this, adding a defence to ROBUST_METHODS silently omits it from the
    robustness results -- and the omitted ones were fu_shapley/robust_fu_shapley,
    i.e. the paper's own aggregation rule.
    """
    swept = set(ROBUST_AGGREGATORS)
    missing = ROBUST_METHODS - swept
    assert not missing, f"Robust aggregators never swept in the Byzantine study: {sorted(missing)}"


def test_robustness_sweep_names_are_real_aggregators():
    """Every swept name must dispatch in aggregation.aggregate()."""
    unknown = set(ROBUST_AGGREGATORS) - ALL_METHODS
    assert not unknown, f"ROBUST_AGGREGATORS contains unknown aggregators: {sorted(unknown)}"
    assert len(ROBUST_AGGREGATORS) == len(set(ROBUST_AGGREGATORS)), \
        "ROBUST_AGGREGATORS contains duplicates -> duplicated runs in the sweep"


def test_proposed_aggregator_is_in_the_robustness_sweep():
    """The aggregator of the headline method must appear in the Byzantine study."""
    ours = METHODS["fedfairgnn"]["aggregator"]
    assert ours in ROBUST_AGGREGATORS, \
        f"Headline aggregator '{ours}' is absent from the robustness sweep"


if __name__ == "__main__":
    test_ablation_arms_isolate_one_factor()
    print("[*] test_ablation_arms_isolate_one_factor: PASS")
    test_baselines_do_not_borrow_our_backbone()
    print("[*] test_baselines_do_not_borrow_our_backbone: PASS")
    test_robustness_sweep_covers_every_defence()
    print("[*] test_robustness_sweep_covers_every_defence: PASS")
    test_robustness_sweep_names_are_real_aggregators()
    print("[*] test_robustness_sweep_names_are_real_aggregators: PASS")
    test_proposed_aggregator_is_in_the_robustness_sweep()
    print("[*] test_proposed_aggregator_is_in_the_robustness_sweep: PASS")
    print("\n[ALL INVARIANT TESTS PASSED 100%]")
