import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pytest
except ImportError:
    pytest = None

from src.config import ExperimentConfig
from experiments.stage4_remediation_runner import ABLATION_ARMS


def test_canonical_matches_protocol():
    """Verify that canonical configuration matches the preregistered research protocol."""
    c = ExperimentConfig.canonical()
    assert (c.dirichlet_alpha, c.rounds, c.num_clients) == (0.3, 20, 5)
    assert c.fser_mode == "sub"
    assert c.fu_val_source == "server_holdout"
    assert c.model == "trustfedgnn"
    assert c.dp_enabled is True
    assert c.dp_mode == "ftgd"
    assert c.fu_alpha == 0.1
    assert c.fu_ema_beta == 0.9
    assert c.fairness_weight == 1.0


def test_ablation_arms_differ_from_M1_by_exactly_one_field():
    """Verify each ablation arm differs from M1 by EXACTLY the field(s) claimed in its name."""
    m1 = ABLATION_ARMS["M1_Full"](42).to_dict()
    expected = {
        "M2_wo_FSER": {"model"},
        "M3_wo_FTGD": {"dp_enabled"},
        "M4_Full_DPSGD": {"dp_mode"},
        "M5_wo_FairScore": {"fu_alpha"},
        "M6_wo_TwoTier": {"fu_val_source", "fu_score"},
        "M7_wo_EMA": {"fu_ema_beta"},
    }
    for name, exp_fields in expected.items():
        arm_cfg = ABLATION_ARMS[name](42).to_dict()
        diff = {k for k, v in arm_cfg.items() if m1.get(k) != v} - {"exp_name"}
        assert diff == exp_fields, f"{name} differs by {diff}, expected {exp_fields}"


if __name__ == "__main__":
    test_canonical_matches_protocol()
    test_ablation_arms_differ_from_M1_by_exactly_one_field()
    print("✅ All canonical invariant tests PASSED!")
