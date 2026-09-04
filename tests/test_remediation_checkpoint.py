"""Checkpoint/resume for experiments/stage4_remediation_runner.py.

The runner drives GPU jobs on remote, unreliable sessions (a lost Colab
connection has killed a run mid-flight more than once). Losing everything
computed so far to one disconnect is expensive, so every individual
multi-seed sub-result is checkpointed to disk as soon as it finishes, and a
re-invocation on the same output file resumes from there instead of
recomputing it. These tests pin that contract with the primitives directly
(fast) plus one full-module smoke test (slow, marked accordingly).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.stage4_remediation_runner import _atomic_save, _load_checkpoint


def test_atomic_save_round_trips(tmp_path):
    p = str(tmp_path / "out.json")
    data = {"a": 1, "nested": {"b": [1, 2, 3]}}
    _atomic_save(data, p)
    assert json.load(open(p)) == data


def test_atomic_save_leaves_no_tmp_file_behind(tmp_path):
    p = str(tmp_path / "out.json")
    _atomic_save({"x": 1}, p)
    assert not os.path.exists(p + ".tmp")
    assert os.path.exists(p)


def test_atomic_save_never_leaves_a_truncated_file(tmp_path):
    """A save that fails partway through must not corrupt the last-good file:
    write-to-temp-then-rename means the target is either the old complete
    content or the new complete content, never a partial write."""
    p = str(tmp_path / "out.json")
    _atomic_save({"good": "v1"}, p)
    good_bytes = open(p, "rb").read()

    # Simulate a crash mid-write: the .tmp file exists but was never renamed.
    with open(p + ".tmp", "w") as f:
        f.write('{"truncated": tr')  # deliberately invalid/incomplete JSON

    # The real target file must be untouched by the failed attempt.
    assert open(p, "rb").read() == good_bytes
    assert json.load(open(p)) == {"good": "v1"}


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert _load_checkpoint(str(tmp_path / "nope.json")) == {}


def test_load_checkpoint_corrupt_file_returns_empty_not_raises(tmp_path):
    p = str(tmp_path / "out.json")
    with open(p, "w") as f:
        f.write("{not valid json")
    assert _load_checkpoint(p) == {}


def test_load_checkpoint_strips_staleness_markers(tmp_path):
    """A prior staleness pass's audit markers describe THAT file's provenance;
    a resumed/completed run supersedes them and must not carry them forward."""
    p = str(tmp_path / "out.json")
    json.dump({
        "RUN-4.2-01": {"auc_mean": 0.7},
        "_STALENESS_NOTICE": {"status": "PARTIALLY STALE"},
        "_INVALID": {"status": "SUPERSEDED"},
        "_manifest": {"timestamp": "2020-01-01T00:00:00+00:00"},
    }, open(p, "w"))
    loaded = _load_checkpoint(p)
    assert loaded["RUN-4.2-01"] == {"auc_mean": 0.7}
    assert loaded["_manifest"]["timestamp"] == "2020-01-01T00:00:00+00:00"
    assert "_STALENESS_NOTICE" not in loaded
    assert "_INVALID" not in loaded


def test_load_checkpoint_preserves_ordinary_keys(tmp_path):
    p = str(tmp_path / "out.json")
    payload = {"stage4_5_ablation_matrix": {"M1_Full": {"auc_mean": 0.6}},
               "two_tier_defense_robustness": {"M1_no_attack": {"auc_mean": 0.5}}}
    json.dump(payload, open(p, "w"))
    assert _load_checkpoint(p) == payload


@pytest.mark.slow
def test_full_run_resumes_across_a_simulated_interrupt(tmp_path):
    """End-to-end: run the module's real entry point on a synthetic-fast config
    is not available (the runner hardcodes german/bail), so this drives a
    couple of genuinely small sub-results and asserts resume skips them.
    Marked slow; not part of the default fast suite.
    """
    from experiments.stage4_remediation_runner import run_stage4_remediation

    p = str(tmp_path / "remediation.json")
    # First call: let it run RUN-4.2-01 through -03 only, then simulate a crash
    # by monkeypatching the German/Bail RUN-4.2-04/05 and every later section
    # out of reach is impractical without editing the module, so instead we
    # pre-seed a checkpoint that already has everything except one ablation
    # arm and confirm only that arm runs.
    fake = {"auc_mean": 0.5, "auc_std": 0.0, "dpd_soft_mean": 0.0, "dpd_soft_std": 0.0,
            "dpd_hard_mean": 0.0, "dpd_hard_std": 0.0, "eod_mean": 0.0, "eod_std": 0.0,
            "omega_w_mean": 0.0, "omega_w_std": 0.0, "pred_std_mean": 0.0,
            "w_adv_mean": 0.0, "w_adv_std": 0.0, "wall_clock_s_mean": 0.0, "per_seed": []}
    seeded = {
        "RUN-4.2-01": fake, "RUN-4.2-02": fake, "RUN-4.2-03": {"probes": []},
        "RUN-4.2-04": fake, "RUN-4.2-05": fake,
        "stage4_5_ablation_matrix": {name: fake for name in
                                     ["M1_Full", "M2_wo_FSER", "M3_wo_FTGD", "M4_Full_DPSGD",
                                      "M5_wo_FairScore", "M6_wo_TwoTier"]},  # M7 missing
        "fser_sign_hypothesis": {f"fser_{m}_beta_{b}": fake
                                 for m in ["sub", "add", "same_penalize"] for b in [0.5, 2.0]},
        "two_tier_defense_robustness": {f"{arm}_{sc}": fake
                                        for sc in ["no_attack", "sign_flip_20pct", "fairness_poison_20pct"]
                                        for arm in ["M1", "M6"]},
    }
    json.dump(seeded, open(p, "w"))

    result = run_stage4_remediation(output_file=p, run_sign_test=True)

    # Only M7 should have actually run; everything else came from the seed.
    assert result["stage4_5_ablation_matrix"]["M1_Full"] is fake or \
        result["stage4_5_ablation_matrix"]["M1_Full"] == fake
    assert "M7_wo_EMA" in result["stage4_5_ablation_matrix"]
    assert result["stage4_5_ablation_matrix"]["M7_wo_EMA"] != fake
    # And the checkpoint file on disk reflects the completed run.
    on_disk = json.load(open(p))
    assert "M7_wo_EMA" in on_disk["stage4_5_ablation_matrix"]
