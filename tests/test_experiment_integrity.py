"""Guards against experiments that appear to measure something and do not.

Each test here corresponds to a defect that shipped into committed results:

  * a Byzantine sweep whose two arms were distinguished by config flags that no
    code reads, so both arms were the same run and the hypothesis was scored on
    floating-point noise;
  * `w_adv = 0.000` reported as an aggregator's "immunity" when it was a missing
    dictionary key falling through to a hardcoded default;
  * `Omega_w = 0.000` reported as perfect weight stability for rules that expose
    no weight vector at all;
  * summary lines that could not distinguish two incompatible protocols because
    the fields separating them were never logged.
"""
from __future__ import annotations

import ast
import dataclasses
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.methods import METHODS
from src.config import ExperimentConfig
from src.federated.aggregation import aggregate
from src.utils.metrics import weight_oscillation

REPO = pathlib.Path(__file__).resolve().parent.parent
FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)}


# --------------------------------------------------------------------------- #
# 1. No knob that nothing reads
# --------------------------------------------------------------------------- #
def test_config_rejects_undeclared_fields():
    """Assigning a non-field must raise, not silently create a dead attribute."""
    cfg = ExperimentConfig()
    cfg.rounds = 7                      # a real field still works
    assert cfg.rounds == 7
    with pytest.raises(AttributeError, match="no field"):
        cfg.fu_cosine_filter = True     # the exact flag that produced a null experiment
    with pytest.raises(AttributeError):
        cfg.definitely_not_a_field = 1


def test_method_overrides_are_all_real_fields():
    """Every METHODS override key must be a field, or that arm silently no-ops."""
    for name, overrides in METHODS.items():
        unknown = set(overrides) - FIELDS
        assert not unknown, f"method '{name}' sets non-fields {sorted(unknown)}"


def test_no_experiment_assigns_a_phantom_config_field():
    """Static scan: `cfg.<x> = ...` in experiments/ must name a declared field.

    __setattr__ catches this at runtime, but only if the line is reached. This
    catches it without running a multi-hour sweep.
    """
    offenders = []
    for path in sorted((REPO / "experiments").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in ("cfg", "config", "base_cfg")
                        and target.attr not in FIELDS):
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno} "
                        f"{target.value.id}.{target.attr}")
    assert not offenders, "assignments to non-fields of ExperimentConfig:\n  " + \
        "\n  ".join(offenders)


# --------------------------------------------------------------------------- #
# 2. Unmeasured must not read as measured-zero
# --------------------------------------------------------------------------- #
def test_weightless_aggregators_report_no_weights():
    """median/trimmed_mean expose no weight vector -- pin that, so downstream
    code must handle it rather than defaulting the absence to 0.0."""
    ups = [torch.randn(16) for _ in range(5)]
    meta = [{"n": 100, "perf": 0.8, "dpd": 0.05, "loss": 0.2} for _ in range(5)]
    for method in ("median", "trimmed_mean"):
        _, info = aggregate(method, ups, meta, tau=0.05)
        assert info.get("weights") is None, \
            f"{method} now reports weights; update the NaN handling downstream"
    for method in ("fedavg", "bfwa", "krum"):
        _, info = aggregate(method, ups, meta, tau=0.05)
        assert info.get("weights") is not None, f"{method} must report weights"


def test_weight_oscillation_is_nan_when_unmeasurable():
    """Omega_w must be NaN, not 0.0, when there is nothing to measure.

    0.0 is the *best possible* score, so defaulting to it silently awards perfect
    stability to any rule whose weights were never observed.
    """
    assert np.isnan(weight_oscillation([]))
    assert np.isnan(weight_oscillation([None, None]))
    assert np.isnan(weight_oscillation([[0.5, 0.5]]))          # single round
    # a genuine measurement still comes through
    assert weight_oscillation([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(2.0)
    assert weight_oscillation([[0.5, 0.5], [0.5, 0.5]]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. Protocol provenance must be recoverable from a summary line
# --------------------------------------------------------------------------- #
def test_summary_logs_the_fields_that_distinguish_protocols():
    """A summary row must pin down which protocol produced it."""
    import inspect

    from src.utils import logging_utils

    src = inspect.getsource(logging_utils.ResultLogger.save)
    required = ["dirichlet_alpha", "partition", "rounds", "local_epochs",
                "fu_val_source", "fu_score", "fser_mode", "dp_mode",
                "fairness_budget", "fw_iterations", "krum_f", "sampling"]
    missing = [k for k in required if f'"{k}"' not in src]
    assert not missing, f"summary line omits protocol fields: {missing}"
