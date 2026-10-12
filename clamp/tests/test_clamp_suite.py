"""Smoke tests for the CLAMP experiment runner and table helpers (no training)."""
import math

from experiments.methods import METHODS
from experiments.make_clamp_tables import holm
from experiments.run_clamp_suite import E1_METHODS, E3_ROWS, expand
from experiments.run_experiment import build_config, run_id_for
from src.federated.aggregation import ALL_METHODS

EXPECTED = {"e1": 560, "e2": 90, "e3": 630, "e4": 81, "e5": 160, "e6": 42}


def test_job_counts_per_group():
    for group, n in EXPECTED.items():
        assert len(expand(group)) == n, group


def test_run_ids_are_unique_within_each_group():
    for group in EXPECTED:
        ids = []
        for j in expand(group):
            cfg = build_config(j["method"], j["dataset"], j["seed"], **j["overrides"])
            ids.append(run_id_for(cfg, j["tag"]))
        assert len(ids) == len(set(ids)), f"duplicate run ids in {group}"


def test_tags_are_unique_per_method_dataset_seed():
    for group in EXPECTED:
        seen = set()
        for j in expand(group):
            key = (j["method"], j["dataset"], j["seed"], j["tag"],
                   j["overrides"].get("attack"), j["overrides"].get("num_byzantine"))
            assert key not in seen, (group, key)
            seen.add(key)


def test_every_named_method_and_aggregator_exists():
    for m in E1_METHODS:
        assert m in METHODS, m
    for _, extra, _ in E3_ROWS:
        agg = extra.get("aggregator", METHODS["fedavg-gat"]["aggregator"])
        assert agg in ALL_METHODS, agg


def test_clamp_period_defaults_follow_dataset():
    for j in expand("e1"):
        if j["method"].startswith("clamp"):
            want = 10 if j["dataset"] in ("german", "nba", "facebook") else 1
            assert j["overrides"]["r2_stat_period"] == want


def test_common_settings_applied_to_every_job():
    for group in EXPECTED:
        for j in expand(group):
            ov = j["overrides"]
            assert ov["rounds"] == 100 and ov["dirichlet_alpha"] == 0.3
            assert ov["partition"] == "dirichlet"
            if group != "e5":  # e5 sweeps dp_epsilon on purpose
                assert ov["dp_epsilon"] == 8.0


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> running max 0.09, 0.2*1=0.2
    assert adj[0] == 0.04
    assert math.isclose(adj[2], 0.09) and math.isclose(adj[1], 0.09)
    assert math.isclose(adj[3], 0.2)
    assert holm([float("nan"), 0.5])[1] == 0.5
