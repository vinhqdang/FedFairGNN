"""Method presets: map a short method name to ExperimentConfig overrides.

A "method" is a (backbone, aggregator, training-mode) combination. This keeps
the experiment matrix readable and every reported row traceable to an exact
configuration. Non-private by default; the privacy study toggles dp_enabled.
"""
from __future__ import annotations

# name -> config overrides
METHODS = {
    # --- non-fair backbones (FedAvg) ---
    "fedavg-gcn":   dict(model="gcn", aggregator="fedavg", local_fairness=False, dp_enabled=False),
    "fedavg-gat":   dict(model="gat", aggregator="fedavg", local_fairness=False, dp_enabled=False),

    # --- fairness-aware FL aggregation baselines (GCN backbone) ---
    "fairfed":      dict(model="gcn", aggregator="fairfed", local_fairness=True, dp_enabled=False),
    "qffl":         dict(model="gcn", aggregator="qffl",    local_fairness=False, dp_enabled=False),
    "fedfb":        dict(model="gcn", aggregator="fedavg",  local_fairness=True, dp_enabled=False),

    # --- fair-GNN baselines (FedAvg) ---
    "fairgnn":      dict(model="fairgnn", aggregator="fedavg", dp_enabled=False),
    "fairsin":      dict(model="fairsin", aggregator="fedavg", local_fairness=False, dp_enabled=False),

    # --- federated fair-GNN peer ---
    "f2gnn":        dict(model="gat", aggregator="f2gnn", local_fairness=True, dp_enabled=False),

    # --- privacy+fairness baseline (full-gradient DP-SGD, PUFFLE/FedFDP family) ---
    "dp-fedavg":    dict(model="gcn", aggregator="fedavg", local_fairness=True,
                         dp_enabled=True, dp_mode="gradient"),

    # --- ours (and ablations) ---
    "fedfairgnn":       dict(model="fedfairgnn", aggregator="bfwa", dp_enabled=True, dp_mode="ftgd"),
    "fedfairgnn-nodp":  dict(model="fedfairgnn", aggregator="bfwa", dp_enabled=False),
    "ours-nofser":      dict(model="gat",        aggregator="bfwa", local_fairness=True, dp_enabled=False),
    "ours-nobfwa":      dict(model="fedfairgnn", aggregator="fedavg", dp_enabled=False),
    "ours-robust":      dict(model="fedfairgnn", aggregator="robust_bfwa", dp_enabled=True, dp_mode="ftgd"),
}

# robust aggregators to sweep in the Byzantine study (backbone = ours)
ROBUST_AGGREGATORS = ["fedavg", "bfwa", "krum", "multikrum", "median", "trimmed_mean", "robust_bfwa"]

FAIR_BASELINES = ["fedavg-gcn", "fedavg-gat", "fairgnn", "fairsin", "fairfed", "qffl", "f2gnn"]


def apply_method(config, method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHODS)}")
    for k, v in METHODS[method].items():
        setattr(config, k, v)
    config.exp_name = method
    return config
