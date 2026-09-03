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

    # --- 2026 competitors ---
    # FaVGNN (Wang & Jin, Information Fusion 2026): horizontal adaptation of the
    # completion-driven adversarial fusion (hetero-feature fusion + adversary).
    "favgnn":       dict(model="favgnn", aggregator="fedavg", dp_enabled=False),
    # FDP-Fair (Xue & Yu, arXiv 2603.24392, 2026): DP training + demographic-
    # parity post-processing (group-offset calibration).
    "fdp-fair":     dict(model="gcn", aggregator="fedavg", local_fairness=False,
                         dp_enabled=True, dp_mode="gradient", postproc_fair=True),

    # --- 2025 competitors (uploaded PDFs, reimplemented; see docs/BASELINES_AND_SOURCES.md) ---
    # FairGFL (Zhou et al., IEEE TPDS 2026; arXiv 2512.23235): overlap-aware
    # aggregation reweighting.
    "fairgfl":      dict(model="gcn", aggregator="fairgfl", local_fairness=False, dp_enabled=False),
    # FedGraph-Fair (Khan, Information Sciences 2026): minimax/DRO dual-ascent
    # reweighting toward high-loss clients (personalisation/graph-mixing dropped).
    "fedgraphfair": dict(model="gcn", aggregator="fedgraphfair", local_fairness=False, dp_enabled=False),
    # PUFFLE (Corbucci et al., ECML-PKDD 2024): auto-tuned fairness weight via
    # a momentum controller tracking a target disparity, + DP-SGD.
    "puffle":       dict(model="gcn", aggregator="fedavg", local_fairness=True,
                         dp_enabled=True, dp_mode="puffle"),
    # FedFACT (Zhang et al., NeurIPS 2025): two-level global+local group-
    # fairness post-processing (closed-form mean-matching special case).
    "fedfact":      dict(model="gcn", aggregator="fedavg", local_fairness=False,
                         dp_enabled=False, fedfact_post=True),
    # PoPETs'25 (Bendoukha et al.): FHE-friendly FairFed (statistical core only;
    # the threshold-CKKS cryptography is not reimplemented).
    "popets-fairfed": dict(model="gcn", aggregator="popets_fairfed", local_fairness=False, dp_enabled=False),

    # --- privacy+fairness baseline (full-gradient DP-SGD) ---
    "dp-fedavg":    dict(model="gcn", aggregator="fedavg", local_fairness=True,
                         dp_enabled=True, dp_mode="gradient"),

    # --- ours (and ablations) ---
    # Method dict KEYS below ("fedfairgnn", "fedfairgnn-nodp") are preserved
    # as-is even though the paper now calls the method TrustFedGNN: exp_name
    # (= this key) is baked into every already-logged run_id in results/, so
    # renaming the key would orphan hundreds of hours of completed runs
    # (including the multi-hour ogbn-products study) rather than just
    # relabelling them. Only the internal `model=` dispatch string and every
    # user-facing display name (PRETTY dict, manuscript) were renamed.
    "fedfairgnn":       dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=True, dp_mode="ftgd"),
    "fedfairgnn-nodp":  dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=False, dp_mode="ftgd"),
    "ours-nofser":      dict(model="gat",         aggregator="fu_shapley", dp_enabled=True, dp_mode="ftgd"),
    "ours-nobfwa":      dict(model="trustfedgnn", aggregator="fedavg", dp_enabled=False),
    "ours-robust":      dict(model="trustfedgnn", aggregator="robust_fu_shapley", dp_enabled=True, dp_mode="ftgd"),
    # + EquFL-style server-side fairness calibration (Yu et al. 2026) stacked
    # on top of FSER+BFWA -- see docs/BASELINES_AND_SOURCES.md. New method
    # key (not "fedfairgnn-nodp") so existing cached results are untouched;
    # only kept as the reported configuration if it's a genuine improvement.
    "trustfedgnn-plus": dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=False, server_calib=True),

    # --- FairShare-GNN (R1): FU-Shapley server aggregation, no self-report ---
    # Same FSER+FTGD client backbone as TrustFedGNN; only the server aggregation
    # rule changes (metadata ignored for weights). See incentive_mechanism_
    # proposal.md / implementation_plan_and_ac_review.md.
    "fairshare":        dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=False),
    "fairshare-robust": dict(model="trustfedgnn", aggregator="robust_fu_shapley", dp_enabled=False),
    # GTG-Shapley-style utility-only ablation = FairShare with fu_alpha=0 (drops
    # the fairness target); a free baseline (B4 in recommended_related_work).
    "gtg-shapley":      dict(model="trustfedgnn", aggregator="fu_shapley", dp_enabled=False, fu_alpha=0.0),
    # CGSV (Xu et al., NeurIPS 2021): cosine-gradient SV on standard GCN backbone, no server D_val.
    "cgsv":             dict(model="gcn", aggregator="cgsv", local_fairness=False, dp_enabled=False),
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
