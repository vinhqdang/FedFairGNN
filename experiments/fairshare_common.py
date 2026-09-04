"""Shared helpers for the FairShare-GNN experiment harness (Phase 2).

Kept deliberately small and dependency-light so every analysis script
(exact_shapley_correlation, incentive_audit, ablation_val_size, plot_shapley,
topology_shapley_analysis) draws its trainer/gradient/metric plumbing from one
audited place. Nothing here trains heavy by default -- callers pass tiny configs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.config import ExperimentConfig
from src.data import load_dataset
from src.data.partition import edge_retention, partition_summary
from src.federated import FederatedTrainer
from src.federated.client import load_flat_state
from src.utils.metrics import all_metrics, sensitive_homophily
from experiments.methods import apply_method


# --------------------------------------------------------------------------- #
# Global-graph properties (measured BEFORE partitioning)
# --------------------------------------------------------------------------- #
# ``FederatedTrainer`` loads the dataset, optionally carves the server holdout
# and immediately partitions it; it keeps ``clients_data`` and (maybe)
# ``server_holdout``, but never a handle on the full pre-partition graph. Every
# runner that wanted a graph-level property therefore guarded it behind
# ``hasattr(trainer, "global_data")`` -- an attribute that does not exist, so
# the guard never fired and the "measurement" was a hardcoded 0.0 in every row
# ever written. These helpers reload the same graph the trainer loaded (same
# name/root/seed => same object) and measure it directly.
@lru_cache(maxsize=8)
def _full_graph_cached(dataset: str, root: str, seed: int):
    return load_dataset(dataset, root=root, seed=seed)


def full_graph(cfg: ExperimentConfig):
    """The unpartitioned graph the trainer was built from.

    Cached per (dataset, root, seed); treat the returned ``Data`` as read-only.
    """
    return _full_graph_cached(cfg.dataset, cfg.data_root, cfg.seed)


def global_sensitive_homophily(cfg: ExperimentConfig) -> float:
    """h_s = P[s_u == s_v | (u,v) in E] on the FULL graph for this config.

    Measured pre-partition on purpose: h_s is a property of the dataset, not of
    the federation, and the per-client induced subgraphs drop every cross-client
    edge (see ``edge_retention``), so a client-side average would silently
    report the homophily of whatever survived the split.
    """
    d = full_graph(cfg)
    return float(sensitive_homophily(d.edge_index, d.sensitive_attr))


def partition_edge_retention(trainer) -> Dict[str, float]:
    """Edge budget of a built trainer's partition, relative to the full graph.

    Returns ``edge_retention`` (share of the ORIGINAL graph's edges still
    visible to some client), ``edge_retention_post_holdout`` (the same ratio
    against the graph that actually reached ``partition_graph``, i.e. after the
    server holdout was carved out), ``expected_retention_iid`` (``sum_k p_k^2``,
    what a topology-blind split of the same node shares would keep) and the raw
    counts. The two retention figures differ only when
    ``fu_val_source='server_holdout'``.
    """
    d = full_graph(trainer.cfg)
    clients = trainer.clients_data
    total = int(d.edge_index.shape[1])
    kept = sum(int(c.edge_index.shape[1]) for c in clients)
    holdout_edges = int(trainer.server_holdout.edge_index.shape[1]) \
        if getattr(trainer, "server_holdout", None) is not None else 0
    summary = partition_summary(d, clients)
    post = int(total - holdout_edges)
    return {
        "edge_retention": summary["edge_retention"],
        "edge_retention_post_holdout": float(kept) / float(post) if post > 0 else 0.0,
        "expected_retention_iid": summary["expected_retention_iid"],
        "original_edges": total,
        "client_edges": kept,
        "server_holdout_edges": holdout_edges,
        "num_clients": len(clients),
    }


# --------------------------------------------------------------------------- #
# Trainer construction
# --------------------------------------------------------------------------- #
def make_trainer(dataset="german", seed=0, num_clients=5, rounds=20,
                 method="fairshare", alpha=None, dirichlet_alpha=0.3,
                 **overrides) -> FederatedTrainer:
    cfg = ExperimentConfig(dataset=dataset, seed=seed, num_clients=num_clients,
                           rounds=rounds, dirichlet_alpha=dirichlet_alpha,
                           dp_enabled=False)
    apply_method(cfg, method)
    if alpha is not None:
        cfg.fu_alpha = alpha
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return FederatedTrainer(cfg)


def warm_rounds(trainer: FederatedTrainer, n: int) -> None:
    """Advance the global model by ``n`` real rounds (side effects on trainer)."""
    for t in range(n):
        trainer._round(t)


def client_pseudo_grads(trainer: FederatedTrainer) -> List[torch.Tensor]:
    """Retrain each client at the current global model and return their
    pseudo-gradients g_k = theta_global - theta_local_k (no poisoning). Mirrors
    ``trainer._round`` exactly up to the aggregation step."""
    grads = []
    for c in trainer.clients:
        c.set_flat(trainer.global_flat)
        c.train()
        grads.append(trainer.global_flat - c.get_flat())
    return grads


# --------------------------------------------------------------------------- #
# Pooled-validation value function (for exact Shapley / metrics at a weight set)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pooled_val_metrics(trainer: FederatedTrainer, flat_weights: torch.Tensor
                       ) -> Dict[str, float]:
    """Evaluate a candidate global model (flat state) on the pooled client
    validation nodes."""
    load_flat_state(trainer.ref_model, flat_weights.to(trainer.device))
    trainer.ref_model.eval()
    ys, ps, ss = [], [], []
    for d in trainer.clients_data:
        d = d.to(trainer.device)
        m = d.val_mask
        if int(m.sum()) == 0:
            continue
        ps.append(trainer.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu())
        ys.append(d.y[m].cpu()); ss.append(d.sensitive_attr[m].cpu())
    if not ys:
        return {}
    return all_metrics(torch.cat(ys), torch.cat(ps), torch.cat(ss))


@torch.no_grad()
def pooled_val_loss(trainer, flat_weights: torch.Tensor, alpha: float) -> float:
    """Composite validation LOSS of a candidate global model on the pooled client
    validation nodes:  L = weighted_BCE + alpha * soft_DPD.

    This is exactly the objective the clients descend and the server's
    ``g_target`` differentiates, so it is the value function that is *consistent*
    with the gradient score phi (plan finding B1/D9)."""
    from src.federated.client import _weighted_bce, _soft_dpd
    load_flat_state(trainer.ref_model, flat_weights.to(trainer.device))
    trainer.ref_model.eval()
    ps, ys, ss = [], [], []
    for d in trainer.clients_data:
        d = d.to(trainer.device)
        m = d.val_mask
        if int(m.sum()) == 0:
            continue
        ps.append(trainer.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu())
        ys.append(d.y[m].cpu()); ss.append(d.sensitive_attr[m].cpu())
    if not ys:
        return float("nan")
    p = torch.cat(ps); y = torch.cat(ys).float(); s = torch.cat(ss)
    return float(_weighted_bce(p, y) + alpha * _soft_dpd(p, s))


def value_of_coalition(trainer, grads, S, alpha: float, game: str = "loss") -> float:
    """Cooperative-game value of coalition ``S`` (FedAvg over its members),
    evaluated on the pooled validation set. Empty coalition = current global.

    game="loss"  (DEFAULT, plan B1/D9):  V(S) = -(weighted_BCE + alpha*soft_DPD)
        The *loss-game*. phi_k = <g_k, g_target> is a first-order expansion of
        exactly this quantity, so a Shapley/gradient comparison is only
        meaningful under this game.
    game="auc"   (LEGACY, Phase-3 runs):  V(S) = AUC - alpha*DPD
        Kept only to reproduce the original Phase-3 numbers for continuity.
        AUC is not differentiable and AUC != -BCE, so correlating phi against
        Shapley values of this game measures two different objectives.
    """
    if len(S) == 0:
        theta = trainer.global_flat
    else:
        g = torch.stack([grads[k] for k in S]).mean(0)
        theta = trainer.global_flat - g
    if game == "loss":
        v = pooled_val_loss(trainer, theta, alpha)
        return 0.0 if v != v else -v          # NaN-safe; negate: higher = better
    m = pooled_val_metrics(trainer, theta)
    if not m:
        return 0.0
    return float(m["auc"] - alpha * m["dpd"])


# --------------------------------------------------------------------------- #
# Topology descriptors (FS-WI-7)
# --------------------------------------------------------------------------- #
def topology_metrics(data) -> Dict[str, float]:
    """Per-client structural descriptors relevant to graph fairness."""
    ei = data.edge_index
    n = int(data.num_nodes)
    y = data.y; s = data.sensitive_attr
    if ei.shape[1] == 0:
        return {"homophily": float("nan"), "avg_degree": 0.0, "boundary_ratio": float("nan")}
    i, j = ei[0], ei[1]
    same_label = (y[i] == y[j]).float().mean().item()          # edge label homophily
    cross_group = (s[i] != s[j]).float().mean().item()         # sensitive boundary edges
    avg_degree = ei.shape[1] / max(1, n)
    return {"homophily": round(same_label, 4),
            "avg_degree": round(avg_degree, 3),
            "boundary_ratio": round(cross_group, 4)}


# --------------------------------------------------------------------------- #
# Statistics: Wilcoxon signed-rank + Holm-Bonferroni (report.py hook)
# --------------------------------------------------------------------------- #
def wilcoxon_holm(comparisons: Dict[str, Tuple[List[float], List[float]]],
                  alpha: float = 0.05) -> List[dict]:
    """Paired Wilcoxon signed-rank per comparison, with Holm-Bonferroni
    correction across the family.

    Args:
        comparisons: {name: (ours_per_seed, baseline_per_seed)}.
    Returns rows sorted by raw p, each with corrected p and significance flag.
    """
    from scipy.stats import wilcoxon
    rows = []
    for name, (a, b) in comparisons.items():
        a = np.asarray(a, float); b = np.asarray(b, float)
        diff = a - b
        if np.allclose(diff, 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(a, b).pvalue)
            except ValueError:
                p = 1.0
        rows.append({"comparison": name, "mean_ours": float(a.mean()),
                     "mean_base": float(b.mean()), "mean_diff": float(diff.mean()),
                     "p_raw": p, "n": len(a)})
    # Holm-Bonferroni
    rows.sort(key=lambda r: r["p_raw"])
    M = len(rows)
    for rank, r in enumerate(rows):
        r["p_holm"] = min(1.0, r["p_raw"] * (M - rank))
    # enforce monotonicity of Holm-adjusted p-values
    for k in range(1, M):
        rows[k]["p_holm"] = max(rows[k]["p_holm"], rows[k - 1]["p_holm"])
    for r in rows:
        r["significant"] = r["p_holm"] < alpha
    return rows


def pearson_spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])


if __name__ == "__main__":
    # tiny self-test of the statistics (no training)
    comps = {"ours_vs_A": ([0.8, 0.82, 0.79, 0.81, 0.83], [0.7, 0.71, 0.69, 0.72, 0.70]),
             "ours_vs_B": ([0.8, 0.82, 0.79, 0.81, 0.83], [0.79, 0.83, 0.78, 0.80, 0.82])}
    for r in wilcoxon_holm(comps):
        print(f"{r['comparison']:12s} diff={r['mean_diff']:+.3f} "
              f"p_raw={r['p_raw']:.4f} p_holm={r['p_holm']:.4f} sig={r['significant']}")
    print("pearson/spearman:", pearson_spearman([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8]))
