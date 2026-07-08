"""Partition a global graph into per-client subgraphs for federated learning.

Three regimes:
    uniform    : nodes split uniformly at random (IID baseline).
    dirichlet  : label- (or sensitive-) skewed non-IID split via a symmetric
                 Dirichlet(alpha) distribution -- the standard FL heterogeneity
                 model (Hsu et al., 2019). Smaller alpha => more skew.
    community  : Louvain/greedy-modularity communities assigned to clients,
                 producing topology-correlated heterogeneity.

Each client receives an *induced subgraph* with relabelled edges and inherits
the global train/val/test masks for its nodes. This mirrors the cross-silo FL
setting where each institution holds a disjoint portion of the entity graph.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _induced(data: Data, node_idx: torch.Tensor) -> Data:
    ei, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True,
                     num_nodes=data.num_nodes)
    sub = Data(
        x=data.x[node_idx],
        edge_index=ei,
        y=data.y[node_idx],
        sensitive_attr=data.sensitive_attr[node_idx],
        train_mask=data.train_mask[node_idx],
        val_mask=data.val_mask[node_idx],
        test_mask=data.test_mask[node_idx],
    )
    if hasattr(data, "meta"):
        sub.meta = data.meta
    return sub


def _uniform(n: int, k: int, gen) -> List[torch.Tensor]:
    perm = torch.randperm(n, generator=gen)
    return [perm[i::k] for i in range(k)]


def _dirichlet(values: np.ndarray, k: int, alpha: float, gen,
               floor: float = 0.05) -> List[torch.Tensor]:
    """Assign nodes to clients so each client's class mix ~ Dirichlet(alpha).

    A small per-client proportion floor guarantees every client receives some
    of every class (avoiding degenerate single-class silos that make local AUC
    undefined and destabilise metric-based aggregation), while preserving the
    Dirichlet heterogeneity that FL papers rely on.
    """
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item()))
    client_idx: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(values):
        idx = np.where(values == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(k))
        props = (1 - floor * k) * props + floor          # mix in a uniform floor
        props = props / props.sum()
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(idx, cuts)):
            client_idx[c].extend(part.tolist())
    return [torch.tensor(sorted(c), dtype=torch.long) for c in client_idx]


def _community(data: Data, k: int, gen) -> List[torch.Tensor]:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(data.edge_index.t().tolist())
    try:
        comms = list(nx.community.greedy_modularity_communities(g))
    except Exception:
        return _uniform(data.num_nodes, k, gen)
    # round-robin communities (largest first) into k buckets
    buckets: List[List[int]] = [[] for _ in range(k)]
    for i, com in enumerate(sorted(comms, key=len, reverse=True)):
        buckets[i % k].extend(com)
    return [torch.tensor(sorted(b), dtype=torch.long) for b in buckets if b]


def partition_graph(data: Data, num_clients: int, method: str = "dirichlet",
                    alpha: float = 0.5, by: str = "label", seed: int = 42) -> List[Data]:
    gen = torch.Generator().manual_seed(seed)
    if method == "uniform":
        parts = _uniform(data.num_nodes, num_clients, gen)
    elif method == "dirichlet":
        vals = (data.y if by == "label" else data.sensitive_attr).cpu().numpy()
        parts = _dirichlet(vals, num_clients, alpha, gen)
    elif method == "community":
        parts = _community(data, num_clients, gen)
    else:
        raise ValueError(f"Unknown partition method '{method}'")
    return [_induced(data, idx) for idx in parts if len(idx) > 0]


def partition_stats(clients: List[Data]) -> List[dict]:
    """Per-client heterogeneity summary (for logging / the paper's setup table)."""
    stats = []
    for i, c in enumerate(clients):
        y, s = c.y, c.sensitive_attr
        stats.append({
            "client": i,
            "nodes": int(c.num_nodes),
            "edges": int(c.edge_index.shape[1]),
            "pos_rate": round(float(y.float().mean()), 4),
            "group1_rate": round(float((s == 1).float().mean()), 4),
            "train": int(c.train_mask.sum()),
        })
    return stats
