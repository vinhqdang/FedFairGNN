"""Model factory."""
from __future__ import annotations

from .gnn import FedFairGNN, GCN, GAT, FairGNN, FairSIN, FSERLayer

_REGISTRY = {
    "fedfairgnn": FedFairGNN,
    "gcn": GCN,
    "gat": GAT,
    "fairgnn": FairGNN,
    "fairsin": FairSIN,
}


def build_model(name: str, in_channels: int, config=None):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    kwargs = dict(in_channels=in_channels, out_channels=1)
    if config is not None:
        kwargs.update(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["build_model", "FedFairGNN", "GCN", "GAT", "FairGNN", "FairSIN", "FSERLayer"]
