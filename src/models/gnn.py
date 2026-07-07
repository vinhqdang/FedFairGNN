"""Graph neural network architectures.

Unified interface: every model's ``forward(x, edge_index, sensitive_attr=None)``
returns per-node fraud/positive probabilities of shape ``[N]`` (post-sigmoid).

Models
    GCN, GAT        -- standard non-fair backbones (baselines)
    FairGNN         -- adversarial debiasing baseline (Dai & Wang, 2021)
    FedFairGNN      -- our method: FSER-GAT backbone whose edge attention is
                       fairness-reweighted and exposed for explainability;
                       dropout can stay active at inference for MC-dropout
                       uncertainty estimation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.utils import softmax


# --------------------------------------------------------------------------- #
# FSER: Fairness-Sensitive Edge Reweighting layer
# --------------------------------------------------------------------------- #
class FSERLayer(MessagePassing):
    """GAT-style attention with a learnable fairness correction.

    The attention logit for edge (i, j) is reduced by ``beta * phi_ij`` where
    ``phi_ij = 1[s_i != s_j] * relu(cos(h_i, h_j))`` penalises *cross-group*
    edges between similarly-embedded nodes -- the structural pattern most
    responsible for propagating demographic bias in homophilous graphs. The
    per-edge attention is cached in ``self.last_attention`` for explanation.
    """

    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.3):
        super().__init__(node_dim=0, aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        self.beta = Parameter(torch.tensor(0.5))  # fairness coefficient (clamped to [0,5])
        self.last_attention = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        with torch.no_grad():
            self.beta.fill_(0.5)

    def forward(self, x, edge_index, sensitive_attr):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        s = sensitive_attr.float() if sensitive_attr is not None else torch.zeros(x.size(0), device=x.device)
        out = self.propagate(edge_index, x=x, s=s)
        return out.mean(dim=1) if not self.concat else out.reshape(-1, self.heads * self.out_channels)

    def message(self, x_i, x_j, s_i, s_j, index, ptr, size_i):
        # standard GAT logit
        e = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(-1)
        e = F.leaky_relu(e, 0.2)                                    # [E, heads]
        # FSER fairness risk
        delta_s = (s_i != s_j).float().unsqueeze(-1)               # [E, 1]
        cos = F.cosine_similarity(x_i, x_j, dim=-1)                 # [E, heads]
        phi = delta_s * cos.clamp(min=0)
        e = e - self.beta * phi
        alpha = softmax(e, index, ptr, size_i)                      # [E, heads]
        self.last_attention = alpha.detach()
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class FedFairGNN(nn.Module):
    """FSER-GAT backbone with input projection, residual blocks and skip
    concatenation. Supports Monte-Carlo dropout at inference (``mc=True``)."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=1,
                 num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.bn_in = nn.BatchNorm1d(hidden_channels)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(FSERLayer(hidden_channels, hidden_channels // heads,
                                         heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.final_lin = nn.Linear(hidden_channels * (num_layers + 1), hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr=None, mc: bool = False, logits: bool = False):
        train = self.training or mc  # mc-dropout keeps dropout active at eval
        h = F.dropout(F.elu(self.bn_in(self.input_proj(x))), p=self.dropout, training=train)
        reps = [h]
        for layer, bn in zip(self.layers, self.bns):
            hn = F.dropout(F.elu(bn(layer(h, edge_index, sensitive_attr))), p=self.dropout, training=train)
            h = hn + h if h.shape == hn.shape else hn  # residual when dims match
            reps.append(h)
        h = F.elu(self.final_lin(torch.cat(reps, dim=1)))
        out = self.classifier(h).squeeze(-1)
        return out if logits else torch.sigmoid(out)

    def clamp_beta(self):
        for layer in self.layers:
            layer.beta.data.clamp_(0.0, 5.0)

    def edge_attention(self):
        """Average attention across FSER layers/heads -> per-edge weight [E]."""
        a = [l.last_attention for l in self.layers if l.last_attention is not None]
        if not a:
            return None
        return torch.stack([x.mean(dim=1) for x in a]).mean(dim=0)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=2, dropout=0.3, **_):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr=None, mc=False, logits=False):
        train = self.training or mc
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=train)
        out = self.classifier(x).squeeze(-1)
        return out if logits else torch.sigmoid(out)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=2,
                 heads=4, dropout=0.3, **_):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr=None, mc=False, logits=False):
        train = self.training or mc
        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, edge_index)), p=self.dropout, training=train)
        out = self.classifier(x).squeeze(-1)
        return out if logits else torch.sigmoid(out)


class FairGNN(nn.Module):
    """Adversarial debiasing baseline (Dai & Wang, 2021).

    A GCN encoder feeds a task classifier and an adversary that tries to
    predict the sensitive attribute from the embedding. The encoder is trained
    to fool the adversary (via ``adv_loss``), removing sensitive information.
    """

    def __init__(self, in_channels, hidden_channels=64, out_channels=1,
                 num_layers=2, dropout=0.3, **_):
        super().__init__()
        self.dropout = dropout
        self.enc = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.enc.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self._emb = None

    def encode(self, x, edge_index, train):
        for conv in self.enc:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=train)
        return x

    def forward(self, x, edge_index, sensitive_attr=None, mc=False, logits=False):
        train = self.training or mc
        self._emb = self.encode(x, edge_index, train)
        out = self.classifier(self._emb).squeeze(-1)
        return out if logits else torch.sigmoid(out)

    def adv_loss(self, sensitive_attr, mask):
        """Adversary predicts S; returned loss is used both to train the
        adversary and (negated) to debias the encoder."""
        if self._emb is None:
            return torch.tensor(0.0)
        s_logit = self.adversary(self._emb).squeeze(-1)[mask]
        return F.binary_cross_entropy_with_logits(s_logit, sensitive_attr[mask].float())
