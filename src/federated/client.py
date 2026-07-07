"""Federated client: local training and evaluation.

Handles three training modes behind one interface:
  * plain          -- weighted BCE task loss (GCN/GAT).
  * fair + FTGD    -- task + soft-DPD fairness loss; the gradient is split into
                      task/fairness subspaces and Gaussian DP noise is added
                      *only* to the (clipped) fairness component (FedFairGNN).
  * adversarial    -- FairGNN minimax debiasing.

A Byzantine client optionally flips its local labels (data poisoning); update-
level attacks are applied server-side (see attacks.py).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from ..models import build_model
from ..utils.metrics import all_metrics
from ..trust.privacy import calibrate_noise_multiplier
from .attacks import flip_labels


def flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.flatten() for v in state.values()])


def load_flat_state(model, flat: torch.Tensor) -> None:
    sd, idx = model.state_dict(), 0
    for k, v in sd.items():
        n = v.numel()
        sd[k] = flat[idx:idx + n].view_as(v).to(v.dtype)
        idx += n
    model.load_state_dict(sd)


def _weighted_bce(pred, y):
    n = len(y)
    npos = y.sum().clamp(min=1.0)
    w_pos = n / (2.0 * npos)
    w_neg = n / (2.0 * (n - npos).clamp(min=1.0))
    return -(w_pos * y * torch.log(pred + 1e-7)
             + w_neg * (1 - y) * torch.log(1 - pred + 1e-7)).mean()


def _soft_dpd(pred, s):
    m0 = pred[s == 0].mean() if (s == 0).any() else pred.new_tensor(0.0)
    m1 = pred[s == 1].mean() if (s == 1).any() else pred.new_tensor(0.0)
    return torch.abs(m0 - m1)


class Client:
    def __init__(self, client_id: int, data, config, device="cpu", byzantine=False):
        self.id = client_id
        self.data = data.to(device)
        self.cfg = config
        self.device = device
        self.byzantine = byzantine
        self.model = build_model(config.model, data.x.shape[1], config).to(device)
        self.is_fair = config.model == "fedfairgnn"
        self.is_adv = config.model == "fairgnn"

        # DP noise scale (FTGD): calibrate multiplier for target epsilon over
        # all local privatised steps, so cumulative accounting hits dp_epsilon.
        self.noise_multiplier = 0.0
        self.dp_sigma = 0.0
        if self.is_fair and config.dp_enabled:
            total_steps = max(1, config.rounds * config.local_epochs)
            self.noise_multiplier = calibrate_noise_multiplier(
                config.dp_epsilon, total_steps, config.dp_delta)
            self.dp_sigma = self.noise_multiplier * config.dp_clip

        # data-poisoning: flip labels once, on the training portion
        self._y = self.data.y.clone()
        if self.byzantine and config.attack == "label_flip":
            m = self.data.train_mask
            self._y[m] = flip_labels(self._y[m])

    # ----- weight I/O -----
    def get_flat(self) -> torch.Tensor:
        return flatten_state(self.model.state_dict()).cpu()

    def set_flat(self, flat: torch.Tensor) -> None:
        load_flat_state(self.model, flat.to(self.device))

    # ----- training -----
    def train(self) -> None:
        cfg = self.cfg
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.local_lr,
                                weight_decay=cfg.weight_decay)
        adv_opt = (torch.optim.Adam(self.model.adversary.parameters(), lr=cfg.local_lr)
                   if self.is_adv else None)
        self.model.train()
        m = self.data.train_mask
        x, ei, s, y = self.data.x, self.data.edge_index, self.data.sensitive_attr, self._y

        for _ in range(cfg.local_epochs):
            if self.is_adv:
                self._adv_step(opt, adv_opt, x, ei, s, y, m)
            elif self.is_fair:
                self._ftgd_step(opt, x, ei, s, y, m)
            else:
                opt.zero_grad()
                pred = self.model(x, ei, s)[m]
                _weighted_bce(pred, y[m].float()).backward()
                opt.step()

    def _ftgd_step(self, opt, x, ei, s, y, m):
        """FTGD: split the objective into an S-independent task pathway
        (released in the clear) and an S-dependent fairness pathway that is
        privatised at the level of its *sufficient statistics* -- the two group
        means (mu0, mu1) of the soft demographic-parity term.

        Privatising these two bounded-sensitivity scalars (Gaussian mechanism,
        post-processing immunity) yields (eps, delta)-DP w.r.t. the sensitive
        attribute at negligible utility cost -- avoiding the curse of
        dimensionality of noising the full |theta|-dimensional gradient.
        """
        opt.zero_grad()
        pred = self.model(x, ei, s)[m]
        s_m = s[m]
        task = _weighted_bce(pred, y[m].float())

        mask0, mask1 = (s_m == 0), (s_m == 1)
        n0 = int(mask0.sum()); n1 = int(mask1.sum())
        mu0 = pred[mask0].mean() if n0 > 0 else pred.sum() * 0.0
        mu1 = pred[mask1].mean() if n1 > 0 else pred.sum() * 0.0
        if self.dp_sigma > 0 and n0 > 0 and n1 > 0:
            # L2 sensitivity of (mu0, mu1) to flipping one node's group.
            sens = (1.0 / n0 ** 2 + 1.0 / n1 ** 2) ** 0.5
            sigma = self.noise_multiplier * sens
            mu0 = mu0 + torch.randn(()) * sigma          # additive constant ->
            mu1 = mu1 + torch.randn(()) * sigma          # gradient still flows
        fair = torch.abs(mu0 - mu1)
        total = task + self.cfg.fairness_weight * fair

        # gradient surgery: orthogonalise the task gradient against the fairness
        # gradient to reduce task/fairness conflict, then recombine.
        total.backward(retain_graph=True)
        g_total = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                             for p in self.model.parameters()])
        opt.zero_grad()
        (self.cfg.fairness_weight * fair).backward()
        g_fair = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                            for p in self.model.parameters()])
        norm_sq = torch.dot(g_fair, g_fair) + 1e-12
        g_task = g_total - (torch.dot(g_total, g_fair) / norm_sq) * g_fair
        g_final = g_task + g_fair

        idx = 0
        for p in self.model.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g_final[idx:idx + n].view_as(p))
            idx += n
        opt.step()
        self.model.clamp_beta()

    def _adv_step(self, opt, adv_opt, x, ei, s, y, m):
        # 1) train adversary to predict S from (detached) embedding
        adv_opt.zero_grad()
        self.model(x, ei, s)
        adv = self.model.adv_loss(s, m)
        adv.backward()
        adv_opt.step()
        # 2) train encoder+classifier: task loss - lambda * adversary loss
        opt.zero_grad()
        pred = self.model(x, ei, s)[m]
        task = _weighted_bce(pred, y[m].float())
        adv2 = self.model.adv_loss(s, m)
        (task - self.cfg.fairness_weight * adv2).backward()
        opt.step()

    # ----- evaluation -----
    @torch.no_grad()
    def evaluate(self, split="val") -> Dict[str, float]:
        self.model.eval()
        mask = getattr(self.data, f"{split}_mask")
        pred = self.model(self.data.x, self.data.edge_index, self.data.sensitive_attr)[mask]
        out = all_metrics(self.data.y[mask], pred, self.data.sensitive_attr[mask])
        out["n"] = int(mask.sum())
        return out

    def meta(self) -> Dict[str, float]:
        v = self.evaluate("val")
        return {"n": int(self.data.train_mask.sum()), "perf": v["auc"],
                "dpd": v["dpd"], "eod": v["eod"], "eo": v["eo"]}
