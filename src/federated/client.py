"""Federated client: local training and evaluation.

Handles three training modes behind one interface:
  * plain          -- weighted BCE task loss (GCN/GAT).
  * fair + FTGD    -- task + soft-DPD fairness loss; the gradient is split into
                      task/fairness subspaces and Gaussian DP noise is added
                      *only* to the (clipped) fairness component (TrustFedGNN).
  * adversarial    -- FairGNN minimax debiasing.

A Byzantine client optionally flips its local labels (data poisoning); update-
level attacks are applied server-side (see attacks.py).
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..models import build_model
from ..utils.metrics import all_metrics
from ..trust.privacy import calibrate_noise_multiplier
from .attacks import flip_labels


@torch.no_grad()
def sampled_predict(model, data, mask, cfg, device):
    """Mini-batch inference over `mask` via NeighborLoader; returns pooled
    (y, pred, sensitive) on the seed nodes. For graphs too large for full-batch."""
    from ..data.sampler import SimpleNeighborLoader
    model.eval()
    loader = SimpleNeighborLoader(data, num_neighbors=cfg.num_neighbors,
                                  input_nodes=mask, batch_size=cfg.batch_size, shuffle=False)
    ys, ps, ss = [], [], []
    for b in loader:
        b = b.to(device); bs = b.batch_size
        out = model(b.x, b.edge_index, b.sensitive_attr)[:bs]
        ys.append(b.y[:bs].cpu()); ps.append(out.cpu()); ss.append(b.sensitive_attr[:bs].cpu())
    return torch.cat(ys), torch.cat(ps), torch.cat(ss)


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
        self.is_fair = config.model == "trustfedgnn"
        self.is_adv = config.model in ("fairgnn", "favgnn")
        self.local_fair = self.is_fair or config.local_fairness

        # Resolve DP mode. `dp_mode` selects the local *training algorithm*, and
        # two of those algorithms stay meaningful with the noise switched off:
        # `fedfairgnn-nodp` is exactly FTGD orthogonalisation at sigma=0 and must
        # differ from `fedfairgnn` by dp_enabled alone (see
        # tests/test_method_registry_invariants.py), and PUFFLE's lambda
        # controller is its fairness mechanism, not its privacy one. So an
        # explicitly-set mode is honoured regardless of dp_enabled; what
        # dp_enabled governs is whether the privacy *mechanism* (clip + Gaussian
        # noise) is live -- that is `self.dp_active`, and every clip/noise site
        # gates on it rather than on the mode name.
        if config.dp_mode == "auto":
            self.dp_mode = ("ftgd" if self.local_fair else "gradient") \
                if config.dp_enabled else "none"
        else:
            self.dp_mode = config.dp_mode

        # DP noise multiplier: calibrate for target epsilon over all local
        # privatised steps, so cumulative RDP accounting hits dp_epsilon.
        self.dp_active = bool(config.dp_enabled and self.dp_mode != "none")
        self.noise_multiplier = 0.0
        self.dp_sigma = 0.0
        if self.dp_active:
            total_steps = max(1, config.rounds * config.local_epochs)
            self.noise_multiplier = calibrate_noise_multiplier(
                config.dp_epsilon, total_steps, config.dp_delta)
            self.dp_sigma = self.noise_multiplier * config.dp_clip
        elif self.dp_mode in ("gradient", "puffle"):
            # These two paths ARE the privacy mechanism. Running them with
            # dp_enabled=False used to clip to dp_clip and add nothing, i.e. a
            # silently non-private variant that is neither the published method
            # nor plain SGD. Clipping is now skipped too (see _privatise_grads),
            # so the arm degrades to its noise-free algorithm -- loudly.
            warnings.warn(
                f"client {client_id}: dp_mode={self.dp_mode!r} with "
                "dp_enabled=False -- running WITHOUT clipping or noise. This "
                "configuration is not differentially private and must not be "
                "reported as such.",
                RuntimeWarning, stacklevel=2)

        # data-poisoning: flip labels once, on the training portion
        self._y = self.data.y.clone()
        if self.byzantine and config.attack == "label_flip":
            m = self.data.train_mask
            self._y[m] = flip_labels(self._y[m])

        # PUFFLE (Corbucci et al., ECML-PKDD'24): per-client momentum-controller
        # state for the auto-tuned fairness weight (see _puffle_step).
        self._puffle_lambda = 0.0
        self._puffle_velocity = 0.0

    # ----- weight I/O -----
    def get_flat(self) -> torch.Tensor:
        return flatten_state(self.model.state_dict()).cpu()

    def set_flat(self, flat: torch.Tensor) -> None:
        load_flat_state(self.model, flat.to(self.device))

    # ----- neighbor-sampling helpers (for graphs too large for full-batch) -----
    def _loader(self, mask, shuffle):
        from ..data.sampler import SimpleNeighborLoader
        d = self.data.clone()
        d.y = self._y
        return SimpleNeighborLoader(d, num_neighbors=self.cfg.num_neighbors,
                                    input_nodes=mask, batch_size=self.cfg.batch_size,
                                    shuffle=shuffle)

    def _train_sampled(self, opt, adv_opt):
        cfg = self.cfg
        loader = self._loader(self.data.train_mask, shuffle=True)
        for _ in range(cfg.local_epochs):
            for b in loader:
                b = b.to(self.device)
                bs = b.batch_size
                seed = slice(0, bs)
                y_s = b.y[seed].float(); s_s = b.sensitive_attr[seed]
                if self.is_adv:
                    adv_opt.zero_grad(); self.model(b.x, b.edge_index, b.sensitive_attr)
                    self.model.adv_loss(b.sensitive_attr, torch.arange(bs, device=self.device)).backward()
                    adv_opt.step()
                    opt.zero_grad()
                    pred = self.model(b.x, b.edge_index, b.sensitive_attr)[seed]
                    task = _weighted_bce(pred, y_s)
                    adv2 = self.model.adv_loss(b.sensitive_attr, torch.arange(bs, device=self.device))
                    (task - cfg.fairness_weight * adv2).backward(); opt.step()
                elif self.dp_mode == "ftgd":
                    self._ftgd_batch(opt, b.x, b.edge_index, b.sensitive_attr, y_s, s_s, bs)
                else:
                    opt.zero_grad()
                    pred = self.model(b.x, b.edge_index, b.sensitive_attr)[seed]
                    loss = _weighted_bce(pred, y_s)
                    if self.local_fair:
                        loss = loss + cfg.fairness_weight * _soft_dpd(pred, s_s)
                    loss.backward()
                    if self.dp_mode == "gradient":
                        self._privatise_grads()
                    opt.step()
            if self.is_fair:
                self.model.clamp_beta()

    def _ftgd_batch(self, opt, x, ei, s, y_s, s_s, bs):
        opt.zero_grad()
        pred = self.model(x, ei, s)[:bs]
        task = _weighted_bce(pred, y_s)
        n0 = int((s_s == 0).sum()); n1 = int((s_s == 1).sum())
        mu0 = pred[s_s == 0].mean() if n0 else pred.sum() * 0.0
        mu1 = pred[s_s == 1].mean() if n1 else pred.sum() * 0.0
        if self.dp_sigma > 0 and n0 and n1:
            sens = (1.0 / n0 ** 2 + 1.0 / n1 ** 2) ** 0.5
            mu0 = mu0 + torch.randn(()) * self.noise_multiplier * sens
            mu1 = mu1 + torch.randn(()) * self.noise_multiplier * sens
        (task + self.cfg.fairness_weight * torch.abs(mu0 - mu1)).backward()
        opt.step()
        if hasattr(self.model, "clamp_beta"):
            self.model.clamp_beta()

    # ----- training -----
    def train(self) -> None:
        cfg = self.cfg
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.local_lr,
                                weight_decay=cfg.weight_decay)
        adv_opt = (torch.optim.Adam(self.model.adversary.parameters(), lr=cfg.local_lr)
                   if self.is_adv else None)
        self.model.train()
        if cfg.sampling:
            self._train_sampled(opt, adv_opt)
            return
        m = self.data.train_mask
        x, ei, s, y = self.data.x, self.data.edge_index, self.data.sensitive_attr, self._y

        # Fairness-poisoning attacker (Kasyap et al., 2025): train the local
        # objective l(w) - lambda * M_fair to *amplify* the demographic-parity
        # gap while preserving accuracy, then lie about its fairness metric
        # (handled in attacks.poison_updates) to capture a fairness-aware server.
        if self.byzantine and cfg.attack == "fairness_poison":
            for _ in range(cfg.local_epochs):
                opt.zero_grad()
                pred = self.model(x, ei, s)[m]
                loss = _weighted_bce(pred, y[m].float()) \
                    - cfg.attack_intensity * _soft_dpd(pred, s[m])
                loss.backward(); opt.step()
            return

        for _ in range(cfg.local_epochs):
            if self.is_adv:
                self._adv_step(opt, adv_opt, x, ei, s, y, m)
            elif self.dp_mode == "ftgd":
                self._ftgd_step(opt, x, ei, s, y, m)
            elif self.dp_mode == "puffle":
                self._puffle_step(opt, x, ei, s, y, m)
            elif self.dp_mode == "gradient":
                self._dp_fedavg_step(opt, x, ei, s, y, m)   # DP-FedAvg baseline
            elif self.local_fair:
                self._fair_step(opt, x, ei, s, y, m)
            else:
                opt.zero_grad()
                pred = self.model(x, ei, s)[m]
                _weighted_bce(pred, y[m].float()).backward()
                opt.step()

    def _privatise_grads(self) -> None:
        """Clip the flat gradient to ``dp_clip`` and add Gaussian noise, in place.

        No-op unless the privacy mechanism is live (``dp_active``): clipping with
        zero noise is neither the private mechanism nor plain SGD, so it must
        never happen implicitly. See the dp_mode resolution in ``__init__``.
        """
        if not self.dp_active:
            return
        params = list(self.model.parameters())
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                       for p in params])
        g = g / max(1.0, float(g.norm(2) / self.cfg.dp_clip))
        g = g + torch.randn_like(g) * self.dp_sigma
        idx = 0
        for p in params:
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g[idx:idx + n].view_as(p))
            idx += n

    def _fair_step(self, opt, x, ei, s, y, m):
        """Task + soft-DPD penalty, no privacy (generic fair baseline)."""
        opt.zero_grad()
        pred = self.model(x, ei, s)[m]
        loss = _weighted_bce(pred, y[m].float()) + self.cfg.fairness_weight * _soft_dpd(pred, s[m])
        loss.backward()
        opt.step()
        if self.is_fair:
            self.model.clamp_beta()

    def _dp_fedavg_step(self, opt, x, ei, s, y, m):
        """Standard full-gradient DP-SGD (DP-FedAvg baseline): clip the entire
        gradient to C and add isotropic Gaussian noise. Contrast for FTGD --
        this noises all |theta| coordinates and typically wrecks utility."""
        opt.zero_grad()
        pred = self.model(x, ei, s)[m]
        loss = _weighted_bce(pred, y[m].float())
        if self.local_fair:
            loss = loss + self.cfg.fairness_weight * _soft_dpd(pred, s[m])
        loss.backward()
        self._privatise_grads()
        opt.step()
        if self.is_fair:
            self.model.clamp_beta()

    def _ftgd_step(self, opt, x, ei, s, y, m):
        """FTGD: split the objective into an S-independent task pathway
        (released in the clear) and an S-dependent fairness pathway that is
        privatised at the level of its *sufficient statistics* -- the two group
        means (mu0, mu1) of the soft demographic-parity term.

        Privatising these two bounded-sensitivity scalars (Gaussian mechanism)
        yields (eps, delta)-DP for the *released fairness statistic* at
        negligible utility cost, avoiding the curse of dimensionality of noising
        the full |theta|-dimensional gradient. NOTE (scope): this does NOT make
        the whole transmitted update DP w.r.t. s -- the fairness-gradient still
        contains the raw group masks (grad flows through mu_g below) and FSER
        uses s in the forward pass. The guarantee is on the released statistic,
        not the update. See the manuscript's FTGD "Scope and limitations".
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
        if hasattr(self.model, "clamp_beta"):
            self.model.clamp_beta()

    def _puffle_step(self, opt, x, ei, s, y, m):
        """PUFFLE (Corbucci et al., ECML-PKDD'24): a per-round auto-tuned
        fairness weight lambda, driven by a momentum feedback-control loop
        that steers the (DP-noised) local demographic-parity gap toward a
        target disparity T -- replacing this codebase's static
        fairness_weight scalar -- combined with standard DP-SGD clipping and
        noise (Algorithm 1). The paper's third privacy channel (noised group-
        count statistics shared for group-imbalanced clients) is not
        reimplemented; see docs/BASELINES_AND_SOURCES.md."""
        cfg = self.cfg
        opt.zero_grad()
        pred = self.model(x, ei, s)[m]
        dpl = _soft_dpd(pred, s[m])
        dpl_val = float(dpl.detach())
        if self.dp_sigma > 0:
            dpl_val += float(torch.randn(()) * self.noise_multiplier * cfg.dp_clip)
        delta = cfg.puffle_target_dpd - dpl_val
        self._puffle_velocity = cfg.puffle_momentum * self._puffle_velocity + delta
        self._puffle_lambda = min(1.0, max(0.0, self._puffle_lambda - cfg.puffle_rho * self._puffle_velocity))

        loss = (1.0 - self._puffle_lambda) * _weighted_bce(pred, y[m].float()) \
            + self._puffle_lambda * dpl
        loss.backward()
        self._privatise_grads()
        opt.step()

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
        if self.cfg.sampling:
            y, pred, s = sampled_predict(self.model, self.data, mask, self.cfg, self.device)
        else:
            pred = self.model(self.data.x, self.data.edge_index, self.data.sensitive_attr)[mask]
            y, s = self.data.y[mask], self.data.sensitive_attr[mask]
        out = all_metrics(y, pred, s)
        out["n"] = int(mask.sum())
        return out

    def meta(self) -> Dict[str, float]:
        """Self-reported summary the server sees. Untrusted by construction.

        Since SPEC 4.0(c), ``evaluate`` returns NaN for a diverged model rather
        than the old ``auc=0.5, dpd=0.0``. This channel, however, feeds the
        metric-based aggregators (BFWA, FairFed) whose weight arithmetic needs a
        finite number, so the protocol-level report is coerced back to neutral
        values -- and the divergence is carried alongside as ``diverged`` so it
        stays visible in the logs instead of being laundered into a claim of
        perfect fairness. Reporting stays honest; only the wire format is fixed.
        """
        v = self.evaluate("val")
        s = self.data.sensitive_attr[self.data.train_mask]
        bad = bool(v.get("diverged", 0.0))
        auc = 0.5 if bad else v["auc"]
        return {"n": int(self.data.train_mask.sum()), "perf": auc,
                "dpd": 0.0 if bad else v["dpd"], "eod": 0.0 if bad else v["eod"],
                "eo": 0.0 if bad else v["eo"],
                "loss": 1.0 - auc, "diverged": float(bad),
                "group1_rate": float((s == 1).float().mean()) if len(s) else 0.5}
