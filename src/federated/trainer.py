"""Federated training loop (cross-silo, synchronous).

Each round: broadcast global weights -> clients train locally -> server forms
per-client pseudo-gradients, optionally poisons Byzantine ones, aggregates with
the configured rule, and updates the global model. The global model is then
evaluated on the *pooled* held-out test nodes across all clients, giving global
utility and fairness. Full per-round history is returned for logging/plots.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..config import ExperimentConfig, set_seed
from ..data import (load_dataset, partition_graph, partition_stats,
                    carve_server_holdout)
from ..models import build_model
from ..utils.metrics import all_metrics
from ..trust.privacy import PrivacyAccountant
from .aggregation import aggregate
from .attacks import poison_updates
from .client import Client, flatten_state, load_flat_state, sampled_predict, _soft_dpd


class FederatedTrainer:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        set_seed(config.seed)
        self.device = config.device

        data = load_dataset(config.dataset, root=config.data_root, seed=config.seed)
        self.in_channels = data.x.shape[1]

        # A1/F10: when the server scores clients it must do so on data no client
        # -- least of all a Byzantine one -- can touch. Carve that set out before
        # partitioning, so 'the server owns a small clean split' is an explicit
        # assumption rather than an artefact of reusing the pooled client nodes.
        self.server_holdout = None
        if config.fu_val_source == "server_holdout":
            self.server_holdout, data = carve_server_holdout(
                data, config.fu_holdout_size, seed=config.seed)

        self.clients_data = partition_graph(
            data, config.num_clients, method=config.partition,
            alpha=config.dirichlet_alpha, by=config.partition_by, seed=config.seed)
        self.partition_stats = partition_stats(self.clients_data)

        byz = set(range(min(config.num_byzantine, len(self.clients_data)))) \
            if config.attack != "none" else set()
        self.byzantine_ids = sorted(byz)
        self.clients: List[Client] = [
            Client(i, d, config, self.device, byzantine=(i in byz))
            for i, d in enumerate(self.clients_data)]

        ref = build_model(config.model, self.in_channels, config).to(self.device)
        self.global_flat = flatten_state(ref.state_dict()).cpu()
        self.ref_model = ref

        self.accountant = PrivacyAccountant(
            self.clients[0].noise_multiplier, delta=config.dp_delta) \
            if self.clients and self.clients[0].dp_sigma > 0 else None
        self.history: List[Dict] = []
        self._agg_state: Dict = {}   # persists stateful-aggregator state (e.g. fedgraphfair's lambda)

    # ----- global evaluation on pooled test nodes -----
    @torch.no_grad()
    def _group_offsets(self):
        """FDP-Fair post-processing: per-group additive offset that equalises
        group mean scores, calibrated on the pooled validation set."""
        vs, ss = [], []
        for d in self.clients_data:
            d = d.to(self.device)
            m = d.val_mask
            if m.sum() == 0:
                continue
            vs.append(self.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu())
            ss.append(d.sensitive_attr[m].cpu())
        if not vs:
            return 0.0, 0.0
        p = torch.cat(vs); s = torch.cat(ss)
        tgt = float(p.mean())
        o0 = tgt - float(p[s == 0].mean()) if (s == 0).any() else 0.0
        o1 = tgt - float(p[s == 1].mean()) if (s == 1).any() else 0.0
        return o0, o1

    @torch.no_grad()
    def _fedfact_offsets(self):
        """FedFACT post-processing: a shared global offset lambda (identical to
        FDP-Fair's, computed over all pooled clients) plus a per-client local
        offset mu_k computed from that client's own validation split only and
        never aggregated -- mirroring the paper's two-level global+local group
        fairness calibration (Prop. 2/Theorem 5), restricted to the linear/DP
        special case where the optimal dual reduces to closed-form mean-
        matching. See docs/BASELINES_AND_SOURCES.md."""
        lam = self._group_offsets()
        mus = []
        for d in self.clients_data:
            d = d.to(self.device)
            m = d.val_mask
            if m.sum() == 0:
                mus.append((0.0, 0.0)); continue
            p = self.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu()
            s = d.sensitive_attr[m].cpu()
            ltgt = float(p.mean())
            m0 = ltgt - float(p[s == 0].mean()) if (s == 0).any() else 0.0
            m1 = ltgt - float(p[s == 1].mean()) if (s == 1).any() else 0.0
            mus.append((m0, m1))
        return lam, mus

    @torch.no_grad()
    def evaluate_global(self) -> Dict[str, float]:
        load_flat_state(self.ref_model, self.global_flat.to(self.device))
        self.ref_model.eval()
        postproc = getattr(self.cfg, "postproc_fair", False)
        fedfact = getattr(self.cfg, "fedfact_post", False)
        offs_per_client = None
        if postproc:
            g = self._group_offsets()
            offs_per_client = [g] * len(self.clients_data)
        elif fedfact:
            lam, mus = self._fedfact_offsets()
            scale = self.cfg.fedfact_local_scale
            offs_per_client = [(lam[0] + scale * mu[0], lam[1] + scale * mu[1]) for mu in mus]
        ys, ps, ss = [], [], []
        cap = getattr(self.cfg, "eval_max_nodes", 0)
        per_client_cap = cap // max(1, len(self.clients_data)) if cap else 0
        for ci, d in enumerate(self.clients_data):
            d = d.to(self.device)
            mask = d.test_mask
            if mask.sum() == 0:
                continue
            offs = offs_per_client[ci] if offs_per_client is not None else None
            # On million-node graphs the test split alone can be millions of
            # nodes; evaluating it every round dominates wall-clock. Cap it to a
            # fixed random subsample (stable across rounds via a seeded generator,
            # so the convergence curve is not noisy).
            if per_client_cap and int(mask.sum()) > per_client_cap:
                idx = mask.nonzero(as_tuple=True)[0]
                g = torch.Generator().manual_seed(self.cfg.seed * 1000 + ci)
                perm = torch.randperm(idx.numel(), generator=g)[:per_client_cap]
                keep = idx[perm.to(idx.device)]
                mask = torch.zeros_like(d.test_mask)
                mask[keep] = True
            if self.cfg.sampling:
                yy, pred, sm = sampled_predict(self.ref_model, d, mask, self.cfg, self.device)
                ys.append(yy); ps.append(pred.cpu() if offs is None else pred); ss.append(sm)
                if offs is not None:
                    pred = (pred + torch.where(sm == 0, pred.new_tensor(offs[0]),
                                               pred.new_tensor(offs[1]))).clamp(0, 1)
                    ps[-1] = pred
                continue
            pred = self.ref_model(d.x, d.edge_index, d.sensitive_attr)[mask]
            sm = d.sensitive_attr[mask]
            if offs is not None:
                pred = (pred + torch.where(sm == 0, pred.new_tensor(offs[0]),
                                           pred.new_tensor(offs[1]))).clamp(0, 1)
            ys.append(d.y[mask].cpu()); ps.append(pred.cpu()); ss.append(sm.cpu())
        if not ys:
            return {}
        y = torch.cat(ys); p = torch.cat(ps); s = torch.cat(ss)
        return all_metrics(y, p, s)

    def _server_calibration_grad(self):
        """EquFL (Yu, Zhao, Zhang, Gao, Quan, Liu & Fang, arXiv:2601.05352,
        Apr 2026): a server-side fairness correction. Computes g0 = grad_w
        soft-DPD(w, D_val_pooled) at the current global model w^t, using the
        pooled validation set across clients as the server's calibration
        data (the paper instead condenses a synthetic dataset from model
        checkpoints because its server has no legitimate data access at
        all; our server already has legitimate access to pooled validation
        statistics for the FDP-Fair/FedFACT post-processing baselines, so we
        use that directly rather than approximating it). The paper proves
        this composes with *any* aggregation rule -- including Byzantine-
        robust ones (Median, Trimmed-mean, Multi-Krum) -- so it is added
        additively on top of BFWA/robust_bfwa without changing their logic.
        """
        load_flat_state(self.ref_model, self.global_flat.to(self.device))
        self.ref_model.zero_grad(set_to_none=True)
        self.ref_model.eval()   # deterministic forward; no dropout noise in the correction
        preds, senss = [], []
        for d in self.clients_data:
            d = d.to(self.device)
            m = d.val_mask
            if m.sum() == 0:
                continue
            preds.append(self.ref_model(d.x, d.edge_index, d.sensitive_attr)[m])
            senss.append(d.sensitive_attr[m])
        if not preds:
            return None
        pred = torch.cat(preds); s = torch.cat(senss)
        fair_loss = _soft_dpd(pred, s)
        named = [(n, p) for n, p in self.ref_model.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(fair_loss, [p for _, p in named], allow_unused=True)
        grad_map = {n: g for (n, _), g in zip(named, grads)}
        sd = self.ref_model.state_dict()
        parts = []
        for k, v in sd.items():
            g = grad_map.get(k)
            parts.append(g.detach().flatten() if g is not None else torch.zeros_like(v).flatten())
        return torch.cat(parts).cpu()

    # ----- one communication round -----
    def _round(self, t: int) -> Dict:
        updates, metas = [], []
        for c in self.clients:
            c.set_flat(self.global_flat)
            c.train()
            g_k = self.global_flat - c.get_flat()   # pseudo-gradient
            updates.append(g_k)
            metas.append(c.meta())
        # DP accounting is per-client: each client makes local_epochs noisy
        # releases per round in parallel, so we advance the (representative)
        # accountant once per round -- not once per client.
        if self.accountant:
            self.accountant.step(self.cfg.local_epochs)

        if self.byzantine_ids and self.cfg.attack not in ("none", "label_flip"):
            updates, metas = poison_updates(
                self.cfg.attack, updates, metas, self.byzantine_ids,
                self.cfg.attack_intensity)

        # FairShare-GNN: build the server target gradient on the pooled client
        # validation nodes (current global model), then let the FU-Shapley rule
        # score clients against it. Only computed when the aggregator needs it.
        g_target = g_task = g_fair = None
        fu_warmup = False
        if "fu_shapley" in self.cfg.aggregator:
            load_flat_state(self.ref_model, self.global_flat.to(self.device))
            from ..trust.incentive import (get_server_target_gradients,
                                           get_server_target_gradients_pooled)
            if self.server_holdout is not None:
                # A1 honoured: scored against a split no client owns.
                tg = get_server_target_gradients(
                    self.ref_model, self.server_holdout.to(self.device),
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            else:
                # A1 NOT honoured: the target is built from every client's
                # validation nodes, Byzantine clients included. Kept as the
                # default only so prior configs stay reproducible -- any
                # verifiability claim must use fu_val_source='server_holdout'.
                tg = get_server_target_gradients_pooled(
                    self.ref_model, self.clients_data, self.device,
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            if tg is not None:
                # Flat weights (and hence the client pseudo-gradients) live on
                # CPU by design; bring the target gradient back to match, so the
                # K-vector scoring stays on one device on GPU runs.
                g_target, g_task, g_fair = (g.cpu() for g in tg)
            fu_warmup = t < self.cfg.fu_warmup_rounds

        g_agg, info = aggregate(
            self.cfg.aggregator, updates, metas,
            tau=self.cfg.fairness_budget, fw_iters=self.cfg.fw_iterations,
            dual_step=self.cfg.dual_step_size, trimmed_beta=self.cfg.trimmed_beta,
            krum_f=max(self.cfg.krum_f, len(self.byzantine_ids)),
            q_ffl=self.cfg.q_ffl, fairfed_beta=self.cfg.fairfed_beta,
            state=self._agg_state,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=self.cfg.fu_alpha, fu_beta_ema=self.cfg.fu_ema_beta,
            fu_normalize=self.cfg.fu_normalize, fu_score=self.cfg.fu_score,
            fu_warmup=fu_warmup, fu_grad_clip=self.cfg.fu_grad_clip,
            fu_warmup_agg=self.cfg.fu_warmup_agg)

        if self.cfg.server_calib and (t + 1) >= self.cfg.rounds * self.cfg.server_calib_start_frac:
            g0 = self._server_calibration_grad()
            if g0 is not None:
                g_agg = g_agg + self.cfg.server_calib_gamma * g0

        self.global_flat = self.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in self.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        for key in ("phi_raw", "phi_util", "phi_fair", "phi_ema", "fu_warmup", "kept",
                    "fu_fallback", "phi_nan_frac", "n_clipped",
                    "g_norm_median", "g_norm_max", "phi_norm"):
            if key in info:
                rec[key] = info[key]
        return rec

    def run(self, verbose=False) -> Dict:
        for t in range(self.cfg.rounds):
            rec = self._round(t)
            self.history.append(rec)
            if verbose and (t % max(1, self.cfg.rounds // 10) == 0 or t == self.cfg.rounds - 1):
                print(f"  round {rec['round']:3d}  AUC={rec.get('g_auc',0):.3f} "
                      f"DPD={rec.get('g_dpd',0):.3f} EOD={rec.get('g_eod',0):.3f}")
        final = self.evaluate_global()
        if self.accountant:
            final["epsilon"] = self.accountant.epsilon()
            final["delta"] = self.cfg.dp_delta
        return {"final": final, "history": self.history,
                "partition_stats": self.partition_stats,
                "byzantine_ids": self.byzantine_ids}
