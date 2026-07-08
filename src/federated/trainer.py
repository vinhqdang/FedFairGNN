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
from ..data import load_dataset, partition_graph, partition_stats
from ..models import build_model
from ..utils.metrics import all_metrics
from ..trust.privacy import PrivacyAccountant
from .aggregation import aggregate
from .attacks import poison_updates
from .client import Client, flatten_state, load_flat_state, sampled_predict


class FederatedTrainer:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        set_seed(config.seed)
        self.device = config.device

        data = load_dataset(config.dataset, root=config.data_root, seed=config.seed)
        self.in_channels = data.x.shape[1]
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
    def evaluate_global(self) -> Dict[str, float]:
        load_flat_state(self.ref_model, self.global_flat.to(self.device))
        self.ref_model.eval()
        offs = self._group_offsets() if getattr(self.cfg, "postproc_fair", False) else None
        ys, ps, ss = [], [], []
        for d in self.clients_data:
            d = d.to(self.device)
            mask = d.test_mask
            if mask.sum() == 0:
                continue
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

        g_agg, info = aggregate(
            self.cfg.aggregator, updates, metas,
            tau=self.cfg.fairness_budget, fw_iters=self.cfg.fw_iterations,
            dual_step=self.cfg.dual_step_size, trimmed_beta=self.cfg.trimmed_beta,
            krum_f=max(self.cfg.krum_f, len(self.byzantine_ids)),
            q_ffl=self.cfg.q_ffl, fairfed_beta=self.cfg.fairfed_beta)
        self.global_flat = self.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in self.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
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
