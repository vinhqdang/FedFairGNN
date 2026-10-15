"""Federated training loop (cross-silo, synchronous).

Each round: broadcast global weights -> clients train locally -> server forms
per-client pseudo-gradients, optionally poisons Byzantine ones, aggregates with
the configured rule, and updates the global model. The global model is then
evaluated on the *pooled* held-out test nodes across all clients, giving global
utility and fairness. Full per-round history is returned for logging/plots.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from ..config import ExperimentConfig, set_seed
from ..data import (load_dataset, partition_graph, partition_stats,
                    carve_server_holdout)
from ..models import build_model
from ..utils.metrics import all_metrics
from ..trust.privacy import PrivacyAccountant, calibrate_noise_multiplier
from .aggregation import aggregate
from .attacks import falsify_group_stats, poison_updates
from .client import Client, flatten_state, load_flat_state, sampled_predict, _soft_dpd


def public_n_cap(clients_data) -> int:
    """Public cap on per-client, per-group validation counts (r2_n_cap = 0).

    The largest per-client validation size, rounded up to a multiple of 50. It
    counts nodes without reading s, so it is independent of the sensitive attribute
    and can be published. Every group count is at most the client's validation size,
    so the cap never cuts a group.
    """
    largest = max(int(d.val_mask.sum()) for d in clients_data)
    return int(math.ceil(largest / 50.0) * 50)


def _honest_diagnostics(stats: List[Dict[str, float]], byzantine_ids: List[int],
                        n_cap: float) -> Dict[str, float]:
    """Honest-only gap D^H and the T3 bound sum_a b_a, from the statistics before
    falsification. The bound is b_a = f n_cap / (N^H_a + f n_cap)."""
    byz = set(byzantine_ids)
    hon = [st for i, st in enumerate(stats) if i not in byz]
    S = [sum(st[f"S{a}"] for st in hon) for a in (0, 1)]
    N = [sum(st[f"N{a}"] for st in hon) for a in (0, 1)]
    r = [S[a] / N[a] if N[a] > 0 else 0.5 for a in (0, 1)]
    f = len(byz)
    b = [f * n_cap / (N[a] + f * n_cap) if f > 0 else 0.0 for a in (0, 1)]
    gap = r[1] - r[0]
    return {"gap_honest": gap, "dhat_honest": abs(gap), "t3_bound": b[0] + b[1],
            "n_cap": n_cap}


def _clip_to_norm(u: torch.Tensor, C: float) -> torch.Tensor:
    """Protocol step 3 of R2: scale a submission down to norm C (no-op inside the ball)."""
    return u * min(1.0, C / max(float(u.norm(2)), 1e-12))


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

        # TrustFedGNN-R2 setup. Public count cap: derived from the partition when
        # r2_n_cap is 0 (see public_n_cap). The statistics noise is one Gaussian
        # release per round with L2 sensitivity 2, so sigma_s = 2 z (spec T1).
        self._r2_n_cap = 0
        self._feasible_flat = None      # theta^t of the last release round with Dhat <= tau
        self._feasible_round = None
        # Warm-up for the feasible rule: round 1 is untrained and trivially feasible.
        self._select_warmup = math.ceil(round(config.r2_select_warmup * config.rounds, 9))
        self._secagg_sigma_s = 0.0
        if config.r2_direction not in ("sign", "prop"):
            raise ValueError(f"r2_direction must be 'sign' or 'prop', got {config.r2_direction!r}")
        if config.r2_output not in ("last", "feasible"):
            raise ValueError(f"r2_output must be 'last' or 'feasible', got {config.r2_output!r}")
        self._secagg_gen = torch.Generator().manual_seed(config.seed)
        if config.aggregator == "secagg_lagrangian":
            if config.r2_n_cap > 0:
                self._r2_n_cap = int(config.r2_n_cap)
            else:
                self._r2_n_cap = public_n_cap(self.clients_data)
            for c in self.clients:
                c.r2_n_cap = self._r2_n_cap
            if config.r2_stat_dp:
                # One release per statistics round: ceil(R / P) of them.
                releases = math.ceil(config.rounds / max(1, config.r2_stat_period))
                z = calibrate_noise_multiplier(config.dp_epsilon, steps=releases,
                                               delta=config.dp_delta)
                self._secagg_sigma_s = 2.0 * z

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
        r2 = self.cfg.aggregator == "secagg_lagrangian"
        # Off-period rounds (r2_stat_period > 1) release no statistics; lambda and
        # the direction stay at the last broadcast values.
        r2_release = r2 and t % max(1, self.cfg.r2_stat_period) == 0
        pre_flat = (self.global_flat.clone()
                    if r2_release and self.cfg.r2_output == "feasible" else None)
        fedfdp = self.cfg.dp_mode == "fedfdp"
        lam_in = float(self._agg_state.get("r2_lambda", 0.0))
        dir_in = float(self._agg_state.get("r2_dir", 0))
        updates, metas, stats = [], [], []
        for c in self.clients:
            c.set_flat(self.global_flat)
            if r2:
                c.lagrange = (lam_in, dir_in)
            if r2_release:
                # Statistics are taken with the received model, before training.
                stats.append(c.group_stats())
            if fedfdp:
                c.fedfdp_fbar = float(self._agg_state.get("fedfdp_fbar", 0.0))
            c.train()
            g_k = self.global_flat - c.get_flat()   # pseudo-gradient
            updates.append(g_k)
            metas.append(c.meta())
        # DP accounting is per-client: each client makes local_epochs noisy
        # releases per round in parallel, so we advance the (representative)
        # accountant once per round -- not once per client.
        if self.accountant:
            # FedFDP also releases one loss statistic per round (Client.train), which
            # is a separate Gaussian release on the same data, so it is counted here.
            self.accountant.step(self.cfg.local_epochs + (1 if fedfdp else 0))

        if self.byzantine_ids and self.cfg.attack not in ("none", "label_flip") \
                and not self.cfg.attack.startswith("stat_falsify"):
            updates, metas = poison_updates(
                self.cfg.attack, updates, metas, self.byzantine_ids,
                self.cfg.attack_intensity,
                ipm_epsilon=self.cfg.ipm_epsilon, alie_z=self.cfg.alie_z,
                                           meta_lie=getattr(self.cfg, 'meta_lie', None))

        secagg = None
        diag = None
        if r2:
            C = self.cfg.r2_clip_norm
            if r2_release:
                diag = _honest_diagnostics(stats, self.byzantine_ids, self._r2_n_cap)
            if r2_release and self.byzantine_ids and self.cfg.attack.startswith("stat_falsify"):
                mode = "mid" if self.cfg.attack == "stat_falsify_mid" else "swap"
                stats = falsify_group_stats(stats, self.byzantine_ids, self._r2_n_cap, mode=mode)
            # Every client clips to C (protocol step 3). A norm-aware update attacker
            # is rescaled to C the same way. Only a non-compliant client skips this,
            # and the server's norm check rejects it.
            updates = [_clip_to_norm(u, C) for u in updates]
            secagg = dict(C=C, eta_lambda=self.cfg.r2_lambda_step, n_cap=self._r2_n_cap,
                          sigma_s=self._secagg_sigma_s, sigma_u=self.cfg.r2_update_noise,
                          generator=self._secagg_gen, release=r2_release,
                          direction_rule=self.cfg.r2_direction)

        # FairShare-GNN: build the server target gradient on the pooled client
        # validation nodes (current global model), then let the FU-Shapley rule
        # score clients against it. Only computed when the aggregator needs it.
        g_target = g_task = g_fair = None
        fu_warmup = False
        # fltrust needs the same server-side reference machinery: it scores
        # against g_task (the root update) rather than the bi-objective g_target,
        # but both are produced by the same holdout pass.
        if "fu_shapley" in self.cfg.aggregator or "fltrust" in self.cfg.aggregator:
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
            fu_warmup_agg=self.cfg.fu_warmup_agg,
            bfwa_persist_dual=self.cfg.bfwa_persist_dual,
            stats=stats if r2 else None, secagg=secagg)

        if fedfdp:
            # Fbar^{t+1} = sum_i p_i Ftilde_i with p_i = n_i / n (spec Section 5).
            n_tot = sum(m["n"] for m in metas)
            if n_tot > 0:
                self._agg_state["fedfdp_fbar"] = float(
                    sum(m["n"] * m["loss"] for m in metas) / n_tot)

        if self.cfg.server_calib and (t + 1) >= self.cfg.rounds * self.cfg.server_calib_start_frac:
            g0 = self._server_calibration_grad()
            if g0 is not None:
                g_agg = g_agg + self.cfg.server_calib_gamma * g0

        if pre_flat is not None and info.get("dhat") is not None \
                and info["dhat"] <= self.cfg.fairness_budget \
                and t + 1 >= self._select_warmup:
            # Server-side selection on the released Dhat: keep theta^t (the model the
            # statistics were computed on) and its round number.
            self._feasible_flat = pre_flat
            self._feasible_round = t + 1
        self.global_flat = self.global_flat - g_agg

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in self.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        for key in ("phi_raw", "phi_util", "phi_fair", "phi_ema", "fu_warmup", "kept",
                    "fu_fallback", "phi_nan_frac", "n_clipped", "n_null",
                    "g_norm_median", "g_norm_max", "phi_norm",
                    # BFWA diagnostics: fulfils the manuscript's promise to report
                    # the per-round constraint residual and feasibility rate (see
                    # bfwa_weights' docstring in aggregation.py) -- neither was
                    # computed anywhere before that fix.
                    "bfwa_mu", "constraint_residual", "feasible",
                    "constraint_residual_preclamp", "feasible_preclamp",
                    # TrustFedGNN-R2 round log (secagg_lagrangian only)
                    "lambda", "lambda_in", "direction", "dhat", "gap",
                    "n_accepted", "n_rejected", "u_norm"):
            if key in info:
                rec[key] = info[key]
        if diag is not None:
            rec.update(diag)
        if fedfdp:
            rec["fedfdp_fbar"] = self._agg_state.get("fedfdp_fbar")
        # FTGD: how many local steps this round degraded to plain weighted-sum
        # training because ||g_fair|| fell below cfg.ftgd_min_fair_norm (see
        # Client._gradient_surgery). Client-local by construction; pooled here
        # rather than left client-side only, so it is visible in the same
        # per-round record as everything else.
        skips = [getattr(c, "_ftgd_skipped_projection_count", 0) for c in self.clients]
        if any(skips):
            rec["ftgd_skipped_projection_total"] = sum(skips)
        return rec

    def _final_metrics(self) -> Dict:
        """Final evaluation. With r2_output="feasible" this scores the model of the
        most recent release round whose released Dhat was <= tau (server-side
        post-processing of released outputs, no extra privacy cost); if no release
        round was feasible, the last model is used and selected_round is None."""
        if self.cfg.aggregator != "secagg_lagrangian" or self.cfg.r2_output != "feasible":
            return self.evaluate_global()
        if self._feasible_flat is None:
            final = self.evaluate_global()
            final["selected_round"] = None
            return final
        last = self.global_flat
        self.global_flat = self._feasible_flat
        try:
            final = self.evaluate_global()
        finally:
            self.global_flat = last
        final["selected_round"] = self._feasible_round
        return final

    def run(self, verbose=False) -> Dict:
        for t in range(self.cfg.rounds):
            rec = self._round(t)
            self.history.append(rec)
            if verbose and (t % max(1, self.cfg.rounds // 10) == 0 or t == self.cfg.rounds - 1):
                print(f"  round {rec['round']:3d}  AUC={rec.get('g_auc',0):.3f} "
                      f"DPD={rec.get('g_dpd',0):.3f} EOD={rec.get('g_eod',0):.3f}")
        final = self._final_metrics()
        if self.accountant:
            final["epsilon"] = self.accountant.epsilon()
            final["delta"] = self.cfg.dp_delta
        return {"final": final, "history": self.history,
                "partition_stats": self.partition_stats,
                "byzantine_ids": self.byzantine_ids}
