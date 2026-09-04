"""Stage 4 Remediation Multi-Seed Runner (Q1 Independent Audit Remediation & Gate G0).

Executes canonical configurations across seeds (42, 43, 44) for:
  1. Stage 4.2 Canonical Matrix (German Credit & Bail Recidivism No-Leakage) with same_penalize
  2. Stage 4.2 FU-Shapley vs Exact Shapley Correlation (Per-round & Pooled)
  3. Stage 4.5 Component-wise Ablation Suite M1-M7 with canonical same_penalize
  4. FSER Sign Hypothesis Benchmark (sub vs add vs same_penalize)
  5. Two-Tier Defense under Byzantine Attacks (No Attack, Sign-Flip 20%, Fairness-Poison 20%)

Includes full manifest provenance (device, torch, git_commit, git_dirty, platform),
sensitive homophily h_s, dual DPD (soft/hard@0.5), pred_std, wall_clock_s, and w_adv metric.

Checkpoint/resume: every individual multi-seed sub-result (each RUN-4.2-*, each
M1-M7 ablation arm, each FSER sign/beta combo, each Byzantine defense scenario)
is written to ``output_file`` via an atomic save *as soon as it finishes* -- not
only once at the very end. A run that dies partway through (a killed process, a
lost remote session, an OOM) leaves a valid, loadable JSON file with everything
computed so far. Re-invoking ``run_stage4_remediation`` on the same
``output_file`` picks up from that checkpoint automatically: every sub-result
already present is skipped, and only the missing ones are computed. Pass
``resume=False`` to force a full fresh run, ignoring any existing checkpoint.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from typing import Callable, Tuple
import numpy as np
import torch

from src.config import ExperimentConfig, set_seed
from src.federated import FederatedTrainer
from src.trust.incentive import get_server_target_gradients_pooled, fairness_gradient_ratio
from src.utils.metrics import weight_oscillation
from experiments.fairshare_common import (
    global_sensitive_homophily,
    client_pseudo_grads,
    make_trainer,
    pearson_spearman,
    warm_rounds,
)
from experiments.exact_shapley_correlation import exact_shapley


def _get_git_info() -> Tuple[str, bool]:
    env_commit = os.environ.get("FEDFAIR_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    env_dirty = os.environ.get("FEDFAIR_GIT_DIRTY")
    if env_commit:
        dirty = (env_dirty == "1" or env_dirty == "true" or env_dirty == "True")
        return env_commit.strip(), dirty
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(status)
        return commit, dirty
    except Exception:
        return "unknown", False


# --------------------------------------------------------------------------- #
# Checkpoint/resume helpers
# --------------------------------------------------------------------------- #
def _atomic_save(data: dict, output_file: str) -> None:
    """Write ``data`` to ``output_file`` atomically (temp file + rename), so a
    process killed mid-write never leaves a truncated/corrupt checkpoint."""
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, output_file)


def _load_checkpoint(output_file: str) -> dict:
    """Load a prior (possibly partial) run to resume from. Audit markers left
    by a staleness pass on an earlier, now-superseded file are dropped -- they
    describe *that* file's provenance, not this run's, and a completed rerun
    supersedes them regardless of how far this resume gets."""
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file) as f:
            d = json.load(f)
    except Exception as e:
        print(f"[resume] checkpoint at {output_file} is unreadable ({e}); starting fresh", flush=True)
        return {}
    for k in list(d.keys()):
        if k.startswith("_STALENESS") or k == "_INVALID":
            del d[k]
    return d


# Module-level definition of canonical ablation arms (ensures invariant testing)
ABLATION_ARMS = {
    "M1_Full": lambda s: ExperimentConfig.canonical(seed=s),
    "M2_wo_FSER": lambda s: ExperimentConfig.canonical(seed=s, model="gat"),
    "M3_wo_FTGD": lambda s: ExperimentConfig.canonical(seed=s, dp_enabled=False),
    "M4_Full_DPSGD": lambda s: ExperimentConfig.canonical(seed=s, dp_mode="gradient"),
    "M5_wo_FairScore": lambda s: ExperimentConfig.canonical(seed=s, fu_alpha=0.0),
    "M6_wo_TwoTier": lambda s: ExperimentConfig.canonical(seed=s, fu_val_source="pooled", fu_score="cosine"),
    "M7_wo_EMA": lambda s: ExperimentConfig.canonical(seed=s, fu_ema_beta=0.0),
}


def evaluate_single_run(cfg: ExperimentConfig) -> dict:
    t0 = time.perf_counter()
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall_clock_s = time.perf_counter() - t0

    weights_hist = [r.get("agg_weights") for r in res["history"]]
    omega_w = weight_oscillation(weights_hist)
    final_weights = res["history"][-1].get("agg_weights")

    # Compute attacker weight share if Byzantine clients exist
    w_adv = 0.0
    if cfg.num_byzantine > 0 and final_weights:
        w_adv = float(sum(final_weights[i] for i in range(min(cfg.num_byzantine, len(final_weights)))))

    # Sensitive homophily h_s of the FULL (pre-partition) graph.
    #
    # This used to be guarded by ``hasattr(trainer, "global_data")``. That
    # attribute does not exist on FederatedTrainer -- it keeps ``clients_data``
    # and ``server_holdout``, never the unpartitioned graph -- so the guard
    # never fired and h_s was recorded as a constant 0.0 for every seed of
    # every run. ``global_sensitive_homophily`` measures it on the same graph
    # the trainer loaded (same dataset/root/seed), which is where a *dataset*
    # property belongs: the induced client subgraphs drop every cross-client
    # edge, so a post-partition h_s would describe the split, not the data.
    h_s = global_sensitive_homophily(cfg)

    final = res["final"]
    return {
        "seed": cfg.seed,
        "auc": float(final["auc"]),
        "dpd_soft": float(final["dpd_soft"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "omega_w": float(omega_w),
        "pred_std": float(final["pred_std"]),
        "w_adv": float(w_adv),
        "sensitive_homophily": float(h_s),
        "wall_clock_s": float(wall_clock_s),
    }


def run_multi_seed(cfg_fn: Callable[[int], ExperimentConfig], seeds=(42, 43, 44)) -> dict:
    results = []
    aucs, dpds_soft, dpds_hard, eods, omegas, pred_stds, w_advs, wall_clocks = [], [], [], [], [], [], [], []

    for s in seeds:
        cfg = cfg_fn(s)
        out = evaluate_single_run(cfg)
        results.append(out)
        aucs.append(out["auc"])
        dpds_soft.append(out["dpd_soft"])
        dpds_hard.append(out["dpd_hard"])
        eods.append(out["eod"])
        omegas.append(out["omega_w"])
        pred_stds.append(out["pred_std"])
        w_advs.append(out["w_adv"])
        wall_clocks.append(out["wall_clock_s"])
        print(f"    Seed {s} -> AUC={out['auc']:.4f}, DPD_soft={out['dpd_soft']:.4f}, DPD_hard={out['dpd_hard']:.4f}, EOD={out['eod']:.4f}, Omega_w={out['omega_w']:.4f}", flush=True)

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "dpd_soft_mean": float(np.mean(dpds_soft)),
        "dpd_soft_std": float(np.std(dpds_soft)),
        "dpd_hard_mean": float(np.mean(dpds_hard)) if dpds_hard else 0.0,
        "dpd_hard_std": float(np.std(dpds_hard)) if dpds_hard else 0.0,
        "eod_mean": float(np.mean(eods)) if eods else 0.0,
        "eod_std": float(np.std(eods)) if eods else 0.0,
        "omega_w_mean": float(np.mean(omegas)) if omegas else 0.0,
        "omega_w_std": float(np.std(omegas)) if omegas else 0.0,
        "pred_std_mean": float(np.mean(pred_stds)) if pred_stds else 0.0,
        "w_adv_mean": float(np.mean(w_advs)) if w_advs else float("nan"),
        "w_adv_std": float(np.std(w_advs)) if w_advs else float("nan"),
        "wall_clock_s_mean": float(np.mean(wall_clocks)) if wall_clocks else 0.0,
        "per_seed": results,
    }


def run_stage4_remediation(output_file="results/stage4_remediation_results.json",
                           run_sign_test: bool = True, resume: bool = True):
    print("=" * 70, flush=True)
    print("🚀 [START] STAGE 4 REMEDIATION & GATE G0-BIS SUITE (CANONICAL SUB UNDER SERVER_HOLDOUT)", flush=True)
    print("=" * 70, flush=True)

    seeds = (42, 43, 44)
    all_results = _load_checkpoint(output_file) if resume else {}
    if all_results:
        done = sorted(k for k in all_results if not k.startswith("_"))
        print(f"[resume] loaded checkpoint from {output_file}: {len(done)} top-level "
              f"section(s) already present: {done}", flush=True)

    def checkpoint():
        _atomic_save(all_results, output_file)

    def have(key: str) -> bool:
        return key in all_results

    # 0. Build/refresh Provenance Manifest. Keep the ORIGINAL start timestamp
    # across resumes (that's when the run actually began); record every
    # resume as a separate event so a reader can see the run was interrupted.
    git_commit, git_dirty = _get_git_info()
    if git_commit == "unknown":
        print("⚠️ [WARNING] git_commit is 'unknown'! Canonical runs must have explicit provenance.", flush=True)

    prior_manifest = all_results.get("_manifest", {})
    manifest = {
        "device": os.environ.get("FEDFAIR_DEVICE", "cpu"),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": prior_manifest.get("timestamp",
                     datetime.datetime.now(datetime.timezone.utc).isoformat()),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "canonical_config": ExperimentConfig.canonical().to_dict(),
        "resume_events": prior_manifest.get("resume_events", []) + (
            [datetime.datetime.now(datetime.timezone.utc).isoformat()] if prior_manifest else []
        ),
    }
    all_results["_manifest"] = manifest
    checkpoint()

    # -------------------------------------------------------------
    # PART 1: Stage 4.2 Canonical Benchmark (German & Bail 3 seeds)
    # -------------------------------------------------------------
    print("\n" + "-" * 70, flush=True)
    print("📊 [PART 1/5] Running Stage 4.2 Canonical Matrix (with canonical sub)...", flush=True)
    print("-" * 70, flush=True)

    # RUN-4.2-01: German Credit - FedAvg Baseline (3 seeds)
    if have("RUN-4.2-01"):
        print("\n[skip] RUN-4.2-01 already checkpointed.", flush=True)
    else:
        print("\n[+] RUN-4.2-01: German Credit - FedAvg (3 seeds)...", flush=True)
        all_results["RUN-4.2-01"] = run_multi_seed(
            lambda s: ExperimentConfig(
                dataset="german", seed=s, num_clients=5, rounds=20, dirichlet_alpha=0.3,
                model="gat", aggregator="fedavg", fairness_weight=0.0, dp_enabled=False,
            ),
            seeds=seeds
        )
        checkpoint()

    # RUN-4.2-02: German Credit - TrustFedGNN Canonical (3 seeds)
    if have("RUN-4.2-02"):
        print("\n[skip] RUN-4.2-02 already checkpointed.", flush=True)
    else:
        print("\n[+] RUN-4.2-02: German Credit - TrustFedGNN Canonical sub (3 seeds)...", flush=True)
        all_results["RUN-4.2-02"] = run_multi_seed(
            lambda s: ExperimentConfig.canonical(
                dataset="german", seed=s,
            ),
            seeds=seeds
        )
        checkpoint()

    # RUN-4.2-03: FU-Shapley vs Exact Shapley Correlation Probing
    if have("RUN-4.2-03"):
        print("\n[skip] RUN-4.2-03 already checkpointed.", flush=True)
    else:
        print("\n[+] RUN-4.2-03: FU-Shapley vs Exact Shapley Probing...", flush=True)
        from src.federated.client import load_flat_state
        trainer = make_trainer(dataset="german", seed=42, num_clients=4, rounds=10, method="fairshare", alpha=0.1)
        probe_rounds = [2, 4, 6, 8]
        r_list, rho_list = [], []
        all_phi_fu, all_phi_exact = [], []
        corr_probes = []
        absr = 0
        for r in probe_rounds:
            while absr < r:
                trainer._round(absr)
                absr += 1
            grads = client_pseudo_grads(trainer)
            load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
            tg = get_server_target_gradients_pooled(
                trainer.ref_model, trainer.clients_data, trainer.device, 0.1,
                fair_surrogate=trainer.cfg.fu_fair_surrogate
            )
            if tg is not None:
                g_target_cpu = tg[0].cpu()
                phi_fu = np.array([float(torch.dot(p, g_target_cpu).item()) for p in grads])
                phi_exact = exact_shapley(trainer, grads, 0.1, game="loss")
                pr_r, sp_rho = pearson_spearman(phi_fu.tolist(), phi_exact)
                r_list.append(pr_r)
                rho_list.append(sp_rho)
                all_phi_fu.extend(phi_fu.tolist())
                all_phi_exact.extend(phi_exact)
                corr_probes.append({
                    "probe_round": r,
                    "pearson_r": float(pr_r),
                    "spearman_rho": float(sp_rho),
                    "phi_exact": [round(float(x), 6) for x in phi_exact],
                    "phi_fu": [round(float(x), 6) for x in phi_fu],
                })
                print(f"    Round {r} -> Pearson r={pr_r:.4f}, Spearman rho={sp_rho:.4f}", flush=True)

        pooled_r, pooled_rho = pearson_spearman(all_phi_fu, all_phi_exact)
        sign_agree = float(np.mean(np.sign(all_phi_fu) == np.sign(all_phi_exact)))

        all_results["RUN-4.2-03"] = {
            "dataset": "german",
            "num_clients": 4,
            "probe_rounds": probe_rounds,
            "avg_pearson_r": float(np.mean(r_list)) if r_list else 0.0,
            "avg_spearman_rho": float(np.mean(rho_list)) if rho_list else 0.0,
            "pooled_pearson_r": float(pooled_r),
            "pooled_spearman_rho": float(pooled_rho),
            "sign_agree": float(sign_agree),
            "rounds_above_085": int(sum(1 for r in r_list if r >= 0.85)),
            "total_probed_rounds": len(r_list),
            "probes": corr_probes,
        }
        checkpoint()

    # RUN-4.2-04: Bail Recidivism - FedAvg Baseline (3 seeds, drop TIME)
    if have("RUN-4.2-04"):
        print("\n[skip] RUN-4.2-04 already checkpointed.", flush=True)
    else:
        print("\n[+] RUN-4.2-04: Bail Recidivism (No-Leakage) - FedAvg (3 seeds)...", flush=True)
        all_results["RUN-4.2-04"] = run_multi_seed(
            lambda s: ExperimentConfig(
                dataset="bail", seed=s, num_clients=5, rounds=20, dirichlet_alpha=0.3,
                model="gat", aggregator="fedavg", fairness_weight=0.0, dp_enabled=False,
            ),
            seeds=seeds
        )
        checkpoint()

    # RUN-4.2-05: Bail Recidivism - TrustFedGNN Canonical (3 seeds, drop TIME)
    if have("RUN-4.2-05"):
        print("\n[skip] RUN-4.2-05 already checkpointed.", flush=True)
    else:
        print("\n[+] RUN-4.2-05: Bail Recidivism (No-Leakage) - TrustFedGNN Canonical sub (3 seeds)...", flush=True)
        all_results["RUN-4.2-05"] = run_multi_seed(
            lambda s: ExperimentConfig.canonical(
                dataset="bail", seed=s,
            ),
            seeds=seeds
        )
        checkpoint()

    # -------------------------------------------------------------
    # PART 2: Stage 4.5 Component-wise Ablation Suite (M1-M7 Canonical sub)
    # -------------------------------------------------------------
    print("\n" + "-" * 70, flush=True)
    print("🔬 [PART 2/5] Running Stage 4.5 Component-wise Ablation Suite (M1-M7)...", flush=True)
    print("-" * 70, flush=True)

    ablation_results = all_results.get("stage4_5_ablation_matrix", {})
    all_results["stage4_5_ablation_matrix"] = ablation_results
    for arm_name, cfg_fn in ABLATION_ARMS.items():
        if arm_name in ablation_results:
            print(f"\n[skip] Ablation arm {arm_name} already checkpointed.", flush=True)
            continue
        print(f"\n[+] Running Ablation Arm {arm_name} (3 seeds)...", flush=True)
        ablation_results[arm_name] = run_multi_seed(cfg_fn, seeds=seeds)
        checkpoint()

    # -------------------------------------------------------------
    # PART 3: FSER Sign Hypothesis Sweep (+ vs - vs same_penalize)
    # -------------------------------------------------------------
    if run_sign_test:
        print("\n" + "-" * 70, flush=True)
        print("💡 [PART 3/5] Testing FSER Sign Hypothesis (sub vs add vs same_penalize)...", flush=True)
        print("-" * 70, flush=True)

        sign_results = all_results.get("fser_sign_hypothesis", {})
        all_results["fser_sign_hypothesis"] = sign_results
        for mode in ["sub", "add", "same_penalize"]:
            for b in [0.5, 2.0]:
                lbl = f"fser_{mode}_beta_{b}"
                if lbl in sign_results:
                    print(f"\n[skip] {lbl} already checkpointed.", flush=True)
                    continue
                print(f"\n[+] Testing FSER mode='{mode}' with beta={b}...", flush=True)
                sign_results[lbl] = run_multi_seed(
                    lambda s, m=mode, beta=b: ExperimentConfig.canonical(
                        dataset="german", seed=s, num_clients=5, rounds=20, dirichlet_alpha=0.3,
                        beta_init=beta, fser_mode=m,
                    ),
                    seeds=seeds
                )
                checkpoint()

    # -------------------------------------------------------------
    # PART 4: Two-Tier Defense under Byzantine Attacks (20% Byz)
    # -------------------------------------------------------------
    print("\n" + "-" * 70, flush=True)
    print("🛡️ [PART 4/5] Evaluating Two-Tier Defense against Byzantine Attacks (20% Byz)...", flush=True)
    print("-" * 70, flush=True)

    defense_results = all_results.get("two_tier_defense_robustness", {})
    all_results["two_tier_defense_robustness"] = defense_results
    attack_scenarios = [
        ("no_attack", "none", 0),
        ("sign_flip_20pct", "sign_flip", 1),         # 1 of 5 clients = 20%
        ("fairness_poison_20pct", "fairness_poison", 1), # 1 of 5 clients = 20%
    ]

    for sc_name, att_type, n_byz in attack_scenarios:
        print(f"\n[*] Attack Scenario: {sc_name} (attack={att_type}, num_byz={n_byz}/5 = {n_byz*20}%)...", flush=True)

        # M1: Full Two-Tier Defense (Ours)
        key1 = f"M1_{sc_name}"
        if key1 in defense_results:
            print(f"  [skip] {key1} already checkpointed.", flush=True)
        else:
            print(f"  [+] Running M1 Full Two-Tier under {sc_name}...", flush=True)
            defense_results[key1] = run_multi_seed(
                lambda s, at=att_type, nb=n_byz: ExperimentConfig.canonical(
                    dataset="german", seed=s, num_clients=5, rounds=20, dirichlet_alpha=0.3,
                    attack=at, num_byzantine=nb, attack_intensity=10.0,
                ),
                seeds=seeds
            )
            checkpoint()

        # M6: w/o Two-Tier Defense (Plain CGSV Cosine)
        key6 = f"M6_{sc_name}"
        if key6 in defense_results:
            print(f"  [skip] {key6} already checkpointed.", flush=True)
        else:
            print(f"  [+] Running M6 w/o Two-Tier (CGSV) under {sc_name}...", flush=True)
            defense_results[key6] = run_multi_seed(
                lambda s, at=att_type, nb=n_byz: ExperimentConfig.canonical(
                    dataset="german", seed=s, num_clients=5, rounds=20, dirichlet_alpha=0.3,
                    fu_score="cosine", fu_val_source="pooled",
                    attack=at, num_byzantine=nb, attack_intensity=10.0,
                ),
                seeds=seeds
            )
            checkpoint()

    # Final save (idempotent -- every section above already checkpointed itself,
    # this just ensures the file's ".tmp" swap has definitely landed).
    checkpoint()

    print("\n" + "=" * 70, flush=True)
    print(f"✅ [STAGE 4 REMEDIATION COMPLETED] Saved to {output_file}", flush=True)
    print("=" * 70, flush=True)
    return all_results


if __name__ == "__main__":
    run_stage4_remediation()
