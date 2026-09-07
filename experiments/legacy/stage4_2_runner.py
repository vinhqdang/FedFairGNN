"""Stage 4.2 Runner: Small Real Benchmarks & Incentive Correlation.

Covers:
1. RUN-4.2-01: FedAvg on German Credit (20 rounds, K=5, Dirichlet 0.3)
2. RUN-4.2-02: TrustFedGNN (FairShare-GNN Ours) on German Credit (20 rounds, K=5, Dirichlet 0.3)
3. RUN-4.2-03: Fast FU-Shapley vs Exact Monte-Carlo Shapley Correlation Probes (K=4, 10 rounds)
4. RUN-4.2-04: FedAvg on Bail Recidivism (20 rounds, K=5, Dirichlet 0.3)
5. RUN-4.2-05: TrustFedGNN (FairShare-GNN Ours) on Bail Recidivism (20 rounds, K=5, Dirichlet 0.3)

Calculates the 5-Metric Suite:
- AUC-ROC
- DPD (Demographic Parity Difference)
- EOD (Equal Opportunity Difference)
- Omega_w (Weight Oscillation / Simplex Stability)
- Divergence Flag
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ExperimentConfig, set_seed
from src.federated import FederatedTrainer
from src.trust.incentive import get_server_target_gradients_pooled
from src.utils.metrics import weight_oscillation
from experiments.fairshare_common import (
    client_pseudo_grads,
    make_trainer,
    pearson_spearman,
    warm_rounds,
)
from experiments.exact_shapley_correlation import exact_shapley


def run_stage4_2():
    print("=" * 70)
    print("🚀 [STAGE 4.2] EXECUTING SYMMETRIC SMALL BENCHMARKS & CORRELATION HARNESS")
    print("=" * 70)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    seed = 42
    results = {}

    # ----------------------------------------------------------------------- #
    # 1. RUN-4.2-01: FedAvg on German Credit (1,000 nodes)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("▶️ [1/5] RUN-4.2-01: Running FedAvg on German Credit (20 rounds)...")
    print("=" * 50)
    set_seed(seed)
    cfg_fedavg_german = ExperimentConfig(
        dataset="german",
        seed=seed,
        num_clients=5,
        rounds=20,
        dirichlet_alpha=0.3,
        model="gat",
        aggregator="fedavg",
        dp_enabled=False,
    )
    trainer_fedavg_german = FederatedTrainer(cfg_fedavg_german)
    res_fedavg_german = trainer_fedavg_german.run(verbose=True)
    fedavg_german_weights = [r.get("agg_weights") for r in res_fedavg_german["history"]]
    omega_w_fedavg_german = weight_oscillation(fedavg_german_weights)

    results["RUN-4.2-01"] = {
        "dataset": "german",
        "method": "fedavg",
        "rounds": 20,
        "final_auc": float(res_fedavg_german["final"]["auc"]),
        "final_dpd": float(res_fedavg_german["final"]["dpd"]),
        "final_eod": float(res_fedavg_german["final"]["eod"]),
        "diverged": float(res_fedavg_german["final"]["diverged"]),
        "weight_oscillation": float(omega_w_fedavg_german),
        "final_weights": res_fedavg_german["history"][-1].get("agg_weights"),
    }
    print(f"[*] RUN-4.2-01 Done: AUC={results['RUN-4.2-01']['final_auc']:.4f}, DPD={results['RUN-4.2-01']['final_dpd']:.4f}, EOD={results['RUN-4.2-01']['final_eod']:.4f}, Omega_w={omega_w_fedavg_german:.4f}")

    # ----------------------------------------------------------------------- #
    # 2. RUN-4.2-02: FairShare-GNN (Ours) on German Credit (1,000 nodes)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("▶️ [2/5] RUN-4.2-02: Running FairShare-GNN (Ours) on German Credit (20 rounds)...")
    print("=" * 50)
    set_seed(seed)
    cfg_ours_german = ExperimentConfig(
        dataset="german",
        seed=seed,
        num_clients=5,
        rounds=20,
        dirichlet_alpha=0.3,
        model="trustfedgnn",
        aggregator="fu_shapley",
        fu_alpha=0.1,
        fu_ema_beta=0.9,
        fu_grad_clip=10.0,
        dp_enabled=False,
    )
    trainer_ours_german = FederatedTrainer(cfg_ours_german)
    res_ours_german = trainer_ours_german.run(verbose=True)
    ours_german_weights = [r.get("agg_weights") for r in res_ours_german["history"]]
    omega_w_ours_german = weight_oscillation(ours_german_weights)

    results["RUN-4.2-02"] = {
        "dataset": "german",
        "method": "fairshare (ours)",
        "rounds": 20,
        "final_auc": float(res_ours_german["final"]["auc"]),
        "final_dpd": float(res_ours_german["final"]["dpd"]),
        "final_eod": float(res_ours_german["final"]["eod"]),
        "diverged": float(res_ours_german["final"]["diverged"]),
        "weight_oscillation": float(omega_w_ours_german),
        "final_weights": res_ours_german["history"][-1].get("agg_weights"),
    }
    print(f"[*] RUN-4.2-02 Done: AUC={results['RUN-4.2-02']['final_auc']:.4f}, DPD={results['RUN-4.2-02']['final_dpd']:.4f}, EOD={results['RUN-4.2-02']['final_eod']:.4f}, Omega_w={omega_w_ours_german:.4f}")

    # ----------------------------------------------------------------------- #
    # 3. RUN-4.2-03: Fast FU-Shapley vs Exact Shapley Correlation on German Credit
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("▶️ [3/5] RUN-4.2-03: Measuring Exact Shapley vs Fast FU-Shapley Correlation...")
    print("=" * 50)
    t_corr = make_trainer(dataset="german", seed=seed, num_clients=4, rounds=10, method="fairshare", alpha=0.1)
    probe_rounds = [2, 4, 6, 8]
    correlations = []
    
    current_round = 0
    for r in probe_rounds:
        step_n = r - current_round
        if step_n > 0:
            warm_rounds(t_corr, step_n)
            current_round = r
        grads = client_pseudo_grads(t_corr)
        phi_exact = exact_shapley(t_corr, grads, alpha=0.1, game="loss")
        tg = get_server_target_gradients_pooled(t_corr.ref_model, t_corr.clients_data, t_corr.device, alpha=0.1)
        g_target = tg[0].cpu()
        phi_fu = [(g.float() @ g_target).item() for g in grads]
        
        pearson_r, spearman_rho = pearson_spearman(phi_exact, phi_fu)
        correlations.append({
            "probe_round": r,
            "pearson_r": float(pearson_r),
            "spearman_rho": float(spearman_rho),
            "phi_exact": [round(float(v), 6) for v in phi_exact],
            "phi_fu": [round(float(v), 6) for v in phi_fu],
        })
        print(f"    Probe Round {r:2d} -> Pearson r = {pearson_r:.4f}, Spearman rho = {spearman_rho:.4f}")

    avg_pearson = np.mean([c["pearson_r"] for c in correlations])
    avg_spearman = np.mean([c["spearman_rho"] for c in correlations])
    results["RUN-4.2-03"] = {
        "dataset": "german",
        "num_clients": 4,
        "probe_rounds": probe_rounds,
        "avg_pearson_r": float(avg_pearson),
        "avg_spearman_rho": float(avg_spearman),
        "probes": correlations,
    }
    print(f"[*] RUN-4.2-03 Done: Average Pearson r = {avg_pearson:.4f}, Average Spearman rho = {avg_spearman:.4f}")

    # ----------------------------------------------------------------------- #
    # 4. RUN-4.2-04: FedAvg on Bail Recidivism (18,876 nodes - Comparative Anchor)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("▶️ [4/5] RUN-4.2-04: Running FedAvg on Bail Recidivism (20 rounds)...")
    print("=" * 50)
    set_seed(seed)
    cfg_fedavg_bail = ExperimentConfig(
        dataset="bail",
        seed=seed,
        num_clients=5,
        rounds=20,
        dirichlet_alpha=0.3,
        model="gat",
        aggregator="fedavg",
        dp_enabled=False,
    )
    trainer_fedavg_bail = FederatedTrainer(cfg_fedavg_bail)
    res_fedavg_bail = trainer_fedavg_bail.run(verbose=True)
    fedavg_bail_weights = [r.get("agg_weights") for r in res_fedavg_bail["history"]]
    omega_w_fedavg_bail = weight_oscillation(fedavg_bail_weights)

    results["RUN-4.2-04"] = {
        "dataset": "bail",
        "method": "fedavg",
        "rounds": 20,
        "final_auc": float(res_fedavg_bail["final"]["auc"]),
        "final_dpd": float(res_fedavg_bail["final"]["dpd"]),
        "final_eod": float(res_fedavg_bail["final"]["eod"]),
        "diverged": float(res_fedavg_bail["final"]["diverged"]),
        "weight_oscillation": float(omega_w_fedavg_bail),
        "final_weights": res_fedavg_bail["history"][-1].get("agg_weights"),
    }
    print(f"[*] RUN-4.2-04 Done: AUC={results['RUN-4.2-04']['final_auc']:.4f}, DPD={results['RUN-4.2-04']['final_dpd']:.4f}, EOD={results['RUN-4.2-04']['final_eod']:.4f}, Omega_w={omega_w_fedavg_bail:.4f}")

    # ----------------------------------------------------------------------- #
    # 5. RUN-4.2-05: FairShare-GNN (Ours) on Bail Recidivism (18,876 nodes)
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 50)
    print("▶️ [5/5] RUN-4.2-05: Running FairShare-GNN (Ours) on Bail Recidivism (20 rounds)...")
    print("=" * 50)
    set_seed(seed)
    cfg_ours_bail = ExperimentConfig(
        dataset="bail",
        seed=seed,
        num_clients=5,
        rounds=20,
        dirichlet_alpha=0.3,
        model="trustfedgnn",
        aggregator="fu_shapley",
        fu_alpha=0.1,
        fu_ema_beta=0.9,
        fu_grad_clip=10.0,
        dp_enabled=False,
    )
    trainer_ours_bail = FederatedTrainer(cfg_ours_bail)
    res_ours_bail = trainer_ours_bail.run(verbose=True)
    ours_bail_weights = [r.get("agg_weights") for r in res_ours_bail["history"]]
    omega_w_ours_bail = weight_oscillation(ours_bail_weights)

    results["RUN-4.2-05"] = {
        "dataset": "bail",
        "method": "fairshare (ours)",
        "rounds": 20,
        "final_auc": float(res_ours_bail["final"]["auc"]),
        "final_dpd": float(res_ours_bail["final"]["dpd"]),
        "final_eod": float(res_ours_bail["final"]["eod"]),
        "diverged": float(res_ours_bail["final"]["diverged"]),
        "weight_oscillation": float(omega_w_ours_bail),
        "final_weights": res_ours_bail["history"][-1].get("agg_weights"),
    }
    print(f"[*] RUN-4.2-05 Done: AUC={results['RUN-4.2-05']['final_auc']:.4f}, DPD={results['RUN-4.2-05']['final_dpd']:.4f}, EOD={results['RUN-4.2-05']['final_eod']:.4f}, Omega_w={omega_w_ours_bail:.4f}")

    # ----------------------------------------------------------------------- #
    # Save Combined Results
    # ----------------------------------------------------------------------- #
    out_path = os.path.join(results_dir, "stage4_2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 70)
    print(f"✅ [STAGE 4.2 COMPLETED] Results saved to {out_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_stage4_2()
