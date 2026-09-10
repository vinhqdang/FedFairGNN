"""Privacy budget epsilon sweep on Bail comparing TrustFedGNN (FTGD) vs DP-FedAvg.

Outputs:
  - results/revision/privacy_bail_sweep.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

import numpy as np

EPSILONS = [1.0, 2.0, 4.0, 8.0, 16.0]
ROUNDS = 15
SEEDS = list(range(42, 52))  # 10 seeds: 42 to 51


def run_sweep():
    ftgd_auc_means = []
    ftgd_auc_stds = []
    ftgd_dpd_means = []
    ftgd_dpd_stds = []
    ftgd_raw = {}

    dp_auc_means = []
    dp_auc_stds = []
    dp_dpd_means = []
    dp_dpd_stds = []
    dp_raw = {}

    t_start = time.time()

    # 1. Run FTGD (TrustFedGNN)
    for eps in EPSILONS:
        print(f"\n[*] Running FTGD at eps={eps} across {len(SEEDS)} seeds...")
        eps_aucs = []
        eps_dpds = []
        ftgd_raw[str(eps)] = []

        for s in SEEDS:
            t0 = time.time()
            cfg = ExperimentConfig.canonical(
                dataset="bail",
                rounds=ROUNDS,
                local_epochs=1,
                seed=s,
                dp_enabled=True,
                dp_mode="ftgd",
                dp_epsilon=eps,
            )
            trainer = FederatedTrainer(cfg)
            res = trainer.run(verbose=False)
            auc_val = float(res["final"]["auc"])
            dpd_val = float(res["final"]["dpd_hard"])
            eps_aucs.append(auc_val)
            eps_dpds.append(dpd_val)
            ftgd_raw[str(eps)].append({
                "seed": s,
                "auc": auc_val,
                "dpd_hard": dpd_val,
                "wall_clock_s": time.time() - t0,
            })
            print(f"    Seed {s}: AUC={auc_val:.4f}, DPD={dpd_val:.4f} ({time.time() - t0:.1f}s)")

        auc_m, auc_s = float(np.mean(eps_aucs)), float(np.std(eps_aucs, ddof=1))
        dpd_m, dpd_s = float(np.mean(eps_dpds)), float(np.std(eps_dpds, ddof=1))
        ftgd_auc_means.append(auc_m)
        ftgd_auc_stds.append(auc_s)
        ftgd_dpd_means.append(dpd_m)
        ftgd_dpd_stds.append(dpd_s)
        print(f"  --> FTGD eps={eps}: AUC={auc_m:.4f} +/- {auc_s:.4f}, DPD={dpd_m:.4f} +/- {dpd_s:.4f}")

    # 2. Run DP-FedAvg (Standard DP-SGD)
    for eps in EPSILONS:
        print(f"\n[*] Running DP-FedAvg at eps={eps} across {len(SEEDS)} seeds...")
        eps_aucs = []
        eps_dpds = []
        dp_raw[str(eps)] = []

        for s in SEEDS:
            t0 = time.time()
            cfg = ExperimentConfig.canonical(
                dataset="bail",
                rounds=ROUNDS,
                local_epochs=1,
                seed=s,
                aggregator="fedavg",
                dp_enabled=True,
                dp_mode="gradient",
                dp_epsilon=eps,
            )
            trainer = FederatedTrainer(cfg)
            res = trainer.run(verbose=False)
            auc_val = float(res["final"]["auc"])
            dpd_val = float(res["final"]["dpd_hard"])
            eps_aucs.append(auc_val)
            eps_dpds.append(dpd_val)
            dp_raw[str(eps)].append({
                "seed": s,
                "auc": auc_val,
                "dpd_hard": dpd_val,
                "wall_clock_s": time.time() - t0,
            })
            print(f"    Seed {s}: AUC={auc_val:.4f}, DPD={dpd_val:.4f} ({time.time() - t0:.1f}s)")

        auc_m, auc_s = float(np.mean(eps_aucs)), float(np.std(eps_aucs, ddof=1))
        dpd_m, dpd_s = float(np.mean(eps_dpds)), float(np.std(eps_dpds, ddof=1))
        dp_auc_means.append(auc_m)
        dp_auc_stds.append(auc_s)
        dp_dpd_means.append(dpd_m)
        dp_dpd_stds.append(dpd_s)
        print(f"  --> DP-FedAvg eps={eps}: AUC={auc_m:.4f} +/- {auc_s:.4f}, DPD={dpd_m:.4f} +/- {dpd_s:.4f}")

    total_time = time.time() - t_start

    data = {
        "manifest": build_manifest(extra={
            "experiment": "privacy_bail_sweep",
            "dataset": "bail",
            "rounds": ROUNDS,
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "wall_clock_s": total_time,
        }),
        "epsilons": EPSILONS,
        "ftgd": {
            "auc": ftgd_auc_means,
            "auc_std": ftgd_auc_stds,
            "dpd": ftgd_dpd_means,
            "dpd_std": ftgd_dpd_stds,
            "raw_per_seed": ftgd_raw,
        },
        "dp_fedavg": {
            "auc": dp_auc_means,
            "auc_std": dp_auc_stds,
            "dpd": dp_dpd_means,
            "dpd_std": dp_dpd_stds,
            "raw_per_seed": dp_raw,
        },
    }

    out_dir = "results/revision"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "privacy_bail_sweep.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[+] Successfully saved {out_path} in {total_time:.1f}s ({total_time/60:.2f} min).")


if __name__ == "__main__":
    run_sweep()
