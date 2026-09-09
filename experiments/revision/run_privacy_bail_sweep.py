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

EPSILONS = [1.0, 2.0, 4.0, 8.0, 16.0]
ROUNDS = 15
SEED = 42

def run_sweep():
    ftgd_auc = []
    ftgd_dpd = []
    dp_auc = []
    dp_dpd = []

    t_start = time.time()

    # 1. Run FTGD (TrustFedGNN)
    for eps in EPSILONS:
        print(f"[*] Running FTGD at eps={eps}...")
        cfg = ExperimentConfig.canonical(
            dataset="bail",
            rounds=ROUNDS,
            local_epochs=1,
            seed=SEED,
            dp_enabled=True,
            dp_mode="ftgd",
            dp_epsilon=eps,
        )
        trainer = FederatedTrainer(cfg)
        res = trainer.run()
        ftgd_auc.append(float(res["final"]["auc"]))
        ftgd_dpd.append(float(res["final"]["dpd_hard"]))
        print(f"    FTGD eps={eps}: AUC={res['final']['auc']:.4f}, DPD={res['final']['dpd_hard']:.4f}")

    # 2. Run DP-FedAvg (Standard DP-SGD)
    for eps in EPSILONS:
        print(f"[*] Running DP-FedAvg at eps={eps}...")
        cfg = ExperimentConfig.canonical(
            dataset="bail",
            rounds=ROUNDS,
            local_epochs=1,
            seed=SEED,
            aggregator="fedavg",
            dp_enabled=True,
            dp_mode="gradient",
            dp_epsilon=eps,
        )
        trainer = FederatedTrainer(cfg)
        res = trainer.run()
        dp_auc.append(float(res["final"]["auc"]))
        dp_dpd.append(float(res["final"]["dpd_hard"]))
        print(f"    DP-FedAvg eps={eps}: AUC={res['final']['auc']:.4f}, DPD={res['final']['dpd_hard']:.4f}")

    total_time = time.time() - t_start

    data = {
        "manifest": build_manifest(extra={
            "experiment": "privacy_bail_sweep",
            "dataset": "bail",
            "rounds": ROUNDS,
            "seed": SEED,
            "wall_clock_s": total_time,
        }),
        "epsilons": EPSILONS,
        "ftgd": {
            "auc": ftgd_auc,
            "dpd": ftgd_dpd,
        },
        "dp_fedavg": {
            "auc": dp_auc,
            "dpd": dp_dpd,
        },
    }

    out_dir = "results/revision"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "privacy_bail_sweep.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[+] Successfully saved {out_path} in {total_time:.1f}s.")

if __name__ == "__main__":
    run_sweep()
