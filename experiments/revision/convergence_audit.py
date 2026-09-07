"""Empirical Convergence Audit for FU-Shapley (Stage S4 - Step 4.8).

Tracks round-by-round convergence of FU-Shapley vs baselines (FedAvg, BFWA)
recording Loss, AUC, DPD, EOD, and Weight Oscillation Omega_w over rounds.
Fulfills documentation requirements for docs/02 Limitation #1 (Empirical Convergence Certificate).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest
from experiments.methods import METHODS


def run_convergence_suite(
    dataset: str = "german",
    seed: int = 42,
    rounds: int = 25,
    out_dir: str = "results/fairshare",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "convergence_empirical.json")

    arms = [
        ("fu_shapley", dict(aggregator="fu_shapley")),
        ("bfwa", dict(aggregator="bfwa")),
        ("fedavg", dict(aggregator="fedavg", model="gat", local_fairness=False)),
    ]

    manifest = build_manifest(
        script="experiments/revision/convergence_audit.py",
        dataset=dataset,
        seed=seed,
        rounds=rounds,
    )

    data: Dict[str, Any] = {
        "_manifest": manifest,
        "dataset": dataset,
        "seed": seed,
        "rounds": rounds,
        "methods": {},
    }

    for label, overrides in arms:
        cfg = ExperimentConfig.canonical(
            dataset=dataset,
            seed=seed,
            rounds=rounds,
            **overrides,
        )
        trainer = FederatedTrainer(cfg)
        out = trainer.run(verbose=False)
        history = out.get("history", [])

        rounds_data: List[Dict[str, Any]] = []
        prev_w = None
        omega_w_cum = 0.0

        for h in history:
            w = h.get("agg_weights")
            w_list = [round(float(x), 4) for x in w] if w is not None else []
            omega_w_round = 0.0
            if prev_w is not None and w_list and len(prev_w) == len(w_list):
                omega_w_round = float(0.5 * sum(abs(a - b) for a, b in zip(w_list, prev_w)))
            if w_list:
                prev_w = w_list
            omega_w_cum += omega_w_round

            rounds_data.append({
                "round": h.get("round"),
                "auc": round(float(h.get("g_auc", 0.0)), 4),
                "dpd_soft": round(float(h.get("g_dpd_soft", 0.0)), 4),
                "dpd_hard": round(float(h.get("g_dpd_hard", 0.0)), 4),
                "eod": round(float(h.get("g_eod", 0.0)), 4),
                "weights": w_list,
                "omega_w_round": round(omega_w_round, 4),
                "omega_w_cum": round(omega_w_cum, 4),
            })

        final_metrics = {
            "auc": round(float(out["final"].get("auc", 0.0)), 4),
            "dpd_soft": round(float(out["final"].get("dpd_soft", 0.0)), 4),
            "dpd_hard": round(float(out["final"].get("dpd_hard", 0.0)), 4),
            "eod": round(float(out["final"].get("eod", 0.0)), 4),
            "omega_w_total": round(omega_w_cum, 4),
        }

        data["methods"][label] = {
            "final": final_metrics,
            "rounds": rounds_data,
        }

    finals = {k: tuple(v["final"].values()) for k, v in data["methods"].items()}
    assert len(set(finals.values())) == len(finals), \
        f"Hai nhánh cho kết quả trùng khít -> override không có hiệu lực: {finals}"

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Convergence audit successfully saved to {out_path}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="german")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--out_dir", default="results/fairshare")
    args = parser.parse_args()

    run_convergence_suite(
        dataset=args.dataset,
        seed=args.seed,
        rounds=args.rounds,
        out_dir=args.out_dir,
    )
