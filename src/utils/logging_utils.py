"""Run logging: every experiment writes a self-describing JSON record so the
manuscript's tables and figures are regenerated from disk, never hand-authored.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional


def _default(o):
    try:
        import numpy as np
        import torch
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except Exception:
        pass
    return str(o)


class ResultLogger:
    def __init__(self, out_dir: str = "results"):
        self.out_dir = out_dir
        self.runs_dir = os.path.join(out_dir, "runs")
        self.summary_path = os.path.join(out_dir, "summary.jsonl")
        os.makedirs(self.runs_dir, exist_ok=True)

    def exists(self, run_id: str) -> bool:
        return os.path.exists(os.path.join(self.runs_dir, f"{run_id}.json"))

    def save(self, run_id: str, config: Dict, result: Dict) -> None:
        record = {"run_id": run_id, "config": config,
                  "final": result.get("final", {}),
                  "history": result.get("history", []),
                  "partition_stats": result.get("partition_stats", []),
                  "byzantine_ids": result.get("byzantine_ids", [])}
        with open(os.path.join(self.runs_dir, f"{run_id}.json"), "w") as f:
            json.dump(record, f, default=_default)
        # Every field that distinguishes one protocol from another belongs here.
        # Without the second group, a summary line cannot tell whether a run used
        # dirichlet_alpha 0.3 or 0.5, a pooled or a held-out scoring set, or 20 vs
        # 60 rounds -- so two incompatible protocols read as one in the tables.
        summary = {"run_id": run_id, **{k: config.get(k) for k in
                   ("exp_name", "model", "dataset", "aggregator", "seed",
                    "attack", "num_byzantine", "dp_epsilon", "dp_enabled",
                    "fairness_weight", "num_clients",
                    # protocol provenance -- do not drop
                    "dirichlet_alpha", "partition", "rounds", "local_epochs",
                    "dp_mode", "fu_val_source", "fu_score", "fu_alpha",
                    "fu_ema_beta", "fser_mode", "fairness_budget",
                    "fw_iterations", "dual_step_size", "krum_f", "sampling")},
                   **{f"final_{k}": v for k, v in result.get("final", {}).items()}}
        with open(self.summary_path, "a") as f:
            f.write(json.dumps(summary, default=_default) + "\n")

    def load_summary(self):
        rows = []
        if os.path.exists(self.summary_path):
            with open(self.summary_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows

    def done_run_ids(self) -> set:
        return {r["run_id"] for r in self.load_summary()}
