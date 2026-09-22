"""Merge the standalone FLTrust GPU sweep into the canonical SOTA raw-run files.

`results/revision/fltrust_sota_results.json` is produced by the dedicated
Colab GPU campaign (10 seeds, GAT backbone, K=10/R=50). This script folds its
records into `results/sota_pokecz.json` and `results/sota_credit.json`:
1. Populates `raw_runs["fltrust"]` with per-seed metrics.
2. Populates `baselines["fltrust"]` with summary statistics and per-seed records.
3. Updates `manifest["baselines"]` list.
4. Updates manifest provenance (`git_commit`, `git_dirty`, `fltrust_source`)
   so that the composite artifact transparently documents its exact provenance.
"""
from __future__ import annotations

import json
import numpy as np

FLTRUST_SRC = "results/revision/fltrust_sota_results.json"


def build_baseline_entry(per_seed: list[dict]) -> dict:
    aucs = [r["auc"] for r in per_seed]
    dpd_softs = [r.get("dpd_soft", 0.0) for r in per_seed]
    dpd_hards = [r["dpd_hard"] for r in per_seed]
    eods = [r["eod"] for r in per_seed]
    omegas = [r.get("omega_w", 0.0) for r in per_seed]
    walls = [r.get("wall_clock_s", 0.0) for r in per_seed]

    summary = {
        "auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs, ddof=1)),
            "min": float(np.min(aucs)),
            "max": float(np.max(aucs)),
        },
        "dpd_soft": {
            "mean": float(np.mean(dpd_softs)),
            "std": float(np.std(dpd_softs, ddof=1)),
            "min": float(np.min(dpd_softs)),
            "max": float(np.max(dpd_softs)),
        },
        "dpd_hard": {
            "mean": float(np.mean(dpd_hards)),
            "std": float(np.std(dpd_hards, ddof=1)),
            "min": float(np.min(dpd_hards)),
            "max": float(np.max(dpd_hards)),
        },
        "eod": {
            "mean": float(np.mean(eods)),
            "std": float(np.std(eods, ddof=1)),
            "min": float(np.min(eods)),
            "max": float(np.max(eods)),
        },
        "omega_w": {
            "mean": float(np.mean(omegas)),
            "std": float(np.std(omegas, ddof=1)) if len(omegas) > 1 else 0.0,
            "min": float(np.min(omegas)),
            "max": float(np.max(omegas)),
        },
        "wall_clock_s": {
            "mean": float(np.mean(walls)),
            "std": float(np.std(walls, ddof=1)) if len(walls) > 1 else 0.0,
            "min": float(np.min(walls)),
            "max": float(np.max(walls)),
        },
    }

    per_seed_entry = [
        {
            "method": "fltrust",
            "seed": r["seed"],
            "auc": r["auc"],
            "dpd_soft": r.get("dpd_soft", 0.0),
            "dpd_hard": r["dpd_hard"],
            "eod": r["eod"],
            "omega_w": r.get("omega_w", 0.0),
            "wall_clock_s": r.get("wall_clock_s", 0.0),
        }
        for r in per_seed
    ]
    return {"summary": summary, "per_seed": per_seed_entry}


def merge(sota_path: str, dataset_key: str) -> None:
    with open(sota_path) as f:
        sota = json.load(f)
    with open(FLTRUST_SRC) as f:
        flt = json.load(f)

    per_seed = flt["results"][dataset_key]["per_seed"]
    raw_runs_entry = [
        {
            "seed": r["seed"],
            "auc": r["auc"],
            "dpd_soft": r["dpd_soft"],
            "dpd_hard": r["dpd_hard"],
            "eod": r["eod"],
            "omega_w": r["omega_w"],
        }
        for r in per_seed
    ]

    # 1. Raw runs entry
    sota["raw_runs"]["fltrust"] = raw_runs_entry

    # 2. Baselines dictionary entry
    if "baselines" not in sota:
        sota["baselines"] = {}
    sota["baselines"]["fltrust"] = build_baseline_entry(per_seed)

    # 3. Manifest baselines list
    if "fltrust" not in sota["manifest"]["baselines"]:
        sota["manifest"]["baselines"].append("fltrust")

    # 4. Manifest provenance update
    sota["manifest"]["git_commit_prior_baselines"] = sota["manifest"].get("git_commit")
    sota["manifest"]["git_commit"] = flt["manifest"]["git_commit"]
    sota["manifest"]["git_dirty"] = flt["manifest"]["git_dirty"]
    sota["manifest"]["last_merged_at"] = flt["manifest"]["timestamp"]
    sota["manifest"]["fltrust_manifest"] = flt["manifest"]
    sota["manifest"]["fltrust_source"] = {
        "artifact": FLTRUST_SRC,
        "git_commit": flt["manifest"]["git_commit"],
        "git_dirty": flt["manifest"]["git_dirty"],
        "device": flt["manifest"]["device"],
        "timestamp": flt["manifest"]["timestamp"],
    }

    with open(sota_path, "w") as f:
        json.dump(sota, f, indent=2)
        f.write("\n")

    print(f"{sota_path}: merged fltrust, {len(raw_runs_entry)} seeds, "
          f"baselines list = {sota['manifest']['baselines']}, "
          f"baselines dict keys = {list(sota['baselines'].keys())}, "
          f"manifest git_commit = {sota['manifest']['git_commit']}")


if __name__ == "__main__":
    merge("results/sota_pokecz.json", "pokec_z")
    merge("results/sota_credit.json", "credit")
