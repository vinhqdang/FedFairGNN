"""Per-dataset end-to-end Differential Privacy accounting table (Issue I1, Stanford Q1, GLM M2).

Uses Renyi Differential Privacy (RDP) via src/trust/privacy.py to compose privacy spend
over federated training rounds R and local epochs E for each dataset:
  - German Credit (1,000 nodes)
  - Bail Recidivism (18,876 nodes)
  - Credit Default (30,000 nodes)
  - Pokec-z (67,796 nodes)
  - Elliptic Bitcoin (203,769 nodes)
  - ogbn-products (2,449,029 nodes)

Outputs:
  - results/revision/dp_accounting.json
  - manuscript/tables/revision/dp_accounting.tex
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.trust.privacy import calibrate_noise_multiplier, compute_epsilon, gaussian_rdp, rdp_to_dp


DATASET_SPECS = [
    {
        "name": "German",
        "nodes": 1000,
        "clients": 10,
        "rounds": 20,
        "local_epochs": 3,
        "sampling": False,
        "delta": 1e-4,
        "target_eps": 8.0,
    },
    {
        "name": "Bail",
        "nodes": 18876,
        "clients": 10,
        "rounds": 50,
        "local_epochs": 3,
        "sampling": False,
        "delta": 1e-5,
        "target_eps": 8.0,
    },
    {
        "name": "Credit",
        "nodes": 30000,
        "clients": 10,
        "rounds": 50,
        "local_epochs": 3,
        "sampling": False,
        "delta": 1e-5,
        "target_eps": 8.0,
    },
    {
        "name": "Pokec-z",
        "nodes": 67796,
        "clients": 10,
        "rounds": 100,
        "local_epochs": 3,
        "sampling": False,
        "delta": 1e-5,
        "target_eps": 8.0,
    },
    {
        "name": "Elliptic",
        "nodes": 203769,
        "clients": 10,
        "rounds": 30,
        "local_epochs": 3,
        "sampling": False,
        "delta": 1e-5,
        "target_eps": 8.0,
    },
    {
        "name": "ogbn-products",
        "nodes": 2449029,
        "clients": 10,
        "rounds": 20,
        "local_epochs": 1,
        "sampling": True,
        "batch_size": 1024,
        "delta": 1e-6,
        "target_eps": 8.0,
    },
]


def generate_dp_accounting(out_json="results/revision/dp_accounting.json",
                           out_tex="manuscript/tables/revision/dp_accounting.tex"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    records = []

    for spec in DATASET_SPECS:
        ds_name = spec["name"]
        K = spec["clients"]
        R = spec["rounds"]
        E = spec["local_epochs"]
        delta = spec["delta"]
        target_eps = spec["target_eps"]

        if spec["sampling"]:
            # Approximate batches per client
            nodes_per_client = spec["nodes"] / K
            batches_per_epoch = max(1, int(math.ceil(nodes_per_client / spec["batch_size"])))
            total_releases = R * E * batches_per_epoch
        else:
            total_releases = R * E

        # Calibrate noise multiplier z for the total release count to hit target_eps
        z_calibrated = calibrate_noise_multiplier(target_eps, total_releases, delta=delta)
        composed_eps = compute_epsilon(z_calibrated, total_releases, delta=delta)

        # Also compute what epsilon would be if fixed z = 1.0 or z = 1.5 was used
        eps_z1 = compute_epsilon(1.0, total_releases, delta=delta)
        eps_z15 = compute_epsilon(1.5, total_releases, delta=delta)

        records.append({
            "dataset": ds_name,
            "nodes": spec["nodes"],
            "clients": K,
            "rounds": R,
            "local_epochs": E,
            "total_releases": total_releases,
            "delta": delta,
            "calibrated_z": float(z_calibrated),
            "composed_epsilon": float(composed_eps),
            "eps_at_z1": float(eps_z1),
            "eps_at_z15": float(eps_z15),
        })

    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[+] Saved DP accounting data to {out_json}")

    # Format publication-ready LaTeX table
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{End-to-End Differential Privacy Accounting across Benchmark Datasets.}",
        "R\\'{e}nyi Differential Privacy (RDP) composition for FTGD statistic releases across $R$ rounds and $E$ local epochs.",
        "Noise multiplier $z = \\sigma / C$ is calibrated via binary search over RDP orders $\\alpha > 1$ to guarantee total spend $\\le \\epsilon_{\\text{target}} = 8.0$ at failure probability $\\delta$.}",
        "\\label{tab:dp_accounting}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Nodes ($N$)} & \\textbf{Clients ($K$)} & \\textbf{Rounds ($R$)} & \\textbf{Epochs ($E$)} & \\textbf{Releases ($T$)} & \\textbf{Target $\\delta$} & \\textbf{Multiplier $z$} & \\textbf{Composed $\\epsilon$} \\\\",
        "\\midrule",
    ]

    for r in records:
        node_str = f"{r['nodes']:,}"
        delta_str = f"$10^{{{int(math.log10(r['delta']))}}}$"
        line = (
            f"{r['dataset']} & {node_str} & {r['clients']} & {r['rounds']} & {r['local_epochs']} & "
            f"{r['total_releases']} & {delta_str} & {r['calibrated_z']:.2f} & {r['composed_epsilon']:.2f} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX DP accounting table to {out_tex}")


if __name__ == "__main__":
    generate_dp_accounting()
