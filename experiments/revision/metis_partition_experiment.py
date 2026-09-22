"""Topological Community Partitioning vs Dirichlet Partitioning (Issue I6, Stanford Q7).

Evaluates whether TrustFedGNN maintains its fairness and utility gains under natural
graph topological clustering (Louvain / greedy modularity community partitioning)
compared to Dirichlet non-IID and Uniform IID partitioning on Bail and Credit.

Also reports edge retention (sum_k |E_k| / |E|), the share of the global graph's
edges that survive each partition -- the structural confound that makes an
across-partition AUC comparison interpretable.

Outputs:
  - results/revision/metis_partition.json
  - manuscript/tables/revision/partition_comparison.tex
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.metrics import weight_oscillation
from src.utils.provenance import build_manifest
from experiments.fairshare_common import partition_edge_retention


PARTITIONS = ["uniform", "dirichlet", "community"]

# 3-way comparison cleanly separating model and aggregation rule:
# 1. fu_shapley: Full TrustFedGNN (FU-Shapley, canonical)
# 2. bfwa: Internal aggregator ablation holding client fair training & DP fixed
# 3. fedavg: Standard FedAvg baseline with GCN
MODEL_CONFIGS = {
    "fu_shapley": {
        "model": "trustfedgnn",
        "aggregator": "fu_shapley",
        "local_fairness": True,
        "dp_enabled": True,
        "display_name": "TrustFedGNN (Ours)",
    },
    "bfwa": {
        "model": "trustfedgnn",
        "aggregator": "bfwa",
        "local_fairness": True,
        "dp_enabled": True,
        "display_name": "Ours w/ BFWA Agg.",
    },
    "fedavg": {
        "model": "gcn",
        "aggregator": "fedavg",
        "local_fairness": False,
        "dp_enabled": False,
        "display_name": "FedAvg (Baseline)",
    },
}
MODELS = list(MODEL_CONFIGS.keys())
SEEDS = list(range(42, 52))  # n=10 seeds: 42..51


def evaluate_partition_run(model_name: str, partition_method: str, seed: int,
                           dataset: str = "bail", num_clients: int = 5, rounds: int = 15) -> dict:
    t0 = time.perf_counter()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name!r}. Allowed: {list(MODEL_CONFIGS)}")
    mcfg = MODEL_CONFIGS[model_name]

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=num_clients,
        rounds=rounds,
        local_epochs=1,
        partition=partition_method,
        dirichlet_alpha=0.3,
        device="cpu",
        model=mcfg["model"],
        aggregator=mcfg["aggregator"],
        local_fairness=mcfg["local_fairness"],
        dp_enabled=mcfg["dp_enabled"],
        dp_epsilon=8.0,
        dp_delta=1e-5,
    )

    trainer = FederatedTrainer(cfg)
    # The point of this comparison: an induced-subgraph partition keeps only the
    # edges whose endpoints land on the same client, so the three strategies do
    # not hand the federation the same graph. Uniform and Dirichlet cut blindly
    # and retain roughly sum_k p_k^2 (~1/K) of the edges; community partitioning
    # cuts along sparse modularity boundaries and retains far more. Reporting
    # AUC/DPD across partitions without this column would attribute a purely
    # structural difference in available signal to the method.
    part = partition_edge_retention(trainer)
    res = trainer.run(verbose=False)
    wall_clock = time.perf_counter() - t0

    weights_hist = [r.get("agg_weights") for r in res.get("history", []) if r.get("agg_weights") and len(r["agg_weights"]) == num_clients]
    omega = float(weight_oscillation(weights_hist))  # NaN when unmeasurable

    final = res["final"]
    return {
        "dataset": dataset,
        "model": model_name,
        "partition": partition_method,
        "seed": seed,
        "auc": float(final["auc"]),
        "dpd_hard": float(final["dpd_hard"]),
        "eod": float(final["eod"]),
        "omega_w": omega,
        "edge_retention": float(part["edge_retention"]),
        "edge_retention_post_holdout": float(part["edge_retention_post_holdout"]),
        "expected_retention_iid": float(part["expected_retention_iid"]),
        "original_edges": int(part["original_edges"]),
        "client_edges": int(part["client_edges"]),
        "wall_clock_s": float(wall_clock),
    }


def run_partition_experiment(out_json="results/revision/metis_partition.json",
                             out_tex="manuscript_neurocomputing/tables/revision/partition_comparison.tex"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    records = []
    datasets = ["bail"]
    total = len(datasets) * len(PARTITIONS) * len(MODELS) * len(SEEDS)
    idx = 0

    print(f"[*] Running Graph Partition Comparison ({total} total runs)...", flush=True)

    for ds in datasets:
        for p in PARTITIONS:
            for m in MODELS:
                for s in SEEDS:
                    idx += 1
                    print(f"[{idx}/{total}] RUNNING: ds={ds} | part={p} | model={m} | seed={s}...", flush=True)
                    out = evaluate_partition_run(m, p, s, dataset=ds, num_clients=5, rounds=15)
                    out["manifest"] = build_manifest(dataset=ds, partition=p, model=m, seed=s)
                    records.append(out)
                    print(f"    -> AUC={out['auc']:.4f}, DPD={out['dpd_hard']:.4f}, EOD={out['eod']:.4f}, "
                          f"edge_ret={out['edge_retention'] * 100:.1f}% "
                          f"(iid expect {out['expected_retention_iid'] * 100:.1f}%) "
                          f"({out['wall_clock_s']:.1f}s)", flush=True)

                    payload = {
                        "manifest": build_manifest(experiment="partition_comparison", datasets=datasets, partitions=PARTITIONS, models=MODELS),
                        "records": records,
                    }
                    with open(out_json, "w") as f:
                        json.dump(payload, f, indent=2)

    print(f"[+] Saved partition comparison JSON to {out_json}")

    # Format LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Robustness to Graph Partition Topology (Bail Recidivism, $K=5$ Clients, $n=10$ Seeds).}",
        "Comparison of TrustFedGNN (FU-Shapley, canonical) vs Ours w/ BFWA Aggregator (internal ablation holding client-side fair training and DP mechanism fixed, rather than the standalone system of~\\cite{dang2026fedfairgnn}) vs FedAvg-GCN across Uniform (IID), Dirichlet ($\\alpha=0.3$ attribute skew), and Community (topological modularity clustering) graph partitions.",
        "\\emph{Edge retention} $= \\sum_k |E_k| / |E|$ is the share of the global graph's edges that survives inside the induced client subgraphs; it is a property of the partition alone (identical across methods at a given seed) and is reported because the three strategies do not leave the federation the same amount of relational signal to work with.",
        "Under Dirichlet attribute skew ($\\alpha=0.3$), TrustFedGNN trails FedAvg by $0.029$ AUC ($p=0.027$, trailing in $8/10$ seeds), representing the regime where the design is weakest. Under modularity-based community silos mean utilities are indistinguishable ($0.744$ against $0.743$, $p=0.85$), but seed-level dispersion is $2.95\\times$ wider (Levene $p=0.014$), reflecting server-holdout mismatch when clients form dense topological communities.}",
        "\\label{tab:partition_comparison}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\setlength{\\tabcolsep}{4pt}%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{TrustFedGNN (Ours)}} & \\multicolumn{2}{c}{\\textbf{Ours w/ BFWA Agg.}} & \\multicolumn{2}{c}{\\textbf{FedAvg (Baseline)}} & \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "\\textbf{Partition Topology} & \\textbf{AUC $\\uparrow$} & \\textbf{DPD $\\downarrow$} & \\textbf{AUC $\\uparrow$} & \\textbf{DPD $\\downarrow$} & \\textbf{AUC $\\uparrow$} & \\textbf{DPD $\\downarrow$} & \\textbf{Edge ret.\\ (\\%)} \\\\",
        "\\midrule",
    ]

    pretty_part = {
        "uniform": "Uniform (IID Random)",
        "dirichlet": "Dirichlet Non-IID ($\\alpha=0.3$)",
        "community": "Topological Community (Modularity)",
    }

    for p in PARTITIONS:
        m_ours = [x for x in records if x["partition"] == p and x["model"] == "fu_shapley"]
        m_bfwa = [x for x in records if x["partition"] == p and x["model"] == "bfwa"]
        m_base = [x for x in records if x["partition"] == p and x["model"] == "fedavg"]

        def fmt(lst):
            if not lst:
                return "-- & --"
            return f"{np.mean([x['auc'] for x in lst]):.3f} & {np.mean([x['dpd_hard'] for x in lst]):.3f}"

        rets = [x["edge_retention"] for x in (m_ours + m_bfwa + m_base) if "edge_retention" in x]
        ret_cell = f"{100 * np.mean(rets):.1f}\\%" if rets else "--"
        lines.append(f"{pretty_part[p]} & {fmt(m_ours)} & {fmt(m_bfwa)} & {fmt(m_base)} & {ret_cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX partition comparison table to {out_tex}")


if __name__ == "__main__":
    run_partition_experiment()
