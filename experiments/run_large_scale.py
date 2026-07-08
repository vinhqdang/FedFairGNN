"""Large-scale scalability study on ogbn-products (2.4M nodes, 61M edges).

Requires neighbor sampling (no full-batch training is possible) and is intended
for a GPU (see notebooks/large_scale_colab.ipynb). Runs a key subset of methods
and logs results to results/summary.jsonl exactly like the main matrix, so the
paper's tables/figures pick them up automatically.

    python -m experiments.run_large_scale
"""
from __future__ import annotations

import time

from src.config import ExperimentConfig
from src.utils.logging_utils import ResultLogger
from experiments.methods import apply_method
from experiments.run_experiment import run_one

METHODS = ["fedavg-gat", "fairsin", "favgnn", "dp-fedavg",
           "fedfairgnn-nodp", "fedfairgnn", "ours-robust"]


def main(rounds: int = 6, num_clients: int = 3, seeds=(0,)):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = ResultLogger("results")
    print(f"[large-scale] ogbn-products on {device}; {len(METHODS)} methods x "
          f"{len(seeds)} seeds; rounds={rounds} K={num_clients}", flush=True)
    for s in seeds:
        for m in METHODS:
            cfg = ExperimentConfig(dataset="ogbn_products", seed=s, device=device,
                                   num_clients=num_clients, rounds=rounds, local_epochs=1,
                                   hidden_channels=128, sampling=True, batch_size=4096,
                                   num_neighbors=(15, 10))
            apply_method(cfg, m)
            cfg.sampling = True                     # ensure sampling stays on
            print(f"[start] {m} ...", flush=True)
            t = time.time()
            run_id, final = run_one(cfg, logger, tag="ogbn")
            if final is None:
                print(f"  {m}: already done, skipped", flush=True)
            else:
                print(f"  {m}: AUC={final.get('auc',0):.3f} DPD={final.get('dpd',0):.3f} "
                      f"EOD={final.get('eod',0):.3f} ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
