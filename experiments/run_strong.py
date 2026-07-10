"""Strengthened evaluation matrix for the manuscript revision.

Goals (all real runs, resumable -- completed run_ids are skipped):
  1. More seeds for stable estimates + significance tests (German is tiny and
     high-variance, so we take 10 seeds there; 5 on the mid-size datasets).
  2. Fill the missing 2025/2026 baseline cells on Credit / Pokec / Elliptic
     (previously only German+Bail), so the comparison is complete.
  3. Run trustfedgnn-plus (server-side global-DPD calibration) on every dataset
     -- the variant that acts on *global* model fairness rather than the BFWA
     surrogate.

Cheap datasets run first so the core comparison lands early. Launch in the
background; it streams to results/summary.jsonl.

    python -m experiments.run_strong
"""
from __future__ import annotations

import time
import traceback

from src.config import ExperimentConfig
from src.utils.logging_utils import ResultLogger
from experiments.methods import apply_method, FAIR_BASELINES
from experiments.run_experiment import run_one

ROUNDS = {"german": 60, "bail": 50, "credit": 40, "elliptic": 25, "pokec_z": 30}
CLIENTS = {"german": 3, "bail": 5, "credit": 5, "elliptic": 10, "pokec_z": 10}

# expanded seed budget (was 3/2); German cheap+noisy so 10 seeds
SEEDS = {"german": list(range(10)), "bail": list(range(5)),
         "credit": list(range(5)), "pokec_z": list(range(5)),
         "elliptic": list(range(3))}

MAIN = FAIR_BASELINES + ["dp-fedavg", "fedfairgnn-nodp", "fedfairgnn",
                         "ours-robust", "trustfedgnn-plus"]
COMP = ["favgnn", "fdp-fair", "fairgfl", "fedgraphfair", "puffle",
        "fedfact", "popets-fairfed"]
ALLM = MAIN + COMP


def cfg_for(method, dataset, seed, **ov):
    c = ExperimentConfig(dataset=dataset, seed=seed,
                         rounds=ROUNDS.get(dataset, 40),
                         num_clients=CLIENTS.get(dataset, 5),
                         local_epochs=2, hidden_channels=64)
    apply_method(c, method)
    for k, v in ov.items():
        setattr(c, k, v)
    return c


def jobs():
    # Phase 1: German -- full coverage, 10 seeds (cheap: ~8s/run)
    for m in ALLM:
        for s in SEEDS["german"]:
            yield cfg_for(m, "german", s)
    # Phase 2: trustfedgnn-plus on the remaining datasets (the new variant)
    for ds in ["bail", "pokec_z", "credit", "elliptic"]:
        for s in SEEDS[ds]:
            yield cfg_for("trustfedgnn-plus", ds, s)
    # Phase 3: fill missing 2025/2026 baseline cells on Credit + Pokec
    for ds in ["credit", "pokec_z"]:
        for m in COMP:
            for s in SEEDS[ds][:3]:
                yield cfg_for(m, ds, s)
    # Phase 4: extend main methods to 5 seeds on Bail / Credit / Pokec
    for ds in ["bail", "pokec_z", "credit"]:
        for m in MAIN:
            for s in SEEDS[ds]:
                yield cfg_for(m, ds, s)
    # Phase 5: Elliptic -- key methods + the newer competitors
    for m in ["fedavg-gat", "fairsin", "fedfairgnn-nodp", "fedfairgnn",
              "ours-robust", "favgnn", "fdp-fair"]:
        for s in SEEDS["elliptic"]:
            yield cfg_for(m, "elliptic", s)


def main():
    logger = ResultLogger("results")
    done = ran = failed = 0
    t0 = time.time()
    for cfg in jobs():
        rid = (f"{cfg.exp_name}__{cfg.model}__{cfg.dataset}__{cfg.aggregator}"
               f"__b0__eps{cfg.dp_epsilon if cfg.dp_enabled else 'inf'}"
               f"__lam{cfg.fairness_weight}__K{cfg.num_clients}__s{cfg.seed}")
        # run_one builds its own run_id; we just call it and let it skip.
        try:
            run_id, res = run_one(cfg, logger)
            if res is None:
                done += 1
            else:
                ran += 1
                print(f"[{ran+done}] {run_id}  AUC={res['auc']:.3f} "
                      f"DPD={res['dpd']:.3f} EOD={res['eod']:.3f} "
                      f"({res.get('wall_s','?')}s)  [{(time.time()-t0)/60:.1f}m]",
                      flush=True)
        except Exception:
            failed += 1
            print(f"[FAIL] {cfg.exp_name}/{cfg.dataset}/s{cfg.seed}", flush=True)
            traceback.print_exc()
    print(f"\n[done] ran={ran} skipped={done} failed={failed} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
