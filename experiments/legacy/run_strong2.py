"""Lean, reordered follow-up to run_strong: prioritise Bail to 5 seeds (the
showcase dataset, needed to populate its significance column) and Pokec, and
skip the prohibitively slow Credit 5-seed/comp expansion (Credit is ~700s/run;
it stays at 2 seeds, noted honestly). Resumable -- completed run_ids skip.

    python -m experiments.run_strong2
"""
from __future__ import annotations
import time, traceback
from src.config import ExperimentConfig
from src.utils.logging_utils import ResultLogger
from experiments.methods import apply_method, FAIR_BASELINES
from experiments.run_experiment import run_one

ROUNDS = {"bail": 50, "pokec_z": 30}
CLIENTS = {"bail": 5, "pokec_z": 10}
MAIN = FAIR_BASELINES + ["dp-fedavg", "fedfairgnn-nodp", "fedfairgnn", "ours-robust"]
COMP = ["favgnn", "fdp-fair", "fairgfl", "fedgraphfair", "puffle", "fedfact", "popets-fairfed"]


def cfg_for(method, dataset, seed):
    c = ExperimentConfig(dataset=dataset, seed=seed, rounds=ROUNDS[dataset],
                         num_clients=CLIENTS[dataset], local_epochs=2, hidden_channels=64)
    apply_method(c, method)
    return c


def jobs():
    # Bail first: all main-table methods to 5 seeds (fills seeds 3,4)
    for m in MAIN + COMP:
        for s in range(5):
            yield cfg_for(m, "bail", s)
    # Pokec: main methods + comp to 5 seeds (fills 2,3,4 and new comp cells)
    for m in MAIN + COMP:
        for s in range(5):
            yield cfg_for(m, "pokec_z", s)


def main():
    logger = ResultLogger("results")
    ran = done = failed = 0
    t0 = time.time()
    for cfg in jobs():
        try:
            rid, res = run_one(cfg, logger)
            if res is None:
                done += 1
            else:
                ran += 1
                print(f"[{ran+done}] {rid}  AUC={res['auc']:.3f} DPD={res['dpd']:.3f} "
                      f"EOD={res['eod']:.3f} ({res.get('wall_s','?')}s) [{(time.time()-t0)/60:.1f}m]",
                      flush=True)
        except Exception:
            failed += 1
            print(f"[FAIL] {cfg.exp_name}/{cfg.dataset}/s{cfg.seed}", flush=True)
            traceback.print_exc()
    print(f"\n[done] ran={ran} skipped={done} failed={failed} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
