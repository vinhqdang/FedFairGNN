# CLAMP

Fairness-constrained federated graph learning with secure aggregation, private
group statistics and bounded Byzantine influence.

This directory is self-contained and independent of the rest of the repository.
It holds:
- the code (`src/`, `experiments/`) and the tests,
- the design specification (`docs/clamp_design.md`),
- the manuscript (`manuscript/`),
- the per-run results (`results/`).

## Setup

    pip install -r requirements.txt

Place the raw datasets under `data/raw/` inside this directory (German, Bail,
Credit, Pokec), in the layout expected by `src/data/datasets.py`.

## Usage (run from this directory)

    python -m pytest -q tests                                     # tests
    python -m experiments.run_clamp_suite --group e1 --dry-run    # list jobs
    python -m experiments.run_clamp_suite --group e1 --workers 4  # run E1
    python -m experiments.make_clamp_tables                       # tables
    python -m experiments.clamp_privacy --datasets german,bail    # privacy study

Set `FEDFAIR_DEVICE=cuda` to run on a GPU.

Experiment groups:
- e1: main comparison
- e2: fairness-budget sweep
- e3: update attacks
- e4: statistics attacks
- e5: privacy budget and release period
- e6: ablations
