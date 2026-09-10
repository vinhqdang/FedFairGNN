# Experiments & Results Reproduction Guide

This guide details how to execute experiments, reproduce the published numbers, and locate all logged results in the `FedFairGNN` repository.

---

## 1. Quick Reproduction Commands

### A. Run a Single Benchmark Experiment
```bash
# Run TrustFedGNN on Bail Recidivism with Seed 42
python -m experiments.run_experiment --method fedfairgnn --dataset bail --seed 42

# Run baseline FedAvg-GCN on Credit Default
python -m experiments.run_experiment --method fedavg-gcn --dataset credit --seed 42
```

### B. Run the Benchmark Matrix
```bash
# Executes main, ablation, and robustness studies across datasets
python -m experiments.run_matrix --study main,ablation,robustness
```

---

## 2. Dedicated Revision Experiment Runners (`experiments/revision/`)

The `experiments/revision/` directory contains 14 specialized, self-contained scripts addressing each experimental requirement from the revision audit:

| Script Name | Purpose | Output Location |
|---|---|---|
| [`ablation_grid_runner.py`](../experiments/revision/ablation_grid_runner.py) | 7-arm full-factorial ablation ($C_0 \dots C_6$) on Bail, Credit, Pokec-z $\times$ 10 seeds | `results/revision/ablation_grid_*.json` |
| [`fser_beta_analysis.py`](../experiments/revision/fser_beta_analysis.py) | Convergence and layer-wise distribution analysis of $\beta$ parameters | Console & paper text |
| [`fser_fairness_extract.py`](../experiments/revision/fser_fairness_extract.py) | Direct $\Delta\text{DPD}$ and $\Delta\text{EOD}$ extraction with Wilcoxon tests | `results/revision/fser_fairness_extract.json` |
| [`robustness_multiseed.py`](../experiments/revision/robustness_multiseed.py) | Multi-seed Byzantine robustness across 7 aggregators and 3 attacks | `results/revision/robustness_multiseed.json` |
| [`dp_accounting_table.py`](../experiments/revision/dp_accounting_table.py) | Analytical RDP-to-$(\epsilon, \delta)$ accounting table across 6 datasets | `manuscript/tables/revision/dp_accounting.tex` |
| [`update_level_attack.py`](../experiments/revision/update_level_attack.py) | Linear & MLP attribute inference probe on parameter updates vs statistics | `results/revision/update_level_attack.json` & `tab:update_attack` |
| [`bfwa_slack_analysis.py`](../experiments/revision/bfwa_slack_analysis.py) | 1,000-sample Monte Carlo analysis of BFWA disparity constraint slack | `results/revision/bfwa_slack.json` & `tab:bfwa_slack` |
| [`adaptive_poisoner.py`](../experiments/revision/adaptive_poisoner.py) | Omniscient stealth adversary breakdown sweep ($f \in [0.1, 0.4]$) | `results/revision/adaptive_poisoner.json` & `tab:adaptive_poisoner` |
| [`dirichlet_sweep.py`](../experiments/revision/dirichlet_sweep.py) | 48-run sweep across $\alpha \in [0.1, 1.0]$ and $K \in [5, 20]$ | `results/revision/dirichlet_sweep.json` & `tab:dirichlet_sweep` |
| [`metis_partition_experiment.py`](../experiments/revision/metis_partition_experiment.py) | Graph topology partition comparison (Uniform vs Dirichlet vs Louvain) | `results/revision/metis_partition_experiment.json` & `tab:partition_comparison` |
| [`trust_score_sensitivity.py`](../experiments/revision/trust_score_sensitivity.py) | 2,000-sample Monte Carlo rank perturbation testing of Composite Trust Score | `results/revision/trust_score_sensitivity.json` & `tab:trust_score_sensitivity` |
| [`centralized_sanity_anchors.py`](../experiments/revision/centralized_sanity_anchors.py) | Centralized vs Federated GCN/GAT bounds ($\Delta_{\text{FL}}$ validation) | `results/revision/centralized_sanity_anchors.json` & `tab:centralized_sanity` |
| [`elliptic_proxy_sensitivity.py`](../experiments/revision/elliptic_proxy_sensitivity.py) | Subgroup proxy sensitivity (Demographic vs Hubs vs Behavioral Quantiles) | `results/revision/elliptic_proxy_sensitivity.json` & `tab:proxy_sensitivity` |

---

## 3. Pillar C2 Evidence Tables Generation (`experiments/make_tables_c2.py`)

The central defensive and systemic claims of TrustFedGNN (Metadata Immunity, Byzantine Defense, Weight Stability, and Computational Cost) are generated directly from experimental JSON logs with zero hand-typed cells:

```bash
# Generate the 4 Pillar C2 evidence tables directly into manuscript tables directory:
python experiments/make_tables_c2.py
```

| Generated Table | Math / Empirical Claim | Key Result |
|---|---|---|
| [`metadata_immunity.tex`](../manuscript_neurocomputing/tables/metadata_immunity.tex) | Theorem 2 (Metadata Immunity) | Under client falsification ($\widehat{\dpd}_k=0.0, \mathrm{Perf}_k=0.99$), Frank–Wolfe/BFWA assigns $86.2\%$ weight share to the liar; FU-Shapley is bit-identical ($\lVert\Delta\bm{w}\rVert_\infty = 0.0000$). |
| [`two_tier_defense.tex`](../manuscript_neurocomputing/tables/two_tier_defense.tex) | Robustness under 20% Byzantine minority | Reports $w_{\text{adv}}$ and AUC across 4 attack scenarios; reports NaN divergence rates transparently (superscripts) demonstrating that removing EMA (M7) causes training divergence. |
| [`weight_stability.tex`](../manuscript_neurocomputing/tables/weight_stability.tex) | Total Weight Variation $\Omega_w$ | Quantifies the cost of per-round re-scoring; FU-Shapley is $27\times$ more stable than Frank–Wolfe dual-ascent re-solving. |
| [`cost.tex`](../manuscript_neurocomputing/tables/cost.tex) | Wall-clock execution time | Measured on a single NVIDIA T4 GPU ($K=10, R=50, n=10$ seeds); overhead is $1.86\times$ vs FedAvg, dominated by server-side holdout gradient evaluation. |

---

## 4. Formal Verification in Lean 4 (`docs/proofs/`)

Four foundational algebraic and geometric theorems are formally verified and machine-checked in Lean 4:
- [`docs/proofs/OrthogonalProjection.lean`](proofs/OrthogonalProjection.lean): FTGD exact orthogonality ($\varepsilon = 0$).
- [`docs/proofs/SimplexProperties.lean`](proofs/SimplexProperties.lean): Aggregation weight simplex validity ($w_k \ge 0, \sum w_k = 1$).
- [`docs/proofs/NullPlayer.lean`](proofs/NullPlayer.lean): Null player receiving strictly zero weight across all execution paths.
- [`docs/proofs/LinearDecomposition.lean`](proofs/LinearDecomposition.lean): Additive bilinearity decomposition of contribution scores.

---

## 5. Results Artifacts & LaTeX Tables

All experimental logs are recorded as reproducible JSON artifacts:
- **`results/`**: Canonical ablation suite (`canonical_suite.json`), SOTA benchmark logs (`sota_pokecz.json`, `sota_credit.json`), and convergence curves.
- **`results/revision/`**: Specialized logs for the 14 revision runners.
- **`manuscript_neurocomputing/tables/`**: Authoritative publication LaTeX tables compiled in `manuscript_neurocomputing/main.tex`.

