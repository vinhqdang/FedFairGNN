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

## 3. Results Artifacts & LaTeX Tables

All experimental logs are automatically recorded as structured JSON files:
- **`results/`**: Primary matrix logs, convergence curves, and historical summaries.
- **`results/revision/`**: Specialized logs for the 14 revision runners.
- **`manuscript/tables/`**: Primary SOTA LaTeX tables (`main_pokecz_sota.tex`, `credit_boundary_sota.tex`, `large_scale.tex`).
- **`manuscript/tables/revision/`**: 9 dedicated revision LaTeX tables embedded directly in `manuscript/main.tex`.
