# TrustFedGNN: Trustworthy Federated Graph Neural Networks

Official reference implementation for **TrustFedGNN (FairShare-GNN)**, a trustworthy federated graph neural network framework unifying **Group Fairness**, **Differential Privacy**, **Byzantine Robustness**, and **Multi-Dimensional Explainability** for high-stakes risk and fraud detection across decentralized institutions.

---

## 🏛️ Core Architecture (The Three Pillars)

TrustFedGNN resolves the fundamental **Trustworthy Trilemma** in cross-silo federated graph learning through three cohesive pillars:

| Pillar | Location | Mechanism & Technical Innovation |
|---|---|---|
| **Pillar C1: Privacy-Aware Fairness Feedback** | Client | **FSER** (Fairness-Sensitive Edge Reweighting) suppresses attention across demographic boundaries; **FTGD** performs orthogonal gradient surgery ($g_{\text{task}}^\perp \perp g_{\text{fair}}$) and injects $O(1)$ Gaussian noise into group-mean statistics, achieving $(\epsilon, \delta)$-DP without utility destruction. |
| **Pillar C2: Byzantine-Robust Fairness Aggregation** | Server | **BFWA** (Bi-objective Frank–Wolfe Aggregation) solves dual constrained optimization on the probability simplex $\Delta_K$ to strictly enforce a fairness budget $\tau$; **Coordinate-wise Median** screening provides theoretical immunity against adaptive stealth fairness poisoners ($w_{\text{adv}} = 0.000$). |
| **Pillar C3: Multi-Dimensional Trust Governance** | Evaluation | Epistemic uncertainty via Monte Carlo Dropout, Expected Calibration Error (ECE), Composite Trust Score ranking, and automated compliance auditing aligned with the EU AI Act (Articles 10 & 14) and NIST AI RMF. |

---

## 📊 Supported Datasets

| Dataset | Nodes | Edges | Sensitive Attribute | Task / Domain |
|---|---:|---:|:---:|---|
| **German Credit** | 1,000 | 22,242 | Gender / Age | Credit risk assessment |
| **Bail Recidivism** | 18,876 | 321,308 | Race | Criminal justice recidivism |
| **Credit Default** | 30,000 | 1,436,858 | Age | Default payment prediction |
| **Pokec-z Social** | 67,796 | 1,241,844 | Region | High-homophily social network |
| **Elliptic Bitcoin** | 203,769 | 234,355 | Time period / Proxy | Anti-money laundering (AML) |
| **ogbn-products** | 2,449,029 | 61,859,140 | Node degree / Proxy | Large-scale commercial graph |

---

## 🚀 Quick Start

### 1. Installation
```bash
# Recommended: Python 3.10+
pip install -r requirements.txt
```

### 2. Run Test Suite (CI Invariants)
```bash
# Run the complete test suite (53 tests pass offline in < 6 seconds)
pytest tests/ -q
```

### 3. Run Experiments
```bash
# Run a single experiment (e.g. TrustFedGNN on Bail dataset)
python -m experiments.run_experiment --method fedfairgnn --dataset bail --seed 42

# Run full baseline matrix across benchmark studies
python -m experiments.run_matrix --study main,ablation,robustness
```

### 4. Specialized Revision Studies (`experiments/revision/`)
To reproduce specific publication tables and findings:
```bash
# 7-arm full-factorial ablation grid (C0..C6)
python -m experiments.revision.ablation_grid_runner

# Multi-seed Byzantine robustness sweep across 7 aggregators
python -m experiments.revision.robustness_multiseed

# Empirical update-level attribute inference probe (Table tab:update_attack)
python -m experiments.revision.update_level_attack

# Monte Carlo disparity slack evaluation (Table tab:bfwa_slack)
python -m experiments.revision.bfwa_slack_analysis

# Adaptive stealth adversary breakdown point analysis (Table tab:adaptive_poisoner)
python -m experiments.revision.adaptive_poisoner

# Dirichlet non-IID 48-run heterogeneity sweep (Table tab:dirichlet_sweep)
python -m experiments.revision.dirichlet_sweep

# Louvain community topological clustering (Table tab:partition_comparison)
python -m experiments.revision.metis_partition_experiment

# 2,000-sample Monte Carlo Trust Score sensitivity (Table tab:trust_score_sensitivity)
python -m experiments.revision.trust_score_sensitivity
```

---

## 📁 Repository Layout

```
FedFairGNN/
├── src/                          # Core implementation modules
│   ├── models/                   # Neural network architectures (TrustFedGNN, GCN, GAT, FairGNN, FairSIN)
│   ├── federated/                # Client lifecycle, FTGD step, BFWA aggregators, attack simulators
│   ├── trust/                    # RDP PrivacyAccountant, MC-Dropout, Trust Score, compliance
│   ├── data/                     # Loaders, Dirichlet non-IID & Louvain community partitioners
│   └── utils/                    # Standardized metrics (AUC, AP, F1, DPD, EOD) & JSONL loggers
├── experiments/                  # Experiment presets, matrix runners, and reporting tools
│   └── revision/                 # 14 dedicated revision study runners
├── colab/                        # Remote GPU execution pipeline for heavy workloads
├── docs/                         # Codebase guides, baseline reproductions, and workflows
│   ├── CODEBASE_GUIDE.md         # Detailed module engineering specification
│   ├── EXPERIMENTS_AND_RESULTS.md# Guide to running experiments and reading results
│   ├── COLAB_WORKFLOW.md         # Guide to offloading GPU runs to Google Colab
│   └── BASELINES_AND_SOURCES.md  # Detailed baseline reimplementation fidelity records
├── tests/                        # Offline pytest suite (53/53 tests pass)
├── results/                      # Logged experimental results (JSON)
├── manuscript/                   # Publication LaTeX sources and revision tables
└── archived/                     # Archived exploratory drafts and historical artifacts
```

---

## ☁️ Remote GPU Acceleration (Colab Workflow)
For heavy workloads on large graphs (Pokec-z, Elliptic, ogbn-products), use the automated pipeline in [`colab/`](colab/):
1. Package codebase: `bash colab/00_pack.sh`
2. Bootstrap remote instance: `python colab/01_setup.py`
3. Execute remote run: `python colab/15_stage4_remediation.py`
4. Pull results back to `results/`.
*(See [`docs/COLAB_WORKFLOW.md`](docs/COLAB_WORKFLOW.md) for full instructions).*

---

## 📚 Master Research Documentation
For comprehensive theoretical proofs, Stanford STORM literature surveys, and publication narrative blueprints, refer to the parent research documentation:
- **Manuscript Narrative Blueprint:** [`../docs/RESEARCH_NARRATIVE_BLUEPRINT.md`](../docs/RESEARCH_NARRATIVE_BLUEPRINT.md)
- **Mathematical Formulations & Proofs:** [`../docs/02_mathematical_formulation_and_formal_proofs.md`](../docs/02_mathematical_formulation_and_formal_proofs.md)
- **Lean 4 Formal Verification:** [`../docs/proofs/`](../docs/proofs/)
- **Master Audit Report:** [`../docs/audit/final_revision_report.md`](../docs/audit/final_revision_report.md)
- **Rebuttal Responses Package:** [`../docs/audit/rebuttal_responses.md`](../docs/audit/rebuttal_responses.md)

---

## 👤 Author & Attribution
**Ngoc-Son-An Nguyen**  
*Trustworthy Federated Graph Learning Research Project*
