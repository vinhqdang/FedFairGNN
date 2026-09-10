# TrustFedGNN: Contribution-Based Aggregation without Self-Reported Metadata for Trustworthy Federated Graph Neural Networks

Official reference implementation for **TrustFedGNN**, an integrated federated graph neural network framework designed to eliminate the critical vulnerability of **self-reported metadata deception** in decentralized, cross-silo learning environments. 

---

## 🏛️ Core Architecture (The Three Pillars)

TrustFedGNN resolves the fundamental **Trustworthy Trilemma** in cross-silo federated graph learning through three cohesive pillars:

| Pillar | Location | Mechanism & Technical Innovation |
|---|---|---|
| **Pillar C1: Client-Side Debiasing & Targeted DP** | Client | **FSER** (Fairness-Sensitive Edge Reweighting) attenuates cross-group attention logits; **FTGD** (Fairness-Targeted Gradient Decomposition) projects updates orthogonal to the fairness ascent direction ($g_{\text{task}}^\perp \perp g_{\text{fair}}$) and injects calibrated Gaussian noise into 2D scalar group-mean statistics, achieving certified $(\epsilon, \delta)$-DP at $O(1)$ noise with only $0.0017$ AUC drop (vs $0.202$ for full DP-SGD). |
| **Pillar C2: Metadata-Immune Contribution Valuation** | Server | **FU-Shapley** scores client updates against a bi-objective target gradient ($g_{\text{target}}$) evaluated on an isolated server-side holdout split ($\mathcal{D}_{\text{val}}$). It **never reads client-reported disparity metadata**, guaranteeing **Metadata Immunity ($\lVert\Delta\bm{w}\rVert_\infty = 0.0000$ bit-exact)** under falsified metadata attacks that capture $86.2\%$ of Frank–Wolfe/BFWA aggregates. **`robust_fu_shapley`** prepends coordinate-wise median screening against Byzantine scaling attacks up to $f/K \le 0.30$. |
| **Pillar C3: Machine-Checked Proofs & Trust Governance** | Audit & Verification | Four algebraic properties are formally machine-checked in **Lean 4** ([`docs/proofs/`](docs/proofs/)); weight stability $\Omega_w$ is $27\times$ steadier than Frank–Wolfe dual-ascent; epistemic uncertainty via MC-Dropout and alignment with EU AI Act (Articles 10 & 14) and NIST AI RMF. |

---

## 📊 Benchmark Datasets & Scoping

| Dataset | Nodes | Edges | Sensitive Attribute | Experimental Role & Scoping |
|---|---:|---:|:---:|---|
| **Pokec-z Social** | 67,796 | 1,241,844 | Region | Flagship SOTA benchmark ($\text{AUC} = 0.7899 \pm 0.0095$, $p=0.0020$ vs 7 baselines over 10 seeds). |
| **Credit Default** | 30,000 | 1,436,858 | Age | Primary financial benchmark ($\text{AUC} = 0.7522 \pm 0.0065$, $\text{DPD} = 0.0615 \pm 0.0371$). |
| **Bail Recidivism** | 18,876 | 321,308 | Race | Criminal justice benchmark for Dirichlet sweep ($\alpha \in [0.1, 1.0]$) & convergence dynamics. |
| **German Credit** | 1,000 | 22,242 | Gender / Age | Canonical 8-arm ablation suite ($M_1 \dots M_7$ + clean arm) and metadata attack evaluation. |
| **Elliptic Bitcoin** | 203,769 | 234,355 | Time period / Proxy | High homophily ($h_s = 1.0000$, transactions intra-timestep only); FSER acts as architectural no-op. |
| **ogbn-products** | 2,449,029 | 61,859,140 | Degree / Proxy | Evaluated strictly as million-node engineering scalability proof without FTGD privacy claims. |

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

# Generate Pillar C2 evidence tables (metadata_immunity, two_tier_defense, weight_stability, cost)
python experiments/make_tables_c2.py
```

---

## 🔬 Machine-Checked Proofs in Lean 4 (`docs/proofs/`)

Four core algebraic and geometric properties underpinning TrustFedGNN's theoretical guarantees are formally machine-checked using **Lean 4**:

| Proof Source File | Mathematics Result | Guaranteed Property |
|---|---|---|
| [`docs/proofs/OrthogonalProjection.lean`](docs/proofs/OrthogonalProjection.lean) | Theorem 4 | Exact vanishing of $\langle g_{\text{task}}^\perp, g_{\text{fair}}\rangle$ ($\varepsilon = 0$), eliminating linear leakage. |
| [`docs/proofs/SimplexProperties.lean`](docs/proofs/SimplexProperties.lean) | Proposition 2 | Valid aggregation probability simplex ($w_k \ge 0, \sum w_k = 1$). |
| [`docs/proofs/NullPlayer.lean`](docs/proofs/NullPlayer.lean) | Proposition 3 | Non-contributing client ($g_k = \mathbf{0}$) strictly receives $w_k = 0$ in all branches. |
| [`docs/proofs/LinearDecomposition.lean`](docs/proofs/LinearDecomposition.lean) | Proposition 4 | Exact bilinearity additive split: $\varphi_k = \varphi_k^{\text{util}} + \alpha \varphi_k^{\text{fair}}$. |

---

## 📁 Repository Layout

```
FedFairGNN/
├── src/                          # Core implementation modules
│   ├── models/                   # Neural network architectures (TrustFedGNN, GCN, GAT, FairGNN, FairSIN)
│   ├── federated/                # Client lifecycle, FTGD step, FU-Shapley aggregators, attack simulators
│   ├── trust/                    # RDP PrivacyAccountant, MC-Dropout, Trust Score, compliance
│   ├── data/                     # Loaders, Dirichlet non-IID & Louvain community partitioners
│   └── utils/                    # Standardized metrics (AUC, AP, F1, DPD, EOD) & JSONL loggers
├── experiments/                  # Experiment presets, matrix runners, and reporting tools
│   ├── make_tables.py            # Primary publication tables generator
│   ├── make_tables_c2.py         # Pillar C2 evidence tables generator
│   ├── make_figures.py           # Publication figures generator
│   └── revision/                 # 14 dedicated revision study runners
├── colab/                        # Remote GPU execution pipeline for heavy workloads
├── docs/                         # Codebase guides, baseline reproductions, and proofs
│   ├── proofs/                   # Machine-checked Lean 4 formal interactive theorem proofs
│   ├── CODEBASE_GUIDE.md         # Detailed module engineering specification
│   ├── EXPERIMENTS_AND_RESULTS.md# Guide to running experiments and reading results
│   ├── COLAB_WORKFLOW.md         # Guide to offloading GPU runs to Google Colab
│   └── BASELINES_AND_SOURCES.md  # Detailed baseline reimplementation fidelity records
├── tests/                        # Offline pytest suite (53/53 tests pass)
├── results/                      # Logged experimental results (JSON)
├── manuscript_neurocomputing/    # Elsevier Neurocomputing manuscript LaTeX sources & figures
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
For comprehensive theoretical derivations, provenance audit manifests, and publication narrative blueprints:
- **Manuscript Blueprint:** [`../docs/05_manuscript_blueprint.md`](../docs/05_manuscript_blueprint.md)
- **Novelty & Advantages Matrix:** [`../docs/04_1_novelty_advantages.md`](../docs/04_1_novelty_advantages.md)
- **Archived Audit Manifests:** [`../docs/archived/audit/04_experiment_execution.md`](../docs/archived/audit/04_experiment_execution.md)
- **Lean 4 Proofs Suite:** [`docs/proofs/`](docs/proofs/)

---

## 👤 Author & Attribution
**Ngoc-Son-An Nguyen**  
*Trustworthy Federated Graph Learning Research Project*

