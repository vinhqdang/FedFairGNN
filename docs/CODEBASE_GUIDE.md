# TrustFedGNN Codebase Architecture & Engineering Guide

This document provides an engineering-level overview of the `FedFairGNN` codebase. It details the module responsibilities, key functions, data structures, and instructions for running the test suite.

---

## 1. Directory Structure

```
FedFairGNN/
├── src/                          # Core source code
│   ├── config.py                 # ExperimentConfig dataclass & deterministic seed setting
│   ├── models/                   # Neural network architectures
│   │   ├── trustfedgnn.py        # Proposed FSERLayer, Skip-GAT, attention debiasing
│   │   ├── gnn.py                # Standard GCN and GAT baselines
│   │   └── baselines.py          # FairGNN and FairSIN baseline models
│   ├── federated/                # Federated training & aggregation protocols
│   │   ├── client.py             # Client lifecycle, FTGD step, local soft-DPD, weighted BCE
│   │   ├── trainer.py            # FederatedTrainer cross-silo orchestration
│   │   ├── aggregation.py        # BFWA, robust_bfwa, coordinate_median, krum, trimmed_mean
│   │   ├── attacks.py            # Poisoning attacks (Gaussian, sign-flip, scaling, fairness-poison)
│   │   └── server.py             # Central orchestrator APIs
│   ├── trust/                    # Trustworthiness & governance modules
│   │   ├── privacy.py            # RDP PrivacyAccountant & Gaussian noise calculations
│   │   ├── trust_score.py        # Composite Trust Index computation
│   │   ├── uncertainty.py        # MC-Dropout epistemic uncertainty & calibration (ECE)
│   │   ├── compliance.py         # EU AI Act & NIST RMF compliance checks
│   │   ├── explain.py            # GNN attention attribution explainability
│   │   └── incentive.py          # [Archived] Target gradient alignment scoring
│   ├── data/                     # Data loading & partitioning
│   │   ├── datasets.py           # Benchmarks (German, Credit, Bail, Pokec-z, Elliptic)
│   │   ├── partition.py          # Dirichlet non-IID & Louvain community graph partitioning
│   │   └── sampler.py            # SimpleNeighborLoader for mini-batch graph inference
│   └── utils/                    # Common utilities
│       ├── metrics.py            # AUC-ROC, AP, F1-macro, DPD, EOD, weight oscillation Ω_w
│       └── logging_utils.py      # JSONL logging and artifact serialization
├── experiments/                  # Experiment runners & presetting
│   ├── run_experiment.py         # Single experiment entry point
│   ├── run_matrix.py             # Benchmark matrix orchestration
│   ├── methods.py                # Registry of baseline and proposed method configurations
│   ├── report.py                 # Automated report generator
│   └── revision/                 # 14 dedicated revision experiment runners
├── colab/                        # Remote Google Colab GPU execution pipeline
├── tests/                        # Pytest suite locking all invariants (53 tests)
├── results/                      # Raw experimental outputs and logs
└── manuscript/                   # LaTeX publication sources & tables
```

---

## 2. Core Implementation Modules

### A. Client-Side Training & FTGD (`src/federated/client.py`)
- **`Client._ftgd_step(model, optimizer, batch, config)`**:
  1. Computes total gradient $g_{\text{total}} = \nabla_\theta (\mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{fair}})$ and fairness gradient $g_{\text{fair}} = \nabla_\theta (\lambda \mathcal{L}_{\text{fair}})$.
  2. Projects task gradient orthogonal to fairness gradient:
     $$g_{\text{task}}^\perp = g_{\text{total}} - \frac{\langle g_{\text{total}}, g_{\text{fair}}\rangle}{\|g_{\text{fair}}\|^2 + \varepsilon} g_{\text{fair}}$$
  3. Evaluates 2D scalar group means $(\mu_0, \mu_1)$. When DP is enabled, injects calibrated Gaussian noise $\mathcal{N}(0, \sigma_{\text{DP}}^2)$ with sensitivity $\Delta \le \sqrt{2}/n_{\min}$.
  4. Releases privatised disparity $\widetilde{\text{DPD}}_k = |\tilde{\mu}_0 - \tilde{\mu}_1|$ to the server.

### B. Graph Debiasing & FSER Layer (`src/models/trustfedgnn.py`)
- **`FSERLayer.message(edge_index, x_j, x_i, s_j, s_i)`**:
  - Modifies attention logits $\tilde{e}_{vu} = e_{vu} - \beta \cdot \mathbb{I}(s_v \neq s_u) \cdot \max(0, \cos(h_v, h_u))$.
  - Clamps the learnable parameter $\beta \in [0.0, 5.0]$ to prevent numerical overflow in softmax.

### C. Server Aggregators (`src/federated/aggregation.py`)
- **`bfwa_weights(perfs, dpds, tau, ...)`**:
  - Implements the Bi-objective Frank–Wolfe optimization on simplex $\Delta_K$.
  - Updates the Lagrange multiplier $\mu$ via dual gradient ascent.
- **`robust_bfwa_weights(updates, perfs, dpds, tau, ...)`**:
  - Combines Euclidean distance screening against the coordinate-wise median with BFWA weight optimization.
- **`coordinate_median(updates)`**:
  - Computes the coordinate-wise median across client parameter updates. Verified as the unique aggregator completely immune to metadata-deceptive fairness poisoners ($w_{\text{adv}} = 0.000$).

---

## 3. Testing & CI Invariants

The codebase enforces strict unit tests and regression guards. To execute the entire test suite:

```bash
# Run all tests offline
pytest tests/ -q

# Run revision invariants specifically (53 tests)
pytest tests/test_revision_invariants.py -v
```

### Key Invariants Locked:
- **Simplex Invariant**: All aggregators must return a valid 1D vector of length $K$ satisfying $\sum w_k = 1.0 \pm 10^{-6}$ and $w_k \ge 0$.
- **Ablation Isolation**: Disabling DP (`dp_enabled=False`) allows FTGD orthogonalization with $\sigma=0.0$, cleanly isolating gradient geometry from noise injection.
- **Attention Clamping**: Parameter $\beta$ must never exceed $[0.0, 5.0]$.
- **Zero Leakage**: Forward pass must not serialize sensitive attributes $s$ across the network.
