# TrustFedGNN Codebase Architecture & Engineering Guide

This document provides an engineering-level overview of the `FedFairGNN` codebase. It details the module responsibilities, key functions, data structures, and instructions for running the test suite.

---

## 1. Directory Structure

```
FedFairGNN/
├── src/                          # Core source code
│   ├── config.py                 # ExperimentConfig dataclass & deterministic seed setting
│   ├── models/                   # Neural network architectures
│   │   ├── trustfedgnn.py        # STALE, UNUSED duplicate - not on the training path (see note below)
│   │   ├── gnn.py                # Actual FSERLayer / TrustFedGNN implementation + GCN, GAT, FairGNN, FairSIN, FaVGNN baselines
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

### B. Graph Debiasing & FSER Layer (`src/models/gnn.py`)

> **Pointer note.** The classes that actually run are `FSERLayer` and `TrustFedGNN` in **`src/models/gnn.py`** — `src/models/__init__.py` builds its `_REGISTRY` (`"trustfedgnn" -> TrustFedGNN`) from imports out of `gnn.py`. The file `src/models/trustfedgnn.py` is a **stale, divergent duplicate that is never instantiated** (its `TrustFedGNN.__init__` accepts neither the `beta_init` nor the `fser_mode` kwargs that `build_model` passes, and hardcodes `beta = 0.5`, so instantiating it would raise `TypeError`); it should be deleted or clearly marked archived. Do not read it as the reference implementation.

- **`FSERLayer.message(edge_index, x_j, x_i, s_j, s_i)`**:
  - Modifies attention logits $\tilde{e}_{vu} = e_{vu} - \beta \cdot \mathbb{I}(s_v \neq s_u) \cdot \max(0, \cos(h_v, h_u))$.
  - Clamps the learnable parameter $\beta \in [0.0, 5.0]$ to prevent numerical overflow in softmax.

### C. Server Aggregators (`src/federated/aggregation.py` & `src/trust/incentive.py`)
- **`fu_shapley` (Canonical Proposed Aggregator)**:
  - Evaluates bi-objective target gradient on server-side holdout split: $g_{\text{target}} = g_{\text{task}}^{\text{srv}} + \alpha g_{\text{fair}}^{\text{srv}}$ ($\alpha = 0.1$).
  - Scores client updates via scale-invariant inner product: $\varphi_k = \langle g_k, g_{\text{target}}\rangle / (\|g_{\text{target}}\| + 10^{-8})$.
  - Smooths scores across rounds via EMA ($\beta_{\text{ema}} = 0.9$), handles non-finite scores safely, and gates onto simplex with explicit null-player mask:
    $$w_k = \frac{\max(0, \bar{\varphi}_k) \cdot \mathbb{I}(g_k \neq \mathbf{0})}{\sum_j \max(0, \bar{\varphi}_j) \cdot \mathbb{I}(g_j \neq \mathbf{0})}$$
  - **Guarantees Metadata Immunity**: Never consumes self-reported fairness disparity $\widehat{\text{DPD}}_k$, proving $\lVert\Delta\bm{w}\rVert_\infty = 0.0000$ bit-exact under falsification attacks.
- **`robust_fu_shapley` (Byzantine-Resilient Variant)**:
  - Prepends coordinate-wise median distance screening (discarding the $f$ farthest updates) before running the FU-Shapley gating.
  - Neutralizes scaling adversaries ($w_{\text{adv}} = 0.0000$ at $f/K \le 0.30$), closing the vulnerability where scaling updates align positively with $g_{\text{target}}$.
- **`bfwa_weights(perfs, dpds, tau, ...)` (Baseline)**:
  - Implements a penalised Bi-objective Frank–Wolfe iteration on simplex $\Delta_K$ steering weights toward budget $\tau$ on *reported* disparity. Vulnerable to metadata falsification (capturing $86.2\%$ aggregate share).
- **`coordinate_median(updates)` (Baseline)**:
  - Computes coordinate-wise median across client parameter updates, bounded by the standard $f < K/2$ breakdown point. Does not produce a client weight vector.

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
