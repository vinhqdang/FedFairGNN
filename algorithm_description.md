# FairFedGNN: Algorithm Specification
### Fairness-Constrained Federated Graph Neural Network for Fraud Detection

> **Purpose of this document**: Complete, self-contained algorithm specification.  
> An engineer should be able to implement FairFedGNN from scratch using only this document.  
> No external references are required.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Notation and Data Structures](#2-notation-and-data-structures)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Component 1 — FSER: Fairness-Sensitive Edge Reweighting](#4-component-1--fser-fairness-sensitive-edge-reweighting)
5. [Component 2 — FTGD: Fairness-Task Gradient Decomposition with Targeted DP](#5-component-2--ftgd-fairness-task-gradient-decomposition-with-targeted-dp)
6. [Component 3 — BFWA: Bi-Objective Frank-Wolfe Aggregation](#6-component-3--bfwa-bi-objective-frank-wolfe-aggregation)
7. [Full Training Loop: FairFedGNN](#7-full-training-loop-fairfedgnn)
8. [Loss Functions](#8-loss-functions)
9. [Model Architecture](#9-model-architecture)
10. [Hyperparameters Reference](#10-hyperparameters-reference)
11. [Data Preprocessing](#11-data-preprocessing)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Federated Communication Protocol](#13-federated-communication-protocol)
14. [Implementation Notes and Edge Cases](#14-implementation-notes-and-edge-cases)

---

## 1. System Overview

### 1.1 Problem Setting

There are **K financial institutions** (clients), each holding a private local transaction graph. No client can share raw data. A central server coordinates training. The goal is to learn a global fraud detection model that is both:
- **Accurate**: high AUC-ROC / F1 for detecting fraudulent transactions
- **Demographically fair**: fraud flagging rates do not disproportionately differ across sensitive attribute groups (e.g., gender, income tier, geographic region)

### 1.2 Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              SERVER                                      │
│                                                                          │
│   Receives per round: (g_task^k, g̃_fair^k, DPD_k, Perf_k) from each k  │
│   Runs:  BFWA → computes optimal aggregation weights w*                 │
│   Sends: Updated global parameters θ_t back to all clients              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  broadcast θ_{t-1}
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Client 1   │     │  Client 2   │  …  │  Client K   │
   │  Bank A     │     │  Mobile $   │     │  Fintech    │
   │             │     │             │     │             │
   │  [FSER]     │     │  [FSER]     │     │  [FSER]     │
   │  [FTGD]     │     │  [FTGD]     │     │  [FTGD]     │
   │             │     │             │     │             │
   │  G_k local  │     │  G_k local  │     │  G_k local  │
   └─────────────┘     └─────────────┘     └─────────────┘
```

### 1.3 Three Novel Components

| Component | Where | What It Solves |
|-----------|--------|----------------|
| **FSER** (Alg. 1) | Client — inside GNN | Prevents homophily-driven demographic bias during message passing |
| **FTGD** (Alg. 2) | Client — after backprop | Separates gradients into task vs. fairness subspaces; adds DP noise only to fairness subspace |
| **BFWA** (Alg. 3) | Server — aggregation | Enforces a hard demographic fairness budget τ during aggregation via Frank-Wolfe optimization |

---

## 2. Notation and Data Structures

### 2.1 Scalars, Vectors, Matrices

| Symbol | Type | Description |
|--------|------|-------------|
| `K` | int | Number of clients (federated parties) |
| `T` | int | Number of communication rounds |
| `E` | int | Number of local epochs per round |
| `L` | int | Number of GNN layers |
| `d` | int | Node feature dimension (input) |
| `d_h` | int | Hidden embedding dimension |
| `n_k` | int | Number of nodes in client k's graph |
| `m_k` | int | Number of edges in client k's graph |
| `β^(l)` | float scalar | Learnable fairness suppression coefficient at layer l |
| `λ` | float | Fairness loss weight in total loss |
| `τ` | float ∈ [0,1] | Fairness budget: max allowed global DPD |
| `ε` | float | Differential privacy epsilon (privacy budget) |
| `δ` | float | DP delta (failure probability); typically 1e-5 |
| `C` | float | DP gradient clipping norm bound |

### 2.2 Per-Client Data Structures

```
Client k holds:
  G_k = (V_k, E_k, X_k, Y_k, S_k)

  V_k  : node set                           shape: [n_k]
  E_k  : edge index (COO format)            shape: [2, m_k]   int64
  X_k  : node feature matrix                shape: [n_k, d]   float32
  Y_k  : fraud label vector (0=legit,1=fraud) shape: [n_k]    int64
  S_k  : sensitive attribute vector (binary) shape: [n_k]     int64
         S_k[i] = 0  (group A, e.g. low-income)
         S_k[i] = 1  (group B, e.g. high-income)
```

> **Note on S_k**: Sensitive attributes must be binarized before training.  
> For multi-valued attributes (e.g., 5 income tiers), use one-vs-rest binarization,  
> run FSER/FTGD for each binary split, and average the resulting fairness penalties.

### 2.3 Model Parameters

```
θ = {
  W^(l)  ∈ R^{d_h × d_h}   for l = 1,...,L   (GNN weight matrices)
  a^(l)  ∈ R^{2d_h}         for l = 1,...,L   (GAT attention vectors)
  β^(l)  ∈ R                for l = 1,...,L   (FSER fairness coefficients, learnable)
  W_cls  ∈ R^{d_h × 1}                        (classifier head weight)
  b_cls  ∈ R                                   (classifier bias)
}

Total parameter count (example): L=3, d=64, d_h=128
  → 3 × (128×128) + 3 × 256 + 3 × 1 + 128 + 1 ≈ 50,000 params
```

### 2.4 Per-Round Communication Payload

```
Client k → Server per round:
  g_task^k   : R^{|θ|}    Clean task gradient (fraud signal)
  g̃_fair^k   : R^{|θ|}    DP-noised fairness gradient
  DPD_k      : float       Local Demographic Parity Difference
  Perf_k     : float       Local validation AUC-ROC

Server → Client per round:
  θ_t        : R^{|θ|}    Updated global model parameters
```

---

## 3. Mathematical Foundations

### 3.1 Graph Attention Network (Baseline Message Passing)

Standard GAT attention (Veličković et al., 2018) — this is what FSER modifies:

```
For node i at layer l:

  Step 1 — Linear transform:
    z_i = W^(l) · h_i^(l)           h_i^(l) ∈ R^{d_h}

  Step 2 — Attention coefficient:
    e_ij = LeakyReLU( a^(l)^T · [z_i || z_j] )    || = concatenation

  Step 3 — Normalize:
    α_ij = exp(e_ij) / Σ_{j' ∈ N(i)} exp(e_ij')     (softmax over neighbors)

  Step 4 — Aggregate:
    h_i^(l+1) = σ( Σ_{j ∈ N(i)} α_ij · W^(l) · h_j^(l) )    σ = ELU
```

### 3.2 Demographic Parity Difference (DPD)

DPD measures disparity in fraud prediction rates between sensitive groups:

```
DPD = | P(Ŷ=1 | S=0) − P(Ŷ=1 | S=1) |

In practice, estimated on a batch:
  DPD_k = | mean(Ŷ_k[S_k=0]) − mean(Ŷ_k[S_k=1]) |

where Ŷ_k are predicted fraud probabilities (soft, pre-threshold).

Range: [0, 1].   Perfect fairness → DPD = 0.
```

### 3.3 Differential Privacy (DP) Guarantee

We use the standard Gaussian mechanism for (ε, δ)-DP:

```
Given:
  - Sensitivity: C  (gradient clipping bound)
  - Privacy budget: ε
  - Failure probability: δ

Noise standard deviation:
  σ_DP = C · √(2 · ln(1.25 / δ)) / ε

Mechanism:
  g̃ = g_clipped + N(0, σ_DP² · I)

This satisfies (ε, δ)-DP by the Gaussian mechanism theorem.
```

### 3.4 Gradient Orthogonal Decomposition

Given total gradient `g` and fairness gradient `g_fair`:

```
Decompose g into:
  g_task  = g − proj(g, g_fair)       (component orthogonal to g_fair)
  g_fair  = g_fair                     (fairness component)

where:
  proj(g, g_fair) = (g · g_fair / ||g_fair||²) · g_fair   (scalar projection)

Verification:
  g_task · g_fair = 0    (orthogonality check, should hold up to float precision)
  g_task + proj_component = g    (reconstruction check)
```

### 3.5 Frank-Wolfe on the Probability Simplex

Frank-Wolfe (Conditional Gradient) solves:

```
min_{w ∈ Δ_K} F(w)

where Δ_K = { w ∈ R^K : Σ_k w_k = 1, w_k ≥ 0 }

Each iteration:
  1. Compute gradient: ∇F(w^t)
  2. Linear oracle:    s* = argmin_{s ∈ Δ_K} s^T · ∇F(w^t)
                           → s* = e_{k*}  where k* = argmin_k [∇F(w^t)]_k
                           (this is just the unit vector of the minimizing index)
  3. Update:           w^{t+1} = (1 − γ_t) · w^t + γ_t · s*
                       with step size γ_t = 2 / (t + 2)

Convergence: O(1/T) rate for convex F.
No projection step required — the update naturally stays in Δ_K.
```

---

## 4. Component 1 — FSER: Fairness-Sensitive Edge Reweighting

### 4.1 Motivation

In a transaction graph, nodes from sensitive group A may be structurally connected to fraudsters due to socioeconomic network effects (e.g., shared geography, shared employer). Standard GAT will assign high attention to these neighbors and learn to flag group A as risky — even if group A members are not actually more fraudulent.

FSER detects this pattern: edges connecting nodes from **different sensitive groups** that have **similar embeddings** are penalized. Such edges represent "suspicious cross-group similarity" that is more likely structural bias than genuine fraud signal.

### 4.2 Algorithm

```
Algorithm 1: FSER — Fairness-Sensitive Edge Reweighting
================================================================
Runs at: Each client k, inside each forward pass
Replaces: Standard GAT attention aggregation

Input:
  h^(l)    : node embeddings at layer l      shape: [n_k, d_h]
  E_k      : edge index (source, target)     shape: [2, m_k]
  S_k      : sensitive attribute vector      shape: [n_k]       int64
  W^(l)    : weight matrix at layer l        shape: [d_h, d_h]
  a^(l)    : attention vector at layer l     shape: [2*d_h]
  β^(l)    : fairness coefficient at layer l  scalar, learnable

Output:
  h^(l+1) : updated node embeddings         shape: [n_k, d_h]
================================================================

For each edge (i, j) in E_k:  [can be vectorized over all edges]

  --- Standard GAT attention ---
  z_i = W^(l) · h_i^(l)                          shape: [d_h]
  z_j = W^(l) · h_j^(l)                          shape: [d_h]

  e_ij = LeakyReLU( a^(l)^T · concat(z_i, z_j) ) scalar, negative_slope=0.2

  --- FSER fairness risk score ---
  Δs_ij = |S_k[i] − S_k[j]|                      ∈ {0, 1}
           (1 if cross-group edge, 0 if same-group)

  cos_ij = (h_i^(l) · h_j^(l)) /
           (||h_i^(l)||_2 · ||h_j^(l)||_2 + 1e-8) ∈ [-1, 1]

  φ_ij = Δs_ij · max(0, cos_ij)
          (positive only for cross-group edges with similar embeddings)

  --- Fairness-corrected attention ---
  ẽ_ij = e_ij − β^(l) · φ_ij
          (suppress attention for biased cross-group edges)

  --- Normalize (softmax per node i over its neighborhood N(i)) ---
  ᾱ_ij = exp(ẽ_ij) / Σ_{j' ∈ N(i)} exp(ẽ_ij')

  --- Aggregate ---
  h_i^(l+1) = ELU( Σ_{j ∈ N(i)} ᾱ_ij · W^(l) · h_j^(l) )

--- Gradient flow for β^(l) ---
β^(l) is updated during backprop via:
  ∂L_total / ∂β^(l) = Σ_{(i,j)} (∂L_total / ∂ẽ_ij) · (−φ_ij)

β^(l) is initialized to 0.5 and constrained to [0, 5] via clipping after each step.
```

### 4.3 Vectorized Implementation Notes

```python
# Pseudocode for efficient batch implementation
# Assumes edge_index shape [2, m_k], node features h shape [n_k, d_h]

src, dst = edge_index[0], edge_index[1]     # [m_k] each

# Linear transform
z = h @ W.T                                  # [n_k, d_h]

# GAT attention scores
z_cat = torch.cat([z[src], z[dst]], dim=1)  # [m_k, 2*d_h]
e = F.leaky_relu(z_cat @ a, 0.2)            # [m_k]

# FSER fairness risk
delta_s = (S[src] != S[dst]).float()        # [m_k]  cross-group indicator

h_norm = F.normalize(h, p=2, dim=1)         # [n_k, d_h]
cos_sim = (h_norm[src] * h_norm[dst]).sum(1) # [m_k]

phi = delta_s * torch.clamp(cos_sim, min=0) # [m_k]

# Corrected attention
e_tilde = e - beta * phi                     # [m_k]

# Softmax per node (use scatter_softmax or manual logsumexp)
alpha = softmax_per_node(e_tilde, dst, n_k) # [m_k]

# Aggregate
h_new = scatter_add(alpha.unsqueeze(1) * (h[src] @ W.T), dst, n_k)  # [n_k, d_h]
h_new = F.elu(h_new)
```

---

## 5. Component 2 — FTGD: Fairness-Task Gradient Decomposition with Targeted DP

### 5.1 Motivation

Standard DP-FedAvg adds Gaussian noise with the same scale to the **entire** gradient vector. For fraud detection, the task gradient (fraud signal) is sparse and high-magnitude. Adding noise large enough for demographic privacy destroys the fraud signal.

Key insight: the sensitive demographic information is encoded **only** in the fairness gradient component. The task gradient carries no group membership information. Therefore, DP noise need only protect the fairness component, leaving the task gradient clean.

This is privacy-preserving by the **post-processing immunity** property of DP: if g̃_fair satisfies (ε, δ)-DP, then any function of g_task (which is computed independently from S_k after decomposition) also satisfies (ε, δ)-DP for the sensitive attribute information.

### 5.2 Algorithm

```
Algorithm 2: FTGD — Fairness-Task Gradient Decomposition with Targeted DP
================================================================
Runs at: Each client k, after each local epoch's backward pass
Replaces: Standard DP gradient computation in DP-FedAvg

Input:
  θ_k      : current local model parameters
  G_k      : local graph
  Y_k      : fraud labels
  S_k      : sensitive attributes
  λ        : fairness loss weight (hyperparameter)
  C        : gradient clipping bound (DP sensitivity)
  ε        : privacy budget
  δ        : DP delta

Output:
  g_task   : clean task gradient        shape: [|θ|]
  g̃_fair   : DP-noised fairness gradient shape: [|θ|]
================================================================

--- Step 1: Forward pass ---
Ŷ_k = FairGNN(G_k; θ_k)               shape: [n_k],  values in (0,1) via sigmoid

--- Step 2: Compute losses ---
# Task loss: binary cross-entropy for fraud detection
L_fraud = − (1/n_k) Σ_i [ Y_k[i] · log(Ŷ_k[i]) + (1 − Y_k[i]) · log(1 − Ŷ_k[i]) ]

# Fairness loss: soft demographic parity violation
μ_0 = mean(Ŷ_k[S_k = 0])              # mean predicted fraud prob for group 0
μ_1 = mean(Ŷ_k[S_k = 1])              # mean predicted fraud prob for group 1
L_fair = |μ_0 − μ_1|                   # absolute difference in prediction rates

# If either group has fewer than 10 samples in this batch, skip L_fair for this step
# (set L_fair = 0) to avoid noisy estimates destabilizing training.

# Total loss
L_total = L_fraud + λ · L_fair

--- Step 3: Compute gradients ---
g       = ∇_θ L_total    via autograd backward   shape: [|θ|]
g_fair  = ∇_θ [λ · L_fair]  via autograd backward  shape: [|θ|]
  # Implementation: compute two separate backward passes, or use retain_graph=True

--- Step 4: Orthogonal decomposition ---
dot     = inner_product(g, g_fair)               scalar
norm_sq = inner_product(g_fair, g_fair) + 1e-12  scalar (avoid division by zero)

proj    = (dot / norm_sq) · g_fair               shape: [|θ|]
g_task  = g − proj                               shape: [|θ|], orthogonal to g_fair

--- Step 5: Clip fairness gradient ---
clip_factor = max(1.0, L2_norm(g_fair) / C)
g_fair_clipped = g_fair / clip_factor            shape: [|θ|]

--- Step 6: Add calibrated Gaussian noise to fairness gradient ONLY ---
σ_DP    = C · sqrt(2 · ln(1.25 / δ)) / ε
noise   = sample N(0, σ_DP² · I_{|θ|})          shape: [|θ|]
g̃_fair  = g_fair_clipped + noise                 shape: [|θ|]

--- Step 7: Return ---
Return (g_task, g̃_fair)

Note: g_task is NOT clipped and NOT noised.
      g_task + g̃_fair approximates the full DP-protected gradient.
```

### 5.3 Privacy Accounting

```
Noise calibration:
  σ_DP = C · √(2 · ln(1.25/δ)) / ε

Example values:
  ε = 1.0,  δ = 1e-5,  C = 1.0
  → σ_DP = 1.0 · √(2 · ln(125000)) / 1.0
           = √(2 · 11.736) = √23.47 ≈ 4.85

  This means each coordinate of g̃_fair gets noise of std ≈ 4.85 × C.

Advantage over full-gradient DP:
  Full DP noise scale:   σ_full  = σ_DP (applied to all |θ| coordinates)
  FTGD effective noise:  σ_FTGD  = σ_DP (applied only to ||g_fair|| subspace)

  Since ||g_fair||_2 << ||g||_2 in practice (fairness is ~20% of total loss),
  the effective signal-to-noise ratio for task gradient is preserved.
```

### 5.4 Implementation Notes

```
Two-backward-pass method (cleaner, slightly slower):

  optimizer.zero_grad()
  loss_total = L_fraud + lambda * L_fair
  loss_total.backward(retain_graph=True)
  g = [p.grad.clone() for p in model.parameters()]

  optimizer.zero_grad()
  (lambda * L_fair).backward()
  g_fair = [p.grad.clone() for p in model.parameters()]

  # Flatten to 1D vectors for decomposition
  g_vec      = torch.cat([gi.flatten() for gi in g])
  g_fair_vec = torch.cat([gi.flatten() for gi in g_fair])

  # Decompose
  dot      = torch.dot(g_vec, g_fair_vec)
  norm_sq  = torch.dot(g_fair_vec, g_fair_vec) + 1e-12
  g_task_vec = g_vec - (dot / norm_sq) * g_fair_vec

  # Clip + noise g_fair
  norm_fair = g_fair_vec.norm(2)
  g_fair_clipped = g_fair_vec / max(1.0, norm_fair / C)
  sigma = C * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
  g_fair_noisy = g_fair_clipped + torch.randn_like(g_fair_clipped) * sigma

  return g_task_vec, g_fair_noisy
```

---

## 6. Component 3 — BFWA: Bi-Objective Frank-Wolfe Aggregation

### 6.1 Motivation

After receiving gradients from all clients, FedAvg computes a simple weighted average by data size. This allows clients whose local data distribution is demographically skewed (high DPD) to dominate the global model update, propagating bias globally.

BFWA frames aggregation as a **constrained optimization problem**: maximize aggregate fraud detection performance subject to a hard upper bound τ on global demographic parity difference. The Frank-Wolfe algorithm solves this efficiently on the probability simplex without projections.

### 6.2 Optimization Formulation

```
Variables: w ∈ R^K    (aggregation weight vector)

Objective:
  maximize    Σ_k w_k · Perf_k             (maximize weighted average performance)
  subject to  Σ_k w_k · DPD_k  ≤ τ        (global fairness constraint)
              w ∈ Δ_K                      (probability simplex)

Equivalently (minimization form):
  minimize    F(w) = −Σ_k w_k · Perf_k + μ · max(0, Σ_k w_k · DPD_k − τ)
  subject to  w ∈ Δ_K

where μ is a dual variable updated via gradient ascent (see Algorithm 3).
```

### 6.3 Algorithm

```
Algorithm 3: BFWA — Bi-Objective Frank-Wolfe Aggregation
================================================================
Runs at: Server, once per communication round
Replaces: FedAvg weighted averaging

Input:
  {g_task^k}_{k=1}^K  : clean task gradients      each shape: [|θ|]
  {g̃_fair^k}_{k=1}^K  : noisy fairness gradients   each shape: [|θ|]
  {DPD_k}_{k=1}^K     : local fairness violations  each scalar ∈ [0,1]
  {Perf_k}_{k=1}^K    : local AUC-ROC scores        each scalar ∈ [0,1]
  θ_{t-1}             : current global parameters   shape: [|θ|]
  τ                   : fairness budget              scalar ∈ [0,1]
  T_fw                : Frank-Wolfe inner iterations (default: 20)
  η_dual              : dual variable step size      (default: 0.1)
  η_global            : global learning rate          (default: 0.01)

Output:
  θ_t                 : updated global parameters   shape: [|θ|]
================================================================

--- Initialize ---
w    = [1/K, 1/K, ..., 1/K]   ∈ R^K     (uniform weights)
μ    = 0.0                                (Lagrange multiplier for DPD constraint)

--- Frank-Wolfe iterations ---
For t_fw = 1, ..., T_fw:

  --- Compute constraint violation ---
  DPD_global = Σ_k w[k] · DPD_k           scalar
  violation  = DPD_global − τ              scalar  (positive = constraint violated)

  --- Compute gradient of Lagrangian w.r.t. w ---
  For each k:
    ∇F_k = −Perf_k + μ · DPD_k            scalar
  ∇F = [∇F_1, ..., ∇F_K]                 shape: [K]

  --- Linear Minimization Oracle (LMO) on simplex ---
  # Minimize s^T · ∇F over s ∈ Δ_K
  # Closed form: put all weight on minimizing index
  k* = argmin_k ∇F[k]
  s  = zero vector of length K
  s[k*] = 1.0                              (unit vector in direction k*)

  --- Frank-Wolfe update ---
  γ_t = 2.0 / (t_fw + 2)                  (step size schedule, decaying)
  w   = (1 − γ_t) · w + γ_t · s

  --- Dual variable update (gradient ascent on μ to enforce constraint) ---
  μ   = max(0.0, μ + η_dual · violation)

--- Compute global gradient update ---
Δθ = Σ_k w[k] · (g_task^k + g̃_fair^k)    shape: [|θ|]

--- Update global model ---
θ_t = θ_{t-1} − η_global · Δθ

Return θ_t
```

### 6.4 Frank-Wolfe Convergence Notes

```
Convergence guarantee (convex case):
  F(w^t) − F(w*) ≤ 2 · C_F / (t + 2)

where C_F is the curvature constant of F over Δ_K.

In practice:
  T_fw = 10 to 20 iterations is sufficient.
  Monitor: duality_gap = (w - s*)^T · ∇F  (should approach 0)

Stopping criterion (optional, use instead of fixed T_fw):
  Stop when duality_gap < 1e-4 OR t_fw > T_fw

Simplex feasibility:
  The Frank-Wolfe update w = (1-γ)w + γs ALWAYS stays in Δ_K
  if w and s are both in Δ_K (convex combination).
  No projection is needed — this is the key computational advantage.
```

### 6.5 Degenerate Cases

```
Case 1: All clients violate fairness (DPD_k > τ for all k)
  → BFWA will assign highest weight to least-violating client.
  → Global model may still violate τ, but minimally so.
  → Server should log a warning: "Fairness budget τ may be too tight."
  → Recommend increasing τ or reducing λ.

Case 2: One client has DPD_k = 0 (perfectly fair, possibly tiny dataset)
  → Frank-Wolfe will put w[k] → 1 for that client.
  → Safeguard: enforce minimum weight floor w[k] ≥ w_min = 1/(5K)
    and re-normalize after each FW step.

Case 3: K = 1 (single client)
  → Skip BFWA entirely, use direct gradient update.
```

---

## 7. Full Training Loop: FairFedGNN

```
Algorithm 4: FairFedGNN — Complete Federated Training
================================================================
Input:
  K, T, E, L, d, d_h   : architecture hyperparameters
  λ, τ, ε, δ, C         : fairness and privacy hyperparameters
  η_local, η_global     : learning rates
  {G_k}_{k=1}^K         : local graphs at each client (NEVER shared)

Output:
  θ*                     : trained global FairFedGNN model
================================================================

--- SERVER: Initialize global model ---
θ_0 = random_init()   (Xavier initialization for W^(l), zeros for biases,
                        0.5 for β^(l), small random for a^(l))

--- Main federated loop ---
For round t = 1, ..., T:

  1. SERVER broadcasts θ_{t-1} to all K clients.

  2. CLIENT k (runs in parallel for all k):

     a. Load global model: θ_k ← θ_{t-1}

     b. Initialize local optimizer:
        opt_k = AdamW(θ_k, lr=η_local, weight_decay=1e-4)

     c. For local epoch e = 1, ..., E:

          i.  Sample mini-batch B ⊆ V_k   (batch_size = 512 nodes + 2-hop subgraph)

          ii. Forward pass with FSER (Algorithm 1):
              H_k   = FairGNN_FSER(G_k[B], S_k; θ_k)      shape: [|B|, d_h]
              Ŷ_k   = sigmoid(H_k @ W_cls + b_cls)          shape: [|B|]

          iii. Compute FTGD (Algorithm 2):
              (g_task^k, g̃_fair^k) = FTGD(θ_k, G_k[B], Y_k[B], S_k[B], λ, C, ε, δ)

          iv. Update local model:
              θ_k ← θ_k − η_local · (g_task^k + g̃_fair^k)

          v.  Clip β^(l) to [0, 5] for all l   (FSER stability)

     d. Compute final local metrics on full local validation set V_k^val:
          Ŷ_k_val  = FairFedGNN_forward(G_k^val; θ_k)
          DPD_k    = |mean(Ŷ_k_val[S_k^val=0]) − mean(Ŷ_k_val[S_k^val=1])|
          Perf_k   = AUC_ROC(Y_k^val, Ŷ_k_val)

     e. Compute final local gradients for this round:
          # Re-run one backward pass on validation set to get round-level gradients
          (g_task^k, g̃_fair^k) = FTGD(θ_k, G_k^val, Y_k^val, S_k^val, λ, C, ε, δ)

  3. CLIENTS → SERVER: each client sends (g_task^k, g̃_fair^k, DPD_k, Perf_k)

  4. SERVER: Apply BFWA (Algorithm 3):
     θ_t = BFWA({g_task^k}, {g̃_fair^k}, {DPD_k}, {Perf_k}, θ_{t-1}, τ)

  5. SERVER: Log global metrics:
     DPD_global_t = Σ_k (n_k / Σ n_j) · DPD_k   (data-size weighted average)
     Perf_global_t = Σ_k (n_k / Σ n_j) · Perf_k
     Print: f"Round {t}: Global AUC={Perf_global_t:.4f}, DPD={DPD_global_t:.4f}"

--- Early stopping ---
  If t ≥ T_warmup (default: T/4):
    If DPD_global_t < τ AND Perf_global_t plateaus (< 0.001 improvement for 5 rounds):
      Break early, save θ_t as θ*

Return θ_T  (or θ* if early stopped)
```

---

## 8. Loss Functions

### 8.1 Task Loss — Weighted Binary Cross-Entropy

Fraud detection is class-imbalanced (typically 1–5% fraud). Use weighted BCE:

```
L_fraud = − (1/|B|) Σ_{i ∈ B} [ w_pos · Y_i · log(Ŷ_i + ε_num)
                                + w_neg · (1 − Y_i) · log(1 − Ŷ_i + ε_num) ]

where:
  w_pos = |B| / (2 · |{i : Y_i = 1}|)    (weight for fraud class)
  w_neg = |B| / (2 · |{i : Y_i = 0}|)    (weight for legit class)
  ε_num = 1e-7                             (numerical stability)

If a batch has zero fraud samples: set w_pos = 10.0, w_neg = 1.0 as fallback.
```

### 8.2 Fairness Loss — Differentiable DPD

The standard DPD uses hard thresholding (non-differentiable). Use soft version:

```
L_fair = | mean(Ŷ[S=0]) − mean(Ŷ[S=1]) |

Expanded:
  n_0  = |{i : S_i = 0}|
  n_1  = |{i : S_i = 1}|
  μ_0  = (1/n_0) · Σ_{i: S_i=0} Ŷ_i
  μ_1  = (1/n_1) · Σ_{i: S_i=1} Ŷ_i
  L_fair = |μ_0 − μ_1|

Gradient:
  ∂L_fair / ∂Ŷ_i = sign(μ_0 − μ_1) · (1/n_0)   if S_i = 0
  ∂L_fair / ∂Ŷ_i = −sign(μ_0 − μ_1) · (1/n_1)  if S_i = 1

This is differentiable everywhere except μ_0 = μ_1 (zero-measure event in practice).
```

### 8.3 Total Loss

```
L_total = L_fraud + λ · L_fair

λ controls the accuracy-fairness trade-off:
  λ = 0.0   → pure fraud detection (no fairness)
  λ = 0.5   → balanced trade-off  (recommended starting point)
  λ = 2.0   → strong fairness enforcement
  λ > 5.0   → typically degrades accuracy unacceptably
```

### 8.4 FSER Regularization (optional)

Add a penalty to prevent β^(l) from collapsing to 0 (disabling FSER):

```
L_beta = −γ_beta · Σ_l log(β^(l) + 1e-3)    (log-barrier)

L_total_final = L_total + L_beta

γ_beta = 0.01   (small, just to keep β > 0)
```

---

## 9. Model Architecture

### 9.1 Full GNN Architecture

```
Input: node features X ∈ R^{n × d}

Layer 0 — Input projection:
  H^(0) = Linear(X)  + BatchNorm1d    shape: [n, d_h]
  H^(0) = ELU(H^(0))
  H^(0) = Dropout(H^(0), p=0.2)

Layer 1 to L — FSER-GAT layers:
  For l = 1, ..., L:
    H^(l) = FSER_GAT(H^{l-1}, E, S; W^(l), a^(l), β^(l))   [Algorithm 1]
    H^(l) = BatchNorm1d(H^(l))
    H^(l) = ELU(H^(l))
    H^(l) = Dropout(H^(l), p=0.2)
    # Residual connection from l ≥ 2:
    If l ≥ 2: H^(l) = H^(l) + H^{l-1}

Skip connection (concatenate all layer outputs):
  H_skip = Concat([H^(0), H^(1), ..., H^(L)])   shape: [n, d_h * (L+1)]
  H_out  = Linear(H_skip, d_h)                    shape: [n, d_h]
  H_out  = ELU(H_out)

Classifier head:
  logits = H_out @ W_cls + b_cls   shape: [n, 1]   (W_cls ∈ R^{d_h × 1})
  Ŷ      = sigmoid(logits)          shape: [n]

Default dimensions:
  d    = 64    (input features, depends on dataset)
  d_h  = 128   (hidden dimension)
  L    = 3     (GNN layers)
```

### 9.2 Multi-Head Extension (Optional)

For stronger expressive power, use H=4 attention heads in FSER-GAT:

```
For head h = 1, ..., H:
  ᾱ^h_ij = FSER attention (Algorithm 1) with W^(l,h), a^(l,h), β^(l)
  z^h_i   = Σ_{j ∈ N(i)} ᾱ^h_ij · W^(l,h) · h_j^{l-1}

Multi-head output:
  h_i^(l) = ELU( concat( z^1_i, ..., z^H_i ) )    shape: [H * d_h/H] = [d_h]

Note: β^(l) is shared across heads (one fairness coefficient per layer).
```

---

## 10. Hyperparameters Reference

| Hyperparameter | Symbol | Default | Range | Description |
|---|---|---|---|---|
| Communication rounds | `T` | 100 | 50–200 | Total federated rounds |
| Local epochs | `E` | 5 | 1–10 | Local epochs per round |
| GNN layers | `L` | 3 | 2–4 | Depth of GNN |
| Hidden dim | `d_h` | 128 | 64–256 | Node embedding size |
| Attention heads | `H` | 4 | 1–8 | Multi-head attention |
| FSER init | `β^(l)` | 0.5 | learned | Fairness suppression init |
| Fairness weight | `λ` | 0.5 | 0.1–2.0 | Trade-off control |
| Fairness budget | `τ` | 0.05 | 0.01–0.15 | Max allowed global DPD |
| DP epsilon | `ε` | 1.0 | 0.1–10.0 | Privacy budget (smaller = more private) |
| DP delta | `δ` | 1e-5 | 1e-7–1e-3 | DP failure probability |
| DP clip norm | `C` | 1.0 | 0.1–10.0 | Gradient clipping bound |
| Local LR | `η_local` | 0.001 | 1e-4–1e-2 | AdamW local learning rate |
| Global LR | `η_global` | 0.01 | 1e-3–0.1 | Server aggregation learning rate |
| FW iterations | `T_fw` | 20 | 5–50 | Frank-Wolfe inner steps |
| Dual step | `η_dual` | 0.1 | 0.01–1.0 | Dual variable update step |
| Batch size | `B` | 512 | 256–2048 | Nodes per mini-batch |
| Dropout | `p` | 0.2 | 0.1–0.5 | Dropout rate |
| Weight decay | `wd` | 1e-4 | 1e-5–1e-3 | L2 regularization |
| Min weight floor | `w_min` | 1/(5K) | — | BFWA weight lower bound |
| Beta regularizer | `γ_beta` | 0.01 | 0.001–0.1 | FSER β log-barrier strength |
| Warmup rounds | `T_warmup` | T/4 | — | Rounds before early stopping |

---

## 11. Data Preprocessing

### 11.1 Graph Construction from Transaction Data

```
If starting from raw transaction records:

Nodes: each unique user/account is a node
Edges: two nodes are connected if they share a transaction

Node features X[i] should include:
  - Transaction statistics: avg_amount, max_amount, tx_count, tx_frequency
  - Temporal features: days_since_first_tx, days_since_last_tx
  - Network features: degree, clustering_coefficient (compute locally)
  - Behavioral: distinct_counterparties, unique_merchants

DO NOT include sensitive attributes (S_k) in X_k.
Sensitive attributes are kept separate and only used in fairness computations.
```

### 11.2 Feature Normalization

```
Per client, normalize each feature independently:
  X_k[:, j] = (X_k[:, j] − mean_j) / (std_j + 1e-6)

Compute mean_j and std_j on training nodes only.
Apply same statistics to validation and test nodes.
```

### 11.3 Train/Val/Test Split (per client)

```
Temporal split (preferred for fraud, avoids data leakage):
  Sort nodes by timestamp of first transaction
  Train: first 60% of nodes by time
  Val:   next 20%
  Test:  last 20%

If no timestamps available: random stratified split (60/20/20),
  stratified by Y_k to preserve fraud class ratio.
```

### 11.4 Mini-Batch Subgraph Sampling

```
For each batch of B seed nodes:
  1. Select B nodes from training set (balanced: B/2 fraud, B/2 legit if possible)
  2. Sample 2-hop neighborhood for each seed node
     (use neighbor sampling: max 10 neighbors per hop to limit memory)
  3. Induce subgraph on sampled nodes
  4. All nodes in subgraph get FSER treatment, but only seed node losses counted
```

### 11.5 Sensitive Attribute Handling

```
Examples of sensitive attributes:
  Gender:        binary 0/1 directly
  Income tier:   binarize at median (below=0, above=1)
  Geography:     urban=1, rural=0  OR  use one-vs-rest for multiple regions
  Age:           binarize at 35 years (young=0, older=1)

Requirement: S_k[i] ∈ {0, 1} for all i.

If sensitive attributes are UNAVAILABLE for some nodes:
  Set S_k[i] = -1 (unknown).
  Exclude nodes with S_k[i] = -1 from L_fair computation and DPD computation.
  These nodes still participate in L_fraud and FSER forward pass.
```

---

## 12. Evaluation Metrics

### 12.1 Fraud Detection Performance Metrics

```
Primary:
  AUC-ROC = Area under ROC curve
           (threshold-independent, handles class imbalance well)

Secondary:
  F1 = 2 · Precision · Recall / (Precision + Recall)  [at threshold 0.5]

  AP  = Average Precision (area under Precision-Recall curve)
        (more informative than AUC for severe imbalance)

  FPR@10%TPR = False positive rate when true positive rate = 10%
               (operational metric: how many legit users flagged per 10 fraudsters caught)
```

### 12.2 Fairness Metrics

```
1. Demographic Parity Difference (DPD):  [Primary fairness metric]
   DPD = |P(Ŷ=1|S=0) − P(Ŷ=1|S=1)|    (lower is fairer)

2. Equal Opportunity Difference (EOD):
   EOD = |TPR(S=0) − TPR(S=1)|
       = |P(Ŷ=1|Y=1,S=0) − P(Ŷ=1|Y=1,S=1)|

3. Equalized Odds:
   EO = max(|TPR(S=0)−TPR(S=1)|, |FPR(S=0)−FPR(S=1)|)

Interpretation thresholds (common in fairness ML literature):
  DPD < 0.05  → acceptable fairness
  DPD < 0.10  → marginal
  DPD > 0.10  → significant unfairness
```

### 12.3 Privacy Metric

```
Reported as: (ε, δ)-DP with ε = [reported value], δ = 1e-5

Compute ε for a given number of rounds T and local steps E using
the moments accountant or Rényi DP composition:
  Total ε ≈ ε_per_step · √(T · E · ln(1/δ))  [rough bound]
  Use the `autodp` Python library for tight composition.
```

### 12.4 Accuracy-Fairness Trade-off Curve

```
Vary λ ∈ {0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0}
Train one model per λ value.
Plot:  x-axis = DPD,  y-axis = AUC-ROC
This curve shows the Pareto frontier of the accuracy-fairness trade-off.
The operating point is where the curve enters the acceptable DPD zone (< 0.05).
```

---

## 13. Federated Communication Protocol

### 13.1 Message Format

```
Client → Server (per round):
  {
    "client_id":  int,
    "round":      int,
    "g_task":     float32 array of shape [|θ|],   # clean task gradient
    "g_fair":     float32 array of shape [|θ|],   # DP-noised fairness gradient
    "DPD":        float32 scalar,
    "Perf":       float32 scalar,
    "n_train":    int   # number of training nodes (for logging only)
  }

Server → Client (per round):
  {
    "round":      int,
    "theta":      float32 array of shape [|θ|],   # updated global parameters
  }
```

### 13.2 Compression (optional, for large models)

```
To reduce communication overhead:
  - Gradient sparsification: send only top-p% of gradient coordinates by magnitude
    (p = 1% typically, reconstruct with error feedback)
  - Quantization: quantize gradients to 8-bit integers before sending
  - These are optional engineering optimizations, not required for correctness.
```

### 13.3 Synchronous vs Asynchronous

```
Default: SYNCHRONOUS
  Server waits for all K clients each round.
  Guarantees BFWA sees all clients' metrics.

For K > 50 clients or unreliable connections: SEMI-SYNCHRONOUS
  Server proceeds when at least ceil(0.8 * K) clients respond (80% participation).
  Use only received clients in BFWA.
  Set w_k = 0 for non-responding clients.
```

---

## 14. Implementation Notes and Edge Cases

### 14.1 Numerical Stability Checklist

```
1. Softmax overflow: use log-sum-exp trick for FSER attention normalization
2. DPD with empty group: if n_0 = 0 or n_1 = 0, set DPD_k = 0 (skip fairness)
3. Gradient norm = 0: add 1e-12 to denominator in orthogonal decomposition
4. β going negative: hard clip to [0, 5] after every optimizer step
5. μ (dual variable) overflow: hard clip to [0, 100]
6. g_fair_clipped all zeros: if norm_fair < 1e-8, skip DP noise addition
```

### 14.2 Reproducibility

```
Set seeds before each experiment:
  torch.manual_seed(42)
  numpy.random.seed(42)
  random.seed(42)
  torch.backends.cudnn.deterministic = True

DP noise is intentionally random — AUC results will vary slightly across runs.
Report mean ± std over 5 random seeds.
```

### 14.3 Memory Estimates

```
For n_k = 100,000 nodes, m_k = 500,000 edges, d_h = 128, L = 3:
  Node embeddings per layer: 100,000 × 128 × 4 bytes = 51 MB
  Edge attention scores:      500,000 × 4 bytes = 2 MB
  Model parameters:           ~50,000 × 4 bytes = 0.2 MB
  Gradient storage:           same as parameters = 0.2 MB

Total per client: ~110 MB per forward pass (manageable on GPU with 8GB VRAM)
For larger graphs: use neighbor sampling (Section 11.4) to limit memory.
```

### 14.4 Minimum Dataset Requirements

```
For FSER to function correctly:
  Each sensitive group must have at least 50 nodes in each mini-batch.
  If n_0 < 50 or n_1 < 50 in a batch: skip FSER fairness score (use standard GAT).

For FTGD to function correctly:
  At least 10 fraud samples per batch (Y=1) for meaningful L_fraud gradient.
  At least 10 samples per sensitive group per batch for meaningful L_fair gradient.

For BFWA to function correctly:
  At least K ≥ 2 clients.
  At least one client must have DPD_k ≤ τ; otherwise τ is unachievable.
```

### 14.5 Ablation Study Template

```
To verify each component's contribution, train the following variants:

Variant 1: Baseline FedAvg + plain GAT          (no FSER, no FTGD, no BFWA)
Variant 2: FedAvg + FSER only                   (FSER active, no FTGD, no BFWA)
Variant 3: DP-FedAvg + FSER                     (full DP on gradient, no FTGD split)
Variant 4: FedAvg + FSER + FTGD                 (no BFWA, uniform aggregation)
Variant 5: FairFedGNN full (FSER + FTGD + BFWA) (proposed)

Expected result: each added component reduces DPD while Variant 5 best
balances AUC-ROC and DPD simultaneously.
```

---

*Document version: 1.0 | FairFedGNN Algorithm Specification*  
*All algorithms are self-contained. No external references needed for implementation.*