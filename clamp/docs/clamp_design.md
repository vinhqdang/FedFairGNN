# TrustFedGNN-R2: design specification

Status: approved design for the resubmission of TrustFedGNN (rejected at
Neurocomputing, NEUCOM-D-26-20010). This document is the single source of truth
for the implementation (`src/`), the experiments (`experiments/r2/`) and the
manuscript (`manuscript_trustfedgnn/`).

## 0. Why the server side is redesigned

The submitted version weighted clients with BFWA, which reads a disparity value
that each client computes and reports about itself. Three problems follow:

1. A client can report any value. The weight is then decided by a number the
   server cannot check.
2. Per-client privatised reports are absolute differences of noisy means. The
   noise is folded to be non-negative before it reaches the server, so it does
   not average out across clients.
3. The robustness theorem covered the coordinate median alone, but the
   evaluated pipeline (`robust_bfwa`) is median screening followed by BFWA
   weighting, and it was the least fair rule at 40% corruption (reviewer 1).

R2 removes per-client weighting altogether. **The server only ever receives
sums over clients, every summand is range-checked, and the fairness constraint
is enforced through a single global Lagrange multiplier that is sent back to
the clients.** The guarantees in Section 3 are stated for this server-side
aggregation and Lagrangian-control pipeline as a whole, not for one component
of it; their scope (what each one does not cover) is stated with each.

Scope boundary: R2 does not use a server-held dataset and does not score
clients against a server reference gradient. That design space belongs to a
different project in this repository and must not be imported here.

## 1. Notation

- K clients; client k holds subgraph G_k with train/val/test node masks, labels
  y in {0,1} and sensitive attribute s in {0,1}.
- theta^t: global parameters at round t. p_theta(v) in (0,1): predicted
  probability for node v.
- Group statistics of client k at round t, computed with the *received*
  global model theta^t on the client's **validation** nodes, using an
  **s-blind forward pass** (the backbone is run with the sensitive attribute
  replaced by a constant, so no node's s influences any prediction; s is used
  only to assign each node's prediction to a group):
  - N_{k,a} = |{v in val_k : s_v = a}|
  - S_{k,a} = sum over those v of p_{theta^t}(v)
- tau: fairness budget on the global demographic-parity gap.
- C: update norm bound. n_cap: public upper bound on N_{k,a}, set per dataset
  to the largest per-client validation group size rounded up to a multiple of
  50 (a public protocol constant, so no client ever needs to subsample).

## 2. Algorithm (one round t)

Server state: theta^t, lambda^t >= 0 (lambda^0 = 0), direction d^t in
{-1, 0, +1} (d^0 = 0).

**Broadcast.** The server sends (theta^t, lambda^t, d^t) to every client.

**Client k (honest).**
1. Compute (S_{k,0}, S_{k,1}, N_{k,0}, N_{k,1}) with theta^t on val_k
   (no grad, eval mode, s-blind forward pass). If a group exceeds n_cap,
   keep a fixed subset chosen once per client from the seed (deterministic,
   identical every round).
2. Local training for E epochs from theta^t on train_k with loss
   `BCE_weighted(p, y) + lambda^t * d^t * (m_1 - m_0)`
   where m_a is the mean prediction over train nodes with s = a (0 if the
   group is empty). Note the term is **signed**: with d^t = sign of the global
   gap, minimising it moves the two group rates toward each other. When
   lambda^t = 0 or d^t = 0 this is plain weighted BCE.
3. Pseudo-gradient Delta_k = theta^t - theta_k; clip:
   Delta_k <- Delta_k * min(1, C / ||Delta_k||_2).
4. Submit (Delta_k, S_{k,0}, S_{k,1}, N_{k,0}, N_{k,1}) to secure aggregation.

**Validity check (simulated verifiable secure aggregation).** A submission is
accepted iff ||Delta_k||_2 <= C(1 + 1e-6) and 0 <= S_{k,a} <= N_{k,a} <= n_cap
for a in {0,1}. Rejected submissions are dropped entirely. In deployment this
check is performed with zero-knowledge range/norm proofs inside secure
aggregation (Prio, RoFL); in our simulation the server code applies the check
and then touches only the sums. K' = number of accepted submissions (public).

**Server (sees sums only).** If K' = 0 the round is skipped (theta, lambda
and d are unchanged).
1. U = sum_k Delta_k + xi_u, with xi_u ~ N(0, sigma_u^2 I) (sigma_u = 0 by
   default; see T1b).
2. S_a = sum_k S_{k,a} + xi_{S,a}, N_a = sum_k N_{k,a} + xi_{N,a}, with
   independent N(0, sigma_s^2) noise.
3. Global group rates r_a = clip(S_a / max(N_a, 1), 0, 1); signed gap
   g = r_1 - r_0; estimate Dhat = |g|.
4. theta^{t+1} = theta^t - U / K'.
5. lambda^{t+1} = max(0, lambda^t + eta_lambda * (Dhat - tau));
   d^{t+1} = sign(g) if |g| > tau/2, else 0 (dead zone: no fairness push while
   the estimated gap is within half the budget; this also limits how an
   adversary can flip the direction, see T3).

Log every round: lambda, Dhat, g, K', number rejected, ||U||.

## 3. Guarantees (to appear in the manuscript with these proofs)

### T1. Differential privacy of the statistics channel

Scope: T1 protects the released group statistics. It does **not** cover the
update channel (train-node attributes enter the local loss through m_0, m_1,
and the backbone may read s during training); that channel is covered only by
secure aggregation, which is a cryptographic, not a differential-privacy,
guarantee (see "Server visibility" below). Nothing is said about "privacy of
the pipeline" as a whole.

Neighbouring relation: two datasets that differ in one validation node's
sensitive attribute, or in the presence of one validation node, with all
training data fixed. Because statistics are computed with an s-blind forward
pass, toggling s_v changes no prediction; it only moves p_v (in [0,1]) from one
group's sum to the other's and one unit between the counts, so the 4-vector
(S_0, S_1, N_0, N_1) of that client changes by at most sqrt(2 p_v^2 + 2) <= 2
in L2 norm. Adding or removing v changes it by sqrt(p_v^2 + 1) <= sqrt(2).
Hence Delta_2 = 2. (If a validation node is also a message-passing neighbour
of other validation nodes, its *features* still influence their predictions,
but features are not the protected attribute; the s-blind pass is what
removes the attribute's influence on other nodes.)

Each round releases the client sums plus N(0, sigma_s^2) per coordinate: a
Gaussian mechanism with noise multiplier z = sigma_s / Delta_2, which is
(alpha, alpha Delta_2^2 / (2 sigma_s^2))-RDP for every alpha > 1. theta^t,
lambda^t and d^t are post-processing of earlier releases and of training data
(fixed under the neighbouring relation), so adaptive composition applies. The
same validation nodes are used in every round, so there is no subsampling
amplification: R rounds give alpha R Delta_2^2 / (2 sigma_s^2), converted to
(epsilon, delta) with the standard RDP conversion. **One release per round,
not per local epoch.** Implementation: sigma_s = 2 z with
z = calibrate_noise_multiplier(dp_epsilon, steps=R, delta=dp_delta) (no
sampling rate).

Noise placement. The simulation adds the noise once to the sum, which is
central DP and assumes the noise is added by a trusted party. In a secure-
aggregation deployment each client adds discrete-Gaussian noise of variance
sigma_s^2 / (K - f_max) (finite-field arithmetic requires a discrete
mechanism, Kairouz et al. 2021); then the sum carries at least sigma_s^2 even
if up to f_max clients (Byzantine or dropped) omit their share.

T1b (ablation only). With sigma_u > 0, U alone is a Gaussian mechanism with
client-level sensitivity C (add/remove one client's clipped update). This
protects U only; the statistics channel is separately accounted (T1) at
node level, not client level. No client-level claim for the full model
sequence is made.

Server visibility of individual updates. Under secure aggregation with a
threshold of honest, non-colluding clients and dropout handling (Bonawitz et
al. 2017), the server learns only the sum of accepted updates. This does not
hold if K' is very small or if K' - 1 clients collude with the server. In our
code the validity check and the sums are simulated in plaintext; the paper
says so. The sum U itself still depends on s; an attribute-inference probe on
U (experiment E4) measures that residual channel, and no claim of zero
leakage is made.

### T2. Bounded per-round influence of Byzantine updates

Fix the round's state (theta^t, lambda^t, d^t). Let A be the accepted
submissions, K' = |A|, B the f' accepted Byzantine ones, H = A \ B, and let
the reference be the honest-only clipped mean. If K' > 0,

|| U/K' - mean_{k in H} Delta_k || = (f'/K') || mean_B Delta_k - mean_H Delta_k ||
                                    <= 2 f' C (1 + 1e-6) / K' <= 2 f C (1 + 1e-6) / (K - f),

using K' >= K - f (all honest clients are accepted). This is a **per-round**
bound against a same-state counterfactual. It does not compose into a
trajectory bound by summation, because later honest updates are computed at
the perturbed parameters and the adversary also acts through (lambda, d);
we do not claim a T-round bound.

### T3. Bounded influence on the fairness estimate

Let S^H_a, N^H_a be the honest totals and S^B_a in [0, N^B_a],
N^B_a in [0, f' n_cap] the accepted Byzantine totals (validity check).
Without noise,

|S_a/N_a - S^H_a/N^H_a| = N^B_a |S^B_a/N^B_a - S^H_a/N^H_a| / (N^H_a + N^B_a)
                       <= f n_cap / (N^H_a + f n_cap)  =: b_a,

since the rate difference is at most 1 and x/(N + x) is increasing in x.
|.| and the clip to [0,1] are 1-Lipschitz, so |Dhat - D^H| <= b_0 + b_1.
With noise: on the event that all four noise terms satisfy |xi| <= e,
e = sigma_s sqrt(2 ln(8/beta)) (probability >= 1 - beta), and if
N^H_a > 3e, each group rate is additionally perturbed by at most
2e / (N^H_a - e), so

|Dhat - D^H| <= sum_a [ b_a + 2e / (N^H_a - e) ]   with probability >= 1 - beta.

Report the numeric value of this bound for every dataset (it can be weak on
small graphs; say so). No per-client weights exist, so there is nothing to
capture. Consequence for lambda: within one round and the same lambda^t, the
projection is 1-Lipschitz, so the adversary shifts lambda^{t+1} by at most
eta_lambda times the bound. Direction: with the dead zone, d^{t+1} differs
from the honest direction only if the bound exceeds |g^H| - tau/2; when the
honest gap is larger than tau/2 plus the bound, the adversary cannot flip d.

### T4. Constraint-violation certificate (constant step)

From lambda^{t+1} = max(0, lambda^t + eta (Dhat^t - tau)) >=
lambda^t + eta (Dhat^t - tau) and lambda^0 = 0, summing over t = 0..T-1:

(1/T) sum_t (Dhat^t - tau) <= lambda^T / (eta T).

This is a certificate, not a convergence theorem. It holds for any primal
sequence (no convexity, no exact minimisation) and for the constant step used
in the experiments, and lambda^T is logged, so it is computable from the run.
Limits to state: it bounds an average over iterates, so violations in some
rounds can be offset by slack in others; it bounds the validation-estimated,
noisy Dhat (which is biased upward by the absolute value), not the test DPD;
it is informative only when lambda^T << eta T. Whether lambda^T stays bounded
is an empirical, a-posteriori fact; report lambda^T / (eta T) for every run.

Remark (textbook, only under its assumptions). If the task loss F and the
constraint D were convex in theta, each round minimised the Lagrangian
exactly, and D were observed exactly, q(lambda) = min_theta F + lambda (D - tau)
is concave (a minimum of affine functions of lambda) and the update above is
projected supergradient ascent with |D - tau| <= 1, giving
q* - max_t q(lambda^t) <= (lambda*^2 + eta^2 T)/(2 eta T). None of these
assumptions holds for a GNN; the remark is context, not a claim.

## 4. Implementation spec (for the implementer)

All changes are **additive**; do not change behaviour of existing methods.

1. `src/config.py` - new dataclass fields (defaults in brackets):
   - `fair_mode: str = "none"` - "lagrangian" activates step 2 of the client.
   - `r2_clip_norm: float = 1.0` (C)
   - `r2_lambda_step: float = 0.5` (eta_lambda)
   - `r2_n_cap: int = 0` - public cap on per-client, per-group counts. 0 means
     "derive at setup": the largest per-client validation group size rounded up
     to a multiple of 50 (computed once from the partition, then treated as a
     public constant). Never 100000 (T3 becomes vacuous).
   - `r2_stat_dp: bool = True` - noise on statistics (sigma_s from T1)
   - `r2_update_noise: float = 0.0` (sigma_u)
   Respect the `_FIELD_NAMES` allowlist (fields must be declared).
2. `src/federated/client.py`
   - `Client.group_stats(self) -> dict` returning S0,S1,N0,N1 with the current
     model (call before training), computed with an **s-blind forward pass**
     (reuse the existing s-blind release machinery, `_release_pred` /
     `_S_DEPENDENT_BACKBONES`: the sensitive attribute is masked/neutralised in
     the input so toggling s_v changes no prediction). If a client has more than
     n_cap validation nodes in a group, use a deterministic fixed subset
     (first n_cap by node index).
   - `Client.lagrange = (lam, direction)` attribute, default (0.0, 0).
   - In `train()`, when `cfg.fair_mode == "lagrangian"` and the client is not a
     fairness-poison attacker, use the signed loss of Section 2 for every local
     epoch (full-batch path; the sampled path may raise NotImplementedError).
3. `src/federated/aggregation.py` - new branch `method == "secagg_lagrangian"`
   implemented in a separate pure function
   `secagg_lagrangian(updates, stats, state, *, C, tau, eta_lambda, n_cap,
   sigma_s, sigma_u, generator)` returning (g_agg, info). It applies the
   validity check, sums, noise, rates, lambda/direction update (persisted in
   `state`). If K' = 0 (everything rejected) the round is skipped: g_agg = 0,
   lambda and direction unchanged. Direction uses a dead zone:
   d = sign(g) if |g| > tau/2 else 0. It returns `info` with keys: weights (uniform 1/K' over accepted,
   0 for rejected, for logging only), lambda, direction, dhat, gap, n_accepted,
   n_rejected. `aggregate()` gains an optional `stats=None` argument passed only
   to this branch.
4. `src/federated/trainer.py`
   - When aggregator == "secagg_lagrangian": before local training, set each
     client's `lagrange` from `self._agg_state` and collect `group_stats()`;
     after training, Byzantine clients' poisoned updates are rescaled to norm C
     if larger (norm-aware adversary - it knows C); pass stats to aggregate.
   - Compute sigma_s once in `__init__`: sigma_s = 2 * z with
     z = calibrate_noise_multiplier(dp_epsilon, steps=rounds, delta=dp_delta),
     no sampling rate (one release per round). sigma_s = 0 if r2_stat_dp False.
   - Resolve r2_n_cap = 0 to the per-dataset public constant described above.
   - Log lambda, dhat, gap, n_accepted, n_rejected into the round record.
5. `src/federated/attacks.py` - new attack `"stat_falsify"`: Byzantine clients
   train honestly but submit statistics with N_{k,a} = n_cap and group rates
   swapped to the opposite of the honest global rates (omniscient adversary that
   wants the estimated gap to shrink, so lambda falls). Statistics must remain
   valid (pass the check).
6. `experiments/methods.py` - register:
   - `trustfedgnn-r2`: model trustfedgnn (FSER), aggregator secagg_lagrangian,
     fair_mode lagrangian, dp_enabled False for the FTGD path (FTGD removed).
   - `trustfedgnn-r2-nofser`: same with model gat.
   - `r2-clipmean`: secagg_lagrangian with r2_lambda_step = 0 (isolates the
     Lagrangian effect).
   - `fedfdp`: see Section 5.
7. Tests (`tests/test_r2.py`), all offline, synthetic data, fast:
   - aggregation equals clipped mean when no noise, all valid;
   - an invalid submission (norm > C or S > N) is rejected and excluded from
     every sum; K' counts correctly;
   - T3 bound holds numerically for random valid adversarial statistics;
   - K' = 0 round leaves the model, lambda, direction unchanged;
   - dead zone: |g| <= tau/2 gives direction 0;
   - group_stats is invariant to flipping s on any validation node except for
     moving that node between groups (s-blind pass);
   - lambda update is projected (never negative) and the T4 certificate
     inequality holds on a random sequence of Dhat values;
   - the client signed loss reduces |m1 - m0| on a toy model after a few steps
     with d = sign(m1 - m0) and lambda > 0;
   - existing methods' outputs are unchanged (run one fedavg round before and
     after - bit-identical).

## 5. Baselines added for the resubmission

- FedFDP (Ling et al., arXiv:2402.16028v6, Algorithm 2). Note: FedFDP targets
  *client-level performance fairness* (weighted variance of client losses,
  Eq. 2 of the paper), not demographic parity; it is included because reviewer 4
  asked for a joint privacy-fairness baseline, and the paper must say which
  fairness notion it optimises. Implementation (dp_mode = "fedfdp"):
  - Server keeps Fbar^t = sum_i p_i Ftilde_i (p_i = n_i / n), initial 0, and
    broadcasts it with theta^t. Aggregation: plain FedAvg (weights p_i).
  - Local step on the train nodes, per epoch: per-node losses l_j (weighted
    BCE terms); Delta_j = l_j - Fbar^t (detached); per-node factor
    c_j = 1 + lambda_fdp * Delta_j, clamped to >= 0; loss = mean_j c_j * l_j.
    Then clip the whole gradient to dp_clip and add N(0, (sigma * dp_clip)^2)
    with the client's calibrated noise multiplier (reuse _privatise_grads).
    Deviation from the paper (document it): the paper clips each per-sample
    gradient to min(1 + lambda Delta_j, C / ||grad_j||); per-node gradients are
    not independent in a message-passing GNN and are expensive, so we use
    batch-level clipping as in our DP-FedAvg baseline.
  - Loss upload: f_j = min(Cl, max(0, l_j)) on train nodes with the updated
    model; Ftilde_i = mean_j f_j + sigma_l * Cl * N(0, 1), sigma_l = sigma.
    Adaptive bound: Cl^{t+1} = Ftilde_i of the current round, floored at 1e-3;
    Cl^0 = 1.0. The client reports Ftilde_i as meta["loss"].
  - Hyperparameter: lambda_fdp = 0.5 (config field `fedfdp_lambda`).
  - Wiring: the trainer stores Fbar in `self._agg_state["fedfdp_fbar"]` after
    each round (weighted by meta["n"]) and sets `client.fedfdp_fbar` before
    local training. Method key `fedfdp` in experiments/methods.py: model gcn,
    aggregator fedavg, dp_enabled True, dp_mode "fedfdp".
- FedFACT already exists (post-processing variant, see trainer).
- PUFFLE already exists (dp_mode = "puffle").
- TrustFedGNN-v1 as submitted: FSER + FTGD + robust_bfwa (method key to be
  registered as `trustfedgnn-v1`), so the old pipeline is compared honestly.

## 6. Experiments (all n = 10 seeds, Wilcoxon + Holm vs TrustFedGNN-R2)

E1 Main: German, Bail, Credit, Pokec-z; methods FedAvg-GCN, FairFed,
   FedGraph-Fair, FairGNN, FairSIN, PUFFLE, FedFACT, FedFDP, TrustFedGNN-v1,
   R2-clipmean, TrustFedGNN-R2, TrustFedGNN-R2-noFSER. Report AUC, DPD, EOD,
   final lambda, T4 certificate value.
E2 Update attacks: sign-flip, scaling, ALIE, IPM, fairness-poison at
   f/K in {0.1,0.2,0.3,0.4} on Bail and German; R2 vs median, trimmed mean,
   Krum, robust_bfwa (v1). Check measured deviation against T2.
E3 Statistics attack (stat_falsify): R2 vs BFWA; check |Dhat - D^H| against T3.
E4 Privacy: epsilon sweep {1,2,4,8,16} for the statistics channel; T1
   accounting table with one release per round; attribute-inference probe on
   the aggregate update.
E5 Ablations: FSER on/off (all four datasets), lambda step, C, sigma_u > 0.
Dropped: ogbn-products, Elliptic (no genuine sensitive attribute; reviewer 2
point 5). The scalability claim is removed from the paper.

## 7. Addendum (after code review and pilot runs) - supersedes earlier sections where they conflict

- The work becomes a new paper. FSER is dropped, which also removes the
  train/statistics mismatch noted in review (the s-blind pass was a different
  function from the s-aware FSER model). The backbone is GAT, with GCN as a
  variant. Working name: CLAMP (Clipped secure aggregation with a LAgrangian
  Multiplier and Private group statistics). Method keys: `clamp`, `clamp-gcn`,
  `clamp-clipmean`.
- Statistics period P (`r2_stat_period`). Statistics are released, and
  lambda and d updated, only in rounds t with t mod P = 0. T1 composes
  ceil(R/P) releases, so sigma_s = 2 z(eps, ceil(R/P), delta). T4 holds over
  the release rounds. T2 is unchanged (it is per round). Motivation: with
  P = 1, R = 100 and eps = 8, sigma_s is about 12.8. On German (n_cap = 50 per
  client and group) that swamps the gap estimate. On Credit (thousands of
  validation nodes) it does not.
- Pilot (no statistics noise, GAT, seed 0):
  - German: test soft DPD 0.151 (step 0), 0.148 (step 0.5), 0.129 (step 0.1).
    The validation gap of about 0.05 does not track the test gap on this
    1000-node graph.
  - Credit: test DPD 0.060 (step 0), 0.046 (step 0.5), with validation gap
    close to test gap.
  - Report German as a small-sample limitation.
- New attack `stat_falsify_mid` (both claimed rates equal the honest pooled
  rate). This drives the aggregate gap to 0 without overshoot and is the
  adversary's best valid response for hiding unfairness. It is used in E3
  alongside `stat_falsify`.
- FedFDP: one extra accountant step per round for the loss release.
- The local term stays linear: lambda * d * (m1 - m0). A hinge on the local
  gap was considered and rejected. Under non-IID partitions a client's local
  gap can be small while the global gap is large, and a hinge would then
  switch off exactly when the server asks for correction.
