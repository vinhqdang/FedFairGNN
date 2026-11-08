# CLAMP-RT design: a falsification-robust statistics channel

Status: design spec for the redesign that follows the Information Sciences
rejection (INS-D-26-14683). Internal name **CLAMP-RT** (robust, trimmed).
Everything not mentioned here (clipped update aggregation, single Lagrange
multiplier, sign direction with dead zone, feasible output rule with warm-up)
is unchanged from `docs/trustfedgnn_r2_design.md`.

## 1. Why the old channel cannot be fixed

CLAMP released four sums (S0, S1, N0, N1). The server could only range-check
each summand, so a liar inside the range moves the pooled rates by up to
f n_cap / (N^H_a + f n_cap). That bound is vacuous in 10 of 12 settings and
one liar raised Credit DPD from 0.011 to 0.064. A sum of unverifiable data
cannot do better than its weight share. Robustness needs an order statistic,
and an order statistic cannot be computed from sums. CLAMP-RT therefore drops
the "sums only" restriction for the statistics channel, which is a single
scalar per client, and computes a trimmed mean of per-client gaps in
two-server MPC. The update channel stays a checked sum.

## 2. Architecture

Two non-colluding aggregation servers A and B (the Prio / ELSA model). Each
client additively secret-shares to A and B, over a ring Z_{2^64} with
fixed-point encoding (16 fractional bits):

* its clipped update u_k (norm ≤ C), and
* one scalar d_k ∈ [-1, 1], its local signed demographic-parity gap on
  validation nodes (definition in Section 3).

The servers jointly:

1. check ||u_k||^2 ≤ C^2 (Beaver-triple dot product, one comparison) and
   -1 ≤ d_k ≤ 1 (two comparisons); a client that fails either check is
   dropped from both channels this round;
2. sum the accepted updates (local on shares) and open the sum;
3. sort the accepted d_k with an odd-even merge sort network, drop the b
   smallest and the b largest, and sum the remaining K_acc − 2b values;
4. each server adds its own Gaussian noise ξ_A, ξ_B ~ N(0, σ_d^2) to its
   share of the trimmed sum divided by (K_acc − 2b); the opened value is
   g̃ = TM_b(d) + ξ_A + ξ_B.

The opened values are the update sum and g̃. Nothing else is revealed:
not d_k, not which clients were trimmed, not group counts. If one server is
honest its noise alone gives the DP guarantee, so the analysis uses σ_d
(conservative; the total variance is 2σ_d^2).

When K_acc − 2b < 1 the round releases nothing (λ and direction stay).

In the experiments the MPC is simulated in plaintext (same arithmetic,
float64, Gaussian noise); costs are measured separately with a real MPC
implementation of steps 1–4 (Section 7).

## 3. Client statistic

Client k holds validation nodes V_k with groups V_{k,0}, V_{k,1}. The forward
pass is s-blind (as before). With soft predictions p_v ∈ [0, 1]:

    m_{k,a} = ( Σ_{v ∈ V_{k,a}} p_v ) / max(|V_{k,a}|, n0_k)
    d_k     = clip(m_{k,1} − m_{k,0}, −1, 1)

The floor n0_k = ceil(c · |V_k|) with a public fraction c (default 0.1) bounds
the attribute-level sensitivity (Section 5). |V_k| is the client's validation
size, which does not depend on s (the same publicity argument as the old
public cap). A group smaller than n0_k is shrunk toward 0; this is a bias that
the experiments report (number of client-groups with |V_{k,a}| < n0_k).

The server-side gap estimate is g̃; the multiplier update and direction are
unchanged: Dhat = |g̃|, λ ← max(0, λ + η (Dhat − τ)), direction = sign(g̃)
with dead zone τ/2.

## 4. Trimming level

b is public. Default b = max(1, floor(0.2 K)) (K = 5 → 1, K = 10 → 2). The
breakdown point is b/K. Ablation: b ∈ {0, 1, 2} on K = 5 and {1, 2, 3} on
K = 10; b = 0 is the plain mean of per-client gaps (no robustness).

## 5. Guarantees (to be proved in the manuscript)

**T1 (attribute-level DP of the statistics channel).** Neighbouring datasets
differ in the sensitive attribute of one validation node of one honest client
k. Then |Δ m_{k,a}| ≤ 1/n0_k for each group (proof: for S ≤ N, |S/N −
(S+p)/(N+1)| ≤ 1/(N+1); the max(·, n0) floor only shrinks the change), so
|Δ d_k| ≤ 2/n0_k. Sorting is non-expansive in ℓ1, so the sum of any fixed set
of order statistics changes by at most |Δ d_k|, and

    Δ_TM = 2 / ( n0_min · (K − 2b) ),   n0_min = min_k n0_k  (public).

Rejected clients: their d_k is replaced by 0 inside the MPC, so the sort
always has K entries and Δ_TM = 2/(n0_min (K − 2b)) holds whatever is
rejected (honest clients never fail the checks).

σ_d = Δ_TM · z(ε, δ, R_rel) with R_rel = ceil(T/P) releases, via the existing
RDP calibration (`calibrate_noise_multiplier`). Post-processing covers λ,
direction and the output rule.

**T2 (update influence).** Unchanged: ≤ 2fC/(K − f) per round.

**T3 (falsified statistics, non-vacuous).** Let f ≤ b clients be Byzantine and
report arbitrary values in [−1, 1] (outside values fail the range check and
become 0). Let h_(1) ≤ … ≤ h_(K−f) be the honest d_k sorted. Then every kept
value lies in [h_(b−f+1), h_(K−b)], hence

    h_(b−f+1) ≤ TM_b(d) ≤ h_(K−b).

Consequences: (i) liars cannot move the estimate outside the honest range;
(ii) if every honest client has gap above τ + r, the estimate is above τ + r
whatever the liars send, so the controller cannot be switched off;
(iii) |TM_b(d) − TM_b(d^H-only analogue)| ≤ h_(K−b) − h_(b−f+1), the honest
inner spread, which the experiments report per round. For f > b no bound
(breakdown).

**T4 (returned model, population level).** With probability ≥ 1 − β over the
release noise, every release satisfies |g̃_t − TM_b(d_t)| ≤ r_R = σ_d
sqrt(2 ln(2R/β)) (union bound over R releases; this replaces 8R in the old
Prop 2 because there is one noisy scalar, not four). Add a validation-to-
population term: for a fixed model, each client's validation group rates
deviate from its population group rates by at most sqrt(ln(4K R/β') /
(2 n_{k,a})) by Hoeffding with a union over clients, groups and the R
candidate models (the returned model is one of the R release-round iterates,
fixed before the validation draw only if validation nodes are not used in
training — they are not). TM_b is 1-Lipschitz in ℓ∞, so the population
trimmed gap of the returned model is at most τ + r_R + 2 max_{k,a} sqrt(...)
(ignoring the n0 shrinkage; state it with the shrunk rates). The fallback
(no feasible round) is reported as a flag, and the statement is conditional
on a feasible round existing, which is observable.

Prop 5 of the old paper (accounting identity) is dropped.

## 6. Code changes

* `src/config.py`: `r2_stat_channel: str = "sums"` (sums | trimmed);
  `r2_trim_b: int = -1` (−1 → default max(1, floor(0.2 K))); `r2_floor_frac:
  float = 0.1` (c).
* `src/federated/client.py`: `local_gap()` returns d_k as in Section 3,
  using the same s-blind forward pass as `group_stats()`; also return the two
  group sizes for diagnostics only (never sent to the server).
* `src/federated/aggregation.py`: `secagg_trimmed(updates, gaps, state, *, C,
  tau, eta_lambda, b, sigma_d, sigma_u, generator, release, direction_rule)`.
  Norm check on updates; range check on gaps (−1 − tol ≤ d ≤ 1 + tol); a
  failing client is dropped from the update sum and its gap replaced by 0;
  sort, trim b each side, mean of the K − 2b kept, add N(0, σ_d^2); then
  identical λ / direction logic. info logs: dhat, gap, lambda, direction,
  n_rejected, tm_clean (pre-noise TM), honest_lo, honest_hi (h_(b−f+1),
  h_(K−b) computed by the trainer from honest gaps, diagnostics only).
* `src/federated/trainer.py`: compute n0_min, σ_d = 2/(n0_min (K − 2b)) · z;
  log them in the result json; call `secagg_trimmed` when
  `r2_stat_channel == "trimmed"`; feasible output rule unchanged.
* `src/federated/attacks.py`: `falsify_gaps(gaps, byz_ids, mode, honest)`
  with modes (all omniscient, all within [−1, 1]):
  - `hide`: report −sign(g^H) · 1 (pull estimate toward / past 0);
  - `inflate`: report +sign(g^H) · 1 (force over-correction);
  - `inlier`: report the honest value closest to 0 on the side that hides
    the gap (h_min if g^H > 0) — the strongest attack that stays inside the
    honest range;
  - `zero`: report 0 (claims perfect fairness).
  Attack names: `gap_hide`, `gap_inflate`, `gap_inlier`, `gap_zero`.
  The old `stat_falsify*` attacks map to `gap_hide` for the comparison
  table (both try to shrink the estimate).
* `experiments/methods.py`: `clamp-rt` (trimmed, last), `clamp-rt-f`
  (trimmed, feasible) — main method; `clamp-rt-b0-f` (b = 0 ablation);
  `clamp-rt-gcn-f` backbone variant.
* Tests (`tests/test_clamp_rt.py`): TM sensitivity bound holds on random
  inputs (flip one node, check |Δ TM| ≤ Δ_TM); T3 bound holds for every
  attack mode and f ≤ b on random honest gaps; rejected client's gap is 0;
  b = 0 reduces to the mean; release=False leaves λ unchanged; σ_d formula.

## 7. MPC cost measurement (answers reviewer comment 4)

Implement steps 1–4 in MP-SPDZ (semi-honest 2PC over Z_{2^64},
`semi2k`, and malicious `spdz2k` if it runs in reasonable time) for K ∈ {5,
10, 20} and model sizes d ∈ {the GAT used, 10^5, 10^6}. Report wall time on
one machine (two processes over localhost, plus a simulated 50 ms RTT via
`tc` if permitted), bytes sent per server, rounds. Separate the cost of the
norm checks (dominant, O(K d) multiplications) from the gap channel (O(K
log^2 K) comparisons, expected negligible). Also report the same for the
noise addition (local, free). Cite RoFL for client-side zero-knowledge proofs
as the one-server alternative.

## 8. Experiment plan

Reused unchanged: all baseline E1 runs, E3 baseline attack runs.
New runs (main method `clamp-rt-f`, 10 seeds unless stated):

* E1: 8 datasets × {clamp-rt-f, clamp-rt-gcn-f} × 10 seeds.
* E3: update attacks on Bail and Pokec-z, K = 10, f = 1..4, 3 seeds (f > b
  is beyond the breakdown point for the gap channel only; the update channel
  is unaffected by b).
* E4: gap attacks {hide, inflate, inlier, zero} × f ∈ {1, 2} (K = 5) and
  {1, 2, 3} (K = 10) on Credit, Pokec-z, Pokec-n, Bail; 3 seeds. Compare with
  the old sums channel under `stat_falsify` (existing E4 results).
* E5: ε ∈ {1, 2, 4, 8, ∞} × P ∈ {1, 5, 10} on Credit, Pokec-z, Pokec-n.
* E6: ablations b, c ∈ {0.05, 0.1, 0.2}, channel sums vs trimmed.
* Privacy probe: unchanged scripts, on clamp-rt-f.

Pilot first (before the full campaign): Credit, Pokec-z, Bail, 2 seeds,
ε = 8, P ∈ {1, 10}, b default, honest and `gap_hide` f = 1. Go / no-go: the
honest soft DPD of clamp-rt-f should be within 0.01 of clamp-f, and the
`gap_hide` f = 1 DPD should stay below 0.04 on Credit and Pokec-z.

## Addendum: bias-corrected magnitude (clamp-rt-db-f)

Pilots showed that with a small validation set the release noise sigma_d is as
large as the gap itself (sigma_d about 1.1-1.3 on German, NBA, Facebook). Then
|g~| is dominated by noise, the multiplier lambda grows without bound and the
model collapses. The magnitude fed to the dual step is therefore replaced by

    Dhat = sqrt( max(0, g~^2 - 2 sigma_d^2) )

where 2 sigma_d^2 is the total noise variance of the two servers. This uses
only the released value and public sigma_d, so it is post-processing and has no
privacy cost. The direction still uses sign(g~) with the dead zone. It requires
gap_ema = 1 (no smoothing). Option `r2_debias`; method name `clamp-rt-db-f`.
The uncorrected variant (`clamp-rt-f`) stays as an ablation. Groups e1db, e3db,
e4db, e5db hold the corrected runs. Summaries: `results/clamp_rt/summary/`.
