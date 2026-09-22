# Response to Reviewers — TrustFedGNN

We thank the reviewers for a reading that was unusually specific. Three criticisms in particular
changed the paper rather than its wording, and we say so plainly below. Where a criticism was
correct and we could not resolve it, we say that too.

Section and table numbers refer to the revised manuscript.

---

## 1. The central criticism, and what we did about it

> **R2(b)2 · "The claimed vulnerability of fairness-aware aggregators is not tested against the
> actual cited methods."**
> **R3(c)1 · "Several named components are empirically unsupported."**

This was correct, and it was the most damaging thing in the reviews because it was true of the
paper's central claim. The previous version demonstrated the metadata attack on **two** aggregation
rules, one of which we had designed ourselves, and reported a capture figure of $86.2\%$ that was a
**single seed's value** rather than a mean.

We have replaced that evidence entirely. Section 7.4 now reports an end-to-end campaign against
**six published aggregation rules** — FairFed (AAAI'23), q-FedAvg (ICLR'20), F²GNN (WWW'23),
FedGraph-Fair (InfoSci'26), PoPETs-FairFed (PoPETs'25) and BFWA (IndabaX'26) — with two
metadata-blind rules, CGSV and FLTrust, as negative controls. Thirty seeds, 720 runs.

The design answers the obvious objection about attribution: the adversary transmits the **same
poisoned update in both arms**, and the arms differ only in the accompanying report. Any difference
is therefore attributable to the declared fields and not to the attack.

| | |
|---|---|
| All six metadata-reading rules move under falsification | **6/6 survive Holm–Bonferroni** |
| Aggregated over the six | lying raised the adversary's weight in **141 of 144** non-tied seed pairs, lowered it in 3 (sign test $p = 4.5\times10^{-38}$) |
| CGSV and FLTrust under the identical falsification | **bit-exact**, $\max\lVert\Delta w\rVert_\infty = 0.0000$ on every seed |

The outcome also splits three ways rather than two, which we had not anticipated. PoPETs-FairFed is
*statistically* steerable ($p = 6.3\times10^{-5}$) but moves so little that the adversary ends at
$0.83\times$ an equal share — **less than under uniform weighting**. Its degree-two polynomial,
adopted so the rule can run under homomorphic encryption, retains little of the steering range of
the exponential it replaces. We report this as inertness rather than robustness, because nothing in
the rule resists the attack; there is simply almost nothing for the attack to move.

---

## 2. The second criticism, and a result we did not have before

> **R1(a)3 · "The privacy result is too weak for the paper's 'trustworthy' framing."**
> **R1(a)1 · "Theorem 3 establishes orthogonality, not fairness improvement."**

Both are correct as stated, and we have stopped claiming otherwise (Sections 4.1, 5.2). But the
first prompted a question we had not asked: if privatising the reported statistic is the natural
repair, does it work?

It does not, and Section 5.2 now proves why. Lemma 3 shows the released statistic is folded normal;
Theorem 4 shows that once noise dominates, its expectation converges to
$\sqrt{2/\pi}\sum_k w_k\tilde\sigma_k$ — **a constant of the noise, independent of every client's
true disparity**. Testing a fairness budget against that quantity is a test on the noise multiplier
and the smallest group, not on fairness. Corollary 3 gives the $\epsilon$ below which the constraint
carries no information.

We state the regime rather than the bare result, because the bound is **conditional and an
unconditional claim would be wrong**: the obstruction bites in small cross-silo deployments over
many rounds — the regime FairFed, PUFFLE and F²GNN operate in — and does not bite at all in large
cross-device settings with a single release, where $\epsilon \approx 1.5$ suffices.

Section 7.5 tests the prediction rather than illustrating it. We hold the data to the **exact**
folded-normal expectation rather than to the asymptotic form, because a measurement compared against
an asymptote can only agree inside the regime the asymptote assumes. Over $200$ paired releases at
each of seven privacy levels, the ratio of observed to predicted aggregate has a bootstrap $95\%$
interval containing $1$ at **every** $\epsilon$ tested. The asymptotic form drifts and fails by
$\epsilon = 32$, which locates the boundary of the approximation instead of merely caveating it.

---

## 3. Missing experiments, now run

> **R1(b)1 · "The principal Pokec-z utility gain is confounded by architecture and training scaffold."**
> **R3(c)2 · "The critical same-backbone ablation is missing on the headline dataset."**

Correct, and the previous version said in Section 7.3 that this experiment "requires the GPU
campaign to be repeated and we have not done it." That sentence was wrong — the experiment had been
run — and we have removed it rather than leave a false statement in the paper.

Table 7 now reports the backbone-fixed control on **both** datasets. The verdicts agree: holding the
architecture constant, the aggregation rule contributes $-0.0031$ AUC on German ($p = 0.8457$) and
$+0.0015$ on Pokec-z ($p = 0.6250$). **We therefore no longer attribute the Pokec-z utility margin
to the aggregation rule.**

> **R2(b)5 · "The server holdout assumption is not stress-tested."**
> **R3(c)4 · "No holdout-quality or hyperparameter sensitivity study is provided."**

Both now exist (Section 7.8), and both returned **null**, with criteria fixed in advance.

Sweeping $\alpha \in \{0,\dots,1.0\}$ at ten seeds per setting gives Spearman $-0.109$ against
disparity ($p = 0.20$) and $-0.060$ against EOD ($p = 0.32$). The point estimates fall across the
grid but seed variance absorbs the trend. Under the pre-registered criterion this removes our right
to call $\alpha$ a control, and the manuscript no longer does: it is reported as **a default the
data does not justify**.

Sweeping the holdout over $\{50,100,200,400\}$ requested nodes moves no metric by more than two
seed-level standard deviations, and the response is not monotone. We also report a limit we did not
anticipate: German caps the holdout at $125$ nodes, so requests of $200$ and $400$ resolve to the
same configuration and **the claim is scoped to $[50,125]$**. Representativeness, as distinct from
size, remains untested and is stated as such in Limitations.

---

## 4. Positioning against FLTrust

> **R1(a)4 · "The core aggregation novelty is incremental relative to FLTrust."**

The overlap is real and we had understated it. FLTrust is metadata-immune, we are not the first to
be, and Section 2.5 now says so directly.

What the revised positioning adds is the shape of the table rather than a defence. Every rule in it
that is metadata-independent — CGSV, FLTrust, Krum, trimmed mean, coordinate median — is a rule with
**no fairness term at all**. They are immune because they have no fairness signal to be steered by,
not because they obtained one safely. Every entry that does target group fairness obtains its signal
by asking, and pays for it. The remaining cell is a rule that is fairness-aware **and** reads no
declared field.

We also state the price in the same place. On clean data FLTrust attains higher AUC than we do
($0.8067$ against $0.7899$ on Pokec-z). It does so with aggregation weights that oscillate at
$\Omega_w = 0.7267$ against our $0.0976$ — a factor of $7.4$ — which we report because it is the
cost side of the same comparison.

---

## 5. Corrections we owe the reviewers

**Three citations in the reviews are mis-attributed, and we verified each against the publisher
rather than assert it.**

1. **DRFA** is Deng, Kamani and Mahdavi, **NeurIPS 2020**, not the venue given. We have added it as a
   distributionally-robust comparator; its dual-weighting structure is close to the FedGraph-Fair
   branch we already implement.
2. **FedMLB** is Kim, Kim and Han, **ICML 2022** (PMLR v162). It is a **client-side knowledge-
   distillation regulariser with no aggregation rule**, so it cannot be evaluated on the axis this
   paper studies; it belongs with the metadata-blind controls.
3. **"FedFairGNN [EAAMO'25]"** is Dang and Nguyen, **IndabaX Nigeria 2026**, PMLR v319, pp. 74–86.
   It is **our own prior work**, and we should have said so in the submitted version. Section 2.5 now
   carries an explicit statement of the relationship: that paper introduced FSER, FTGD and BFWA, and
   BFWA sets its weights from client-declared performance and disparity. That design decision is
   what the present paper revisits, which is why BFWA appears in the attack table on the same footing
   as the other five published rules rather than as a straw man.

---

## 6. What we did not resolve

> **R1(a)5 · "No convergence or optimization guarantee is provided for Eqs. (6)–(9)."**

Correct and unaddressed. We give no convergence guarantee for the alignment-and-rectifier rule, and
the manuscript states this as a limitation rather than deferring it to future work in a way that
implies it is nearly done. The empirical weight-oscillation measurement is a description of
behaviour, not a proof of it.

> **R1(a)1 (second part) · the orthogonality claim**

Theorem 1 establishes orthogonality and nothing more. Section 4.1.2 previously invited a stronger
reading; it now states that the projection is a geometric property keeping the fairness direction
out of the transmitted update, **not a fairness mechanism**, and the isolating ablation finds it
indistinguishable from the full method on disparity ($p = 0.8457$).

> **R1(a)3 (book-keeping)** · the release frequency

The reviewer is right that Algorithm 1 and Table 8 did not agree. The statistic is *computed* once
per local epoch, because the client consumes it in its own objective, but only the final value is
*transmitted*, once per round. We compose all $R \times E$ draws rather than the $R$ transmissions,
which over-charges the budget relative to what an observer of the protocol sees. Section 7.3 now
says this, and the reported $\epsilon$ should be read as an upper bound on the realised one.

---

## 7. Components we removed from the contribution list

Three things the previous version presented as contributions do not survive their own ablation, and
we have demoted rather than defended them.

| Component | Evidence | Status now |
|---|---|---|
| FSER | $\Delta\mathrm{AUC} = -0.0017$ ($p = 0.2324$); $\Delta\dpd = +0.0090$ ($p = 0.7344$) | described with its scope conditions; **no effect claimed** |
| Orthogonal projection | isolating arm indistinguishable on disparity ($p = 0.8457$) | geometric property, **not a fairness mechanism** |
| $\alpha$ as a fairness control | three pre-registered criteria, none met | **a default the data does not justify** |

Section 4.1 now states which client-side stages the evidence supports **before** describing any of
them, so that a reader does not meet a mechanism as a contribution and find it refuted thirty pages
later.
