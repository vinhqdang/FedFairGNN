# Response to Reviewers

**Manuscript ID:** ca5f4c68-82d0-46f5-99ec-5304715a69a6
**Journal:** Discover Artificial Intelligence — Collection: Trustworthy and Responsible Federated Learning

**Revised title:** Unifying Fairness Privacy Robustness and Explainability in Federated Graph Neural Networks for High Stakes Risk Detection
**Previous title:** Trustworthy Federated Graph Neural Networks: Unifying Fairness, Privacy, Robustness, and Explainability for High-Stakes Risk Detection

*(This is a plain-text copy of the formatted response letter, `response_letter.pdf`, for pasting into the submission system.)*

---

Dear Dr. Fisichella and Ms. Baig, and dear Reviewers,

We thank the Editor and both Reviewers for a careful, constructive and unusually useful set of comments. We are grateful that the reviewers found the work technically sound and well motivated, and we have taken the criticisms seriously: **every point raised has been addressed**, and in several cases we went further than asked.

Three of the comments identified places where our text claimed more than our own tables supported, and we want to acknowledge that directly rather than defensively. Reviewer 1 was right that the "only aggregator that stays strong under all three attacks" claim is contradicted by our own Table 9, right that one baseline was counted but never reported, and right that five numbers disagreed between text and tables. We have corrected all of them, and we also re-audited *every* numerical claim in the paper against the tables it cites, which surfaced four further inconsistencies the reviewers did not raise; those are listed at the end of this letter in the interest of a clean record.

Page numbers refer to the clean revised manuscript. **Nothing in the revision required new experiments**: as Reviewer 1 anticipated, all changes concern wording, scope and consistency. Two figures/tables were re-rendered from the *existing* logged results (no re-training) where the presentation itself was at fault; we flag both explicitly below.

---

## Summary of changes

| Raised by | Issue | Where fixed |
|---|---|---|
| Editor | Title contains punctuation | New title (title page) |
| Editor | Declarations need separate subheadings | Declarations, p. 37 |
| Editor | Data availability statement | Declarations, p. 37 |
| Editor / R1 | Overstated claims | §5.8 p. 26, §5.2 p. 19, §6 p. 29 |
| Editor / R1 | Text/table consistency | §5.2, §5.3, §5.5, §5.8 |
| Editor / R1 | Scope of the DP guarantee | Abstract, §1, Table 1 note, §4.2 |
| Editor / R2 | Protected attribute at inference | New paragraph, §6 p. 32 |
| Editor / R2 | Trust-score limitations | §6 p. 29 (moved before the score) |
| R1.1 | `robust_bfwa` exclusivity claim | §5.8 p. 26, §5.3 p. 20 |
| R1.2 | Sixteen vs. fifteen baselines | Throughout; §3.2 p. 15 |
| R1.3 | Uneven baseline coverage | New paragraph, §3.2 p. 17 |
| R1.4 | Single-seed robustness | New paragraph §5.8 p. 26; Limitation (4) |
| R1.5 | Energy column reads 0.00 Wh | Table 11 and Table 10, §5.10 |
| R1.6 | Five text/table mismatches | §5.2, §5.3, §5.5 |
| R1.7 | DP scope in the abstract | Abstract; Table 1 note |
| R2.1 | Inference-time `s` needs a full treatment | New paragraph §6 p. 32; Limitation (12) |
| R2.2 | Trust-score caveat comes too late | §6 p. 29; Limitation (10) |
| *Authors* | Four further self-identified inconsistencies | See final section |

---

## Response to the Editor

### E1. Revised title

> Revise Title: … the title should be read as one concise sentence, without the use of punctuation (full stops, colons, hyphens, question marks etc.).

Done. The previous title used a colon, three commas and a hyphen. The revised title contains **no punctuation of any kind**, reads as a single sentence, and still names all four trust properties and the application domain:

> **Unifying Fairness Privacy Robustness and Explainability in Federated Graph Neural Networks for High Stakes Risk Detection**

We removed the hyphen from "High-Stakes" and kept a short running head ("Trustworthy Federated Graph Neural Networks") for the page footer, as the journal template requires.

### E2. Declarations with individual subheadings

> …please clearly state the following items as individual subheadings: Ethical approval; Consent to participate; Consent to publish.

Done (Declarations, p. 37). The Declarations were previously a single bulleted list. They are now a section in which each item carries its own subheading: Funding; Conflict of interest; **Ethical approval**; **Consent to participate**; **Consent to publish**; Data availability; Materials availability; Code availability; Author contributions. Each is addressed substantively rather than with a bare "not applicable" — for example:

> **Ethical approval.** Not applicable. This study involved no human participants and no animal subjects. It uses only publicly released, de-identified benchmark datasets (German Credit, Credit Defaulter, Bail/Recidivism, Pokec, Elliptic, and ogbn-products), each obtained from its published distribution and used in accordance with its licence and intended research purpose. No ethics-committee approval was therefore required.

### E3. Data availability statement

Done (Declarations, p. 37). The manuscript now contains an explicit **Data availability** subheading naming every dataset and its public source with URLs, and stating that no new data were generated. We will select "yes" in the submission system and paste this statement verbatim so the two match exactly.

We also strengthened the **Code availability** statement. The previous wording ("available from the corresponding author upon request") sat awkwardly beside the paper's reproducibility claims, so the repository is now cited directly: https://github.com/vinhqdang/FedFairGNN

### E4. Overstated claims, text/table consistency, DP scope

These coincide with Reviewer 1's comments 1, 6 and 7, addressed in full below. In summary:

- **Overstated claims.** The `robust_bfwa` exclusivity claim is withdrawn and replaced with an accurate two-aggregator statement (§5.8); the claim that all non-private baselines significantly beat us on German AUC is corrected to name the exception (§5.2); the "lowest DPD/EOD of any method" claim for PUFFLE is corrected to a tie (§5.2); and the Elliptic result is restated to include the fact that EOD moves the *wrong* way (§5.2).
- **Text/table consistency.** We re-audited every numeric claim against its cited table. Nine discrepancies were found and fixed — the five Reviewer 1 identified plus four more.
- **DP scope.** The abstract now states the scope limitation in its own sentence rather than a parenthesis, and Table 1 carries a footnote so our ✓ in the DP column cannot be read as equivalent to DP-FedAvg's record-level guarantee.

### E5. Protected attributes at inference; trust-score limitations

Both done and expanded well beyond a sentence, since Reviewer 2 asked for the same two things — see R2.1 and R2.2 below.

---

## Response to Reviewer 1

### R1.1 Softening the `robust_bfwa` claim

> Section 5.8. The statement that robust-bfwa is the only aggregator that stays strong under all three attacks is stronger than what Table 9 shows. The trimmed-mean aggregator achieves comparable utility and even performs better under the ALIE attack.

The reviewer is correct and we have withdrawn the claim. Trimmed mean retains 0.984/0.956/0.983 AUC across the three attacks against our 0.992/0.953/0.990, so it survives the suite as we do and is indeed marginally better under ALIE. The passage now reads (§5.8, p. 26):

> "… FedAvg and plain BFWA collapse under the Gaussian attack (0.51 and 0.48 AUC) and Krum collapses under ALIE (0.71), whereas `robust_bfwa` (0.992/0.953/0.990) *and the trimmed mean* (0.984/0.956/0.983) both stay strong under all three. We are careful here not to overstate: the table does not support an exclusivity claim for our aggregator, and the trimmed mean in fact retains marginally more utility than ours under ALIE (0.956 vs. 0.953). The accurate statement is that two of the seven aggregators survive the full suite — ours and the trimmed mean — and that among those two, ours is the one that also carries the fairness constraint of Section 4.3. … The honest summary is that `robust_bfwa` achieves a strong overall *balance* … rather than dominating any metric."

Checking this comment also revealed that the same overstatement had propagated to the ablation discussion, where `robust_bfwa` was again called "the only aggregator that stays strong across all three attacks"; that sentence is corrected in the same way (§5.3, p. 20). While re-reading Table 9 we further noticed that our list of cells where competitors beat us was incomplete: trimmed mean attains the lowest EOD under fairness poisoning (0.005) and multi-Krum the lowest EOD under Gaussian corruption (0.017). Both are now named in the text.

### R1.2 Sixteen versus fifteen baselines

> The manuscript refers to sixteen baselines, whereas the main comparison tables appear to contain fifteen.

The reviewer is right: the tables contain fifteen baselines and the count of sixteen was wrong. The discrepancy is **FedFB**, which we described among the compared methods but never ran, because its FairBatch-style local group reweighting coincides in our pipeline with the local soft-demographic-parity objective already carried by FairFed and F²GNN — running it would have duplicated an existing row rather than covering a new axis.

We corrected the count to **fifteen** in all six places where it appeared (abstract, §1 contribution 5, §1 relation-to-preliminary-version, §3.2, §7 conclusion), removed FedFB from the list of *evaluated* baselines in §3.2 and §2, and retained it only in the related-work survey. To prevent recurrence, §3.2 now ties the count directly to the tables (p. 15):

> "The fifteen correspond exactly to the fifteen baseline rows of Tables 4–6, above the rule that separates them from our own three variants; no method is described here that does not appear there. (FedFB, surveyed in Section 2, is *not* among them: its FairBatch-style local group reweighting coincides with the local soft-DPD objective already carried by FairFed and F²GNN in our pipeline, so running it would have duplicated an existing row rather than added an axis.)"

### R1.3 Uneven baseline coverage across datasets

> …the Credit and especially the Elliptic datasets include fewer competing methods. This should be stated explicitly in the experimental setup.

Agreed — this was left to be inferred from dashes in the tables, which is not good enough. The experimental setup now carries a dedicated paragraph (§3.2, p. 17) stating that the full fifteen-baseline comparison runs only on German, Bail and Pokec; that Credit runs against the seven core baselines plus DP-FedAvg; that Elliptic runs against only FedAvg-GAT and FairSIN, the two baselines that scale within our CPU budget; and that a "--" always means *not run*, never a failed or omitted result. The consequence is propagated to the Elliptic discussion in §5.2 ("not a broad competitive ranking"), to the introduction, and to a new limitation item (4b).

### R1.4 Single-seed robustness experiments

> Since robustness is one of the main contributions, the conclusions are currently based on single-seed experiments… If this is not feasible, the claims should be slightly moderated.

We have taken the second option and moderated the claims, and we want to be transparent about why: a multi-seed replication of the 7×3 aggregator–attack grid is a substantial compute commitment we could not complete within the revision window without compromising the rest of the response, and we would rather state the limit of the present evidence precisely than add two seeds and still be unable to support a ranking.

A new paragraph (§5.8, p. 26) separates the order-of-magnitude findings we rely on from the small differences we decline to interpret:

> "**How far this single-seed evidence reaches.** … Differences of a few thousandths in AUC or EOD between the surviving aggregators — ours, the trimmed mean, multi-Krum — are well within the range seed noise alone produces elsewhere in this paper (TrustFedGNN's own Bail DPD has a standard deviation of 0.013 over five seeds), so we make no ranking claim among them, and no conclusion in this paper rests on such a gap. What a single seed *does* support, because these effects are order-of-magnitude rather than marginal, are the two qualitative findings we actually rely on: that several aggregators *collapse* outright under a given attack (FedAvg and plain BFWA to ≈0.5 AUC under Gaussian noise, Krum to 0.71 under ALIE), and that plain BFWA's fairness is captured by the fairness-poisoning attack — an EOD of 0.126, more than five times the 0.024 of the screened variant…"

The limitations section was rewritten to match, and now names the robustness study as the place where statistical power runs out and its multi-seed replication as "the most valuable single addition to this evaluation" (§7, Limitation (4), p. 34).

### R1.5 The energy column

> The energy column in Table 11 reports 0.00 Wh for all methods… Either report the values with higher precision or clarify that the measurements fall below the resolution of the measurement tool.

Investigating this showed the problem was worse than a precision issue, and we thank the reviewer for prompting the check. Our energy proxy is wall-clock × an assumed CPU draw; wall-clock was **never logged** for those canonical small-graph runs, so the proxy silently received zero and the column printed "0.00" for every method. It was not a measurement below resolution — it was not a measurement at all.

We therefore made two changes rather than adjusting precision:

1. **The energy column is removed from Table 11**, with the caption explaining why: "Energy is *not* reported here: wall-clock was not logged for these canonical small-graph runs, so the proxy of Section 4.6 has no input for them. We report it instead where the measurement exists, on the large-scale study (Table 10), rather than print an uninformative zero."
2. **Energy is now reported where the measurement does exist.** The large-scale ogbn-products study logged per-run wall-clock, so Table 10 gains an **Energy (Wh)** column derived from it: FedAvg-GAT 26.2, FairSIN 24.3, DP-FedAvg 22.6, FaVGNN 50.4, TrustFedGNN 66.2 Wh. §5.10 reads the comparison out — roughly 40 Wh, about 16 gCO₂e, as the one-off cost of training a fair, private and robust model on a 2.4M-node graph — while labelling it a proxy on fixed hardware, not a metered measurement.

No experiment was re-run: the wall-clock values were already in Table 10 and the energy column is an arithmetic transformation of them. We also patched the two table-generating scripts so an unmeasured run emits "--" instead of a spurious zero.

### R1.6 Text/table inconsistencies

All five were real. Each is corrected:

**(a) FedFACT DPD on Bail.** The text said 0.015; Table 5 says 0.021. The table is correct and the text now reads 0.021 (§5.2, p. 19). The surrounding claim — that FedFACT achieves the lowest DPD among non-private methods on Bail — remains true at the corrected value.

**(b) FairFed p-value on German.** The text quoted p=0.006 while Table 3 shows 0.01. The table rounds p-values to two decimals; the text now quotes the rounded value p=0.01, and the caption of Table 3 states the rounding convention so text and table cannot drift again.

**(c) The AUC claim does not hold for FairGNN.** Correct, and this was an overgeneralisation on our part. The sentence now names both the rule and the exception (§5.2, p. 19):

> "… on German its AUC is significantly below *most* non-private baselines — FedAvg-GAT, FairSIN, F²GNN (p=0.02) and FairFed (p=0.01) — the cost of DP plus the fairness/robustness constraints. The one non-private baseline it is *not* significantly below is FairGNN (p=0.16), whose adversarial objective is itself unstable on a graph this small (AUC 0.668±0.084); we state this exception rather than generalise over it."

**(d) The attribute-inference baseline: 0.50 versus 0.53.** This was a genuine defect in the figure, not merely inconsistent phrasing, and we are grateful it was caught. Our attack samples targets *in equal numbers from each true group* (2,000 per group), so an adversary ignoring the release cannot beat 0.500 on that target set — the baseline the table reports. The figure, however, drew a line labelled "chance (base rate): 0.53", which is the majority-group *prevalence* in Bail, the correct chance level only for an *unbalanced* target set. The label was wrong.

We re-rendered Figure 5 from the already-logged attack results (no re-training; the seven (ε, accuracy) pairs are exactly those in Table 8, so figure and table now agree by construction). The corrected figure draws the chance line at 0.500 and labels the 0.53 prevalence line for what it is. The text (§5.5, p. 24) and both captions explain the distinction, and the plotting script was patched so the mislabelling cannot return.

We want to be explicit that we resolved this in the direction that makes our own claim *harder* to support, not easier. Reporting against 0.53 would have been superficially flattering — every measured accuracy (0.502–0.513) lies *below* it — but it would have compared a balanced-target accuracy against an unbalanced-target baseline, and an attack that appears to perform below chance signals a misspecified evaluation rather than a privacy guarantee. The balanced 0.500 requires the mechanism to beat a coin flip rather than a class-prior guess, which is the test we should be held to.

Re-examining the comparison this way also let us make a stronger and more defensible statement than before. Each row of Table 8 is a Monte-Carlo estimate over 4,000 target draws, so its standard error is √(0.25/4000) ≈ 0.008; the largest excess over chance in the entire sweep is 0.013, or 1.6 standard errors. We therefore added:

> "Measured against that harder reference, the residual leakage is not merely small but statistically undetectable at our sample size … *At no privacy budget is the attack's accuracy significantly different from 0.500* (two-sided p ≥ 0.10 throughout). We therefore claim that FTGD reduces the differencing adversary to chance, not merely close to it — while noting that this bounds leakage only at the resolution 4,000 trials can resolve, so an advantage below roughly one percentage point would not be visible to this test."

**(e) German DPD in the ablation versus the main table.** The ablation reports 0.045 and Table 5 reports 0.040±0.035. The reason is that the ablation is a *separate set of runs* with its own matched seed set — so all four configurations are compared on identical seeds — whereas the main tables aggregate the full seed sets (10 on German, 5 on Bail). We had not stated this. The ablation caption now explains it and shows that all four affected cells differ by less than one standard deviation of the multi-seed mean (Table 7, p. 23).

### R1.7 Scope of the DP guarantee in the abstract

> …the abstract and the description of the differential privacy guarantee should make it clearer that the guarantee applies to the released fairness statistics, rather than the entire model update.

Agreed. The scope was previously signalled in the abstract only by a parenthetical, which is too easy to skim past for a limitation this important. It is now a sentence of its own:

> "We emphasise at the outset that this (ε,δ) guarantee covers the *released fairness statistic only*, and *not* the full transmitted model update, which retains sensitive-attribute dependence through FSER and the fairness-gradient masks; the comparison against full-gradient DP-SGD is therefore not like-for-like, and we make the exact scope precise in Section 4.2."

Beyond the abstract, the abstract's closing claim and the conclusion now say "*statistic-level* sensitive-attribute DP". Most importantly, **Table 1** previously gave TrustFedGNN a plain ✓ in the DP column directly beside DP-FedAvg, PUFFLE and FDP-Fair, implying guarantees of equal scope; that entry now carries a marker and the caption explains that our entry "is *not* equivalent to the record-level guarantees of DP-FedAvg, PUFFLE, FDP-Fair or FaVGNN … so this column marks the presence of a formal (ε,δ) guarantee, not guarantees of equal scope."

---

## Response to Reviewer 2

### R2.1 Practical implications of requiring the protected attribute at inference

> FSER requires the protected attribute at inference time. This is currently called a "governance nuance" in one line. Given the paper's emphasis on EU AI Act compliance, this deserves a dedicated paragraph in Section 6…

We agree entirely: "governance nuance" understated what is the single design decision in TrustFedGNN with the largest practical and legal footprint. §6 now carries a dedicated paragraph (p. 32) that first makes the technical requirement concrete — because the model attends over a neighbourhood, scoring one applicant requires the protected attribute of the *counterparties* in that applicant's neighbourhood, not only of the applicant — and then works through four consequences:

- **Lawful basis.** Gender, race and ethnicity are special-category data under GDPR Art. 9; processing them in an automated decision requires an Art. 9(2) condition, and in several member states the substantial-public-interest route additionally needs a national-law basis. The EU AI Act anticipates this tension: Art. 10(5) permits processing special-category data *for the purpose of bias detection and correction* in high-risk systems, subject to safeguards including technical limits on re-use, pseudonymisation, and deletion once bias correction is complete. FSER is squarely a bias-correction mechanism and so is the kind of processing that provision contemplates — but an institution must document that reliance rather than assume it, and Art. 10(5) is a permission to process, not a waiver of Art. 9 or of a DPIA.
- **Data-access agreements.** In the cross-silo setting the counterparties may be another institution's customers, so inference-time access to `s` can require inter-institutional agreements that plain FedAvg does not — a governance cost that partly offsets the "no raw data leaves the silo" benefit of federation.
- **Collection risk.** Where `s` is not already held, obtaining it means either new collection (consent, minimisation, retention limits) or *inference* of a protected attribute from proxies, which is itself a discriminatory-profiling risk and which we do not recommend.
- **Purpose limitation.** `s` must reach the attention mechanism without reaching the classifier as a feature: our implementation keeps it out of X, but that separation is an engineering invariant a deployer must verify and an auditor must test, not a property the architecture enforces by itself.

The paragraph closes by telling a reader who *cannot* meet these obligations what to do: FSER is then the wrong component, and the ablation (Table 7) shows FTGD and BFWA operating without it at a quantified fairness cost. We also added limitation (12) recording that "until this is done, TrustFedGNN is only deployable where such processing is lawful and governed", and a future-work direction on removing the dependence altogether — by distilling FSER into an `s`-free student model, or by driving the edge correction from the DP-noised statistics instead of a hard group indicator.

### R2.2 Making the trust-score limitation more prominent

> …the paper mentions the composite trust score is not decisive evidence, but this appears after the score is already presented.

A fair criticism: a caveat that arrives after the number has been read has already failed. We have inverted the order. The subsection is retitled "**Composite trust score: read this caveat first**" and opens with the limitation, before the score is defined (§6, p. 29):

> "Because a single headline number invites exactly the over-reading we want to avoid, we state its limits *before* presenting it rather than after. The composite score below is a *communication tool* … It is not a certification of trustworthiness, not a substitute for the per-axis evidence, and not decisive evidence of superiority. Its ranking depends on choices we make — the reference constants, the unit weights, and the power p — and a reader who chooses different constants may obtain a different ordering. Auditors, deployers and reviewers should therefore treat the per-axis results (Tables 4–11) as the primary empirical findings, and the composite as a summary of them."

The corresponding limitation item was strengthened and now adds a point we had not made anywhere: the composite covers only the axes computable from a given run, so the three-axis version in Table 12 omits calibration and robustness entirely, and "a high score is a statement about the axes included, not about the system as a whole" (§7, Limitation (10), p. 35).

---

## Further corrections made on our own initiative

Prompted by Reviewer 1's comment 6, we re-audited every numerical claim in the paper against the table it cites. Four further discrepancies surfaced, none raised by the reviewers:

1. **A limitation contradicted the paper.** Limitation (4) still read "we … do not run significance tests", true of an earlier draft but false once the paired Wilcoxon tests of Table 3 were added. The item is rewritten to say precisely where the paper *is* statistically supported (German, Bail, Pokec at ≥5 seeds) and where it is not (Credit/Elliptic at 2–3 seeds; robustness, large-scale and calibration at one seed).
2. **An overstated claim about PUFFLE.** §5.2 said PUFFLE "reaches the lowest DPD/EOD of *any* method on Bail (0.001)". Its DPD of 0.001 is *tied* with FDP-Fair and DP-FedAvg, and its EOD of 0.006 is beaten by TrustFedGNN-Robust (0.005). Corrected to a tie among the full-gradient-DP methods.
3. **An imprecise margin.** The AUC margin over the privacy-preserving methods on Bail was quoted as "+0.40"; the three actual margins are +0.38, +0.40 and +0.44. Now reported as a range.
4. **A wall-clock ratio outside its stated range.** The TrustFedGNN / FedAvg per-round cost was quoted as "2.4–2.5×"; Table 10 gives 2.45× and 2.53×. Corrected to 2.45–2.53×.

Additionally, and in the same spirit as R1.1, we restated the Elliptic result in §5.2. It previously read that TrustFedGNN "attains strong subgroup fairness at detection accuracy comparable to the non-fair backbone", which quietly omitted that EOD moves the *wrong* way on that dataset (0.020 → 0.075) — a tension our own limitations section already acknowledged. The passage now gives DPD, AUC *and* the adverse EOD result together.

---

## Closing

We believe the manuscript is substantially more accurate for these comments. The technical content is unchanged — no result has been altered, and no new experiment was needed — but several claims are now bounded by what our evidence actually supports, the text and tables agree everywhere we could check them, the scope of the privacy guarantee is unambiguous, and the two governance questions the Editor and Reviewer 2 raised are treated at the length they deserve.

We thank the Reviewers once more for the care they invested, and the Editor for the opportunity to revise.

Sincerely,
Minh Ngoc Dinh, on behalf of all authors
*Corresponding author*
