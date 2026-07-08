# Baseline reimplementation fidelity & sources to verify

We compare FedFairGNN against SOTA fairness / federated / robust methods. To be
transparent about reproduction fidelity (and to flag sources we could not
retrieve automatically), this document records each baseline's status.

## Faithfully reimplemented (from published algorithm descriptions)

| Baseline | Venue | What we implement | Fidelity |
|----------|-------|-------------------|----------|
| FedAvg-GCN / GAT | AISTATS'17 | standard backbones + FedAvg | exact |
| FairGNN | WSDM'21 | adversarial debiasing (encoder + sensitive adversary, min-max) | faithful core; sensitive-estimator for missing S omitted (all S observed here) |
| FairSIN | AAAI'24 | FairSIN-F: heterogeneous-neighbour feature augmentation + MLP estimator | faithful to the -F variant; per-layer discriminator variant not used |
| FairFed | AAAI'23 | fairness-gap aggregation-weight update rule (exact formula) | exact aggregation rule |
| q-FedAvg | ICLR'20 | q-reweighted aggregation by client loss | exact |
| FedFB | — | FairBatch-style local group reweighting under FedAvg | faithful approximation |
| F$^2$GNN | arXiv'23 | softmax fairness-weighted + group-balance aggregation | faithful to the aggregation design |
| DP-FedAvg | ICLR'18 | full-gradient clipping + Gaussian noise (RDP-accounted) | exact; serves as the FTGD contrast |

Robust aggregators (Krum, Multi-Krum, coordinate median, trimmed mean) and
attacks (label-flip, Gaussian, sign-flip, scaling, IPM, ALIE, fairness-poison)
follow their original papers; formulas cross-checked against the literature.

## Sources we could NOT retrieve automatically (please upload PDFs if a
## head-to-head is desired)

These recent methods occupy an overlapping niche; their PDFs were paywalled or
returned non-extractable content during automated search. We cite them and, for
the closest ones, reimplement the described mechanism, but could not verify
exact numbers/hyperparameters line-by-line:

- **FairGFL** (federated + GNN + fairness + DP), arXiv **2512.23235** (Dec 2025)
  — the single most-overlapping recent method; results tables not extractable.
- **FedGraph-Fair** (personalised + fair federated GNN via DRO), *Information
  Sciences* 2025 — paywalled.
- **FedFACT** (provable global+local group-fair FL), NeurIPS'25 — OpenReview
  `6lCY5bLW8E`; check for code release.
- **PUFFLE** (privacy+utility+fairness FL), ECML-PKDD'24 — code exists
  (`github.com/lucacorbucci/PUFFLE`); we implement the DP+DP-loss idea as
  `dp-fedavg`+local fairness rather than porting the repo.
- **Fairness-constrained optimisation attack**, arXiv **2510.12143** (Oct 2025)
  — strongest single-client fairness-poisoning threat; exact objective in PDF
  only. Our `fairness_poison` attack is a faithful stand-in.

If you upload any of these PDFs, we can (a) verify our reimplementation against
their reported numbers and (b) add an exact head-to-head row.
