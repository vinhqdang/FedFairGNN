# Baseline reimplementation fidelity & sources to verify

We compare TrustFedGNN against SOTA fairness / federated / robust methods. To be
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
| FaVGNN | Information Fusion'26 | horizontal adaptation of hetero-feature fusion + adversary | faithful adaptation (paper is vertical-FL; we adapt to our horizontal setting) |
| FDP-Fair | arXiv 2603.24392 (2026) | DP-SGD + demographic-parity group-offset post-processing | exact |

Robust aggregators (Krum, Multi-Krum, coordinate median, trimmed mean) and
attacks (label-flip, Gaussian, sign-flip, scaling, IPM, ALIE, fairness-poison)
follow their original papers; formulas cross-checked against the literature.

## Reimplemented from uploaded PDFs (partial/approximate fidelity, honestly noted)

The five methods below occupy an overlapping niche and were only cited in
related work in an earlier draft despite PDFs being uploaded. Each is now a
runnable baseline (`experiments/methods.py`, `competitors2025b` study), but
each paper's mechanism only partially transfers to this codebase's single
global-model, star-topology, Dirichlet-partitioned-single-graph setting —
the table below states exactly what was kept and what was dropped so no
number is over-claimed.

| Baseline | Venue | Paper's actual mechanism | What we reimplement | What's dropped / approximated |
|----------|-------|--------------------------|----------------------|--------------------------------|
| FairGFL | IEEE TPDS 2026 (arXiv 2512.23235) | aggregation weight $\propto 1/(1+O_i)$, $O_i$ = a privacy-sanitised **node/edge overlap ratio** across clients' *separate* graphs | same weight rule, but $O_i$ is proxied by each client's normalised sample-count deviation from the mean (aggregator `fairgfl` in `src/federated/aggregation.py`) | the paper assumes multiple distinct client graphs with measurable inter-client node/edge overlap; our setting is one graph Dirichlet-partitioned into disjoint node sets, so the literal overlap ratio doesn't exist — we substitute a data-imbalance proxy that captures the same "atypical client gets down-weighted" intent, not the literal graph-overlap statistic |
| FedGraph-Fair | Info. Sciences 728:122710 (2026) | **personalised** per-client models mixed via a learned peer-similarity graph, *plus* a minimax/DRO dual $\lambda$ reweighting high-loss clients | only the DRO core: simplex-projected $\lambda$ dual-ascended toward clients whose loss exceeds an adaptive cap, persisted across rounds (aggregator `fedgraphfair`) | the personalised-model + dynamic top-$k$ similarity-graph mixing layer (the paper's other headline contribution, decentralised communication) is not reproduced — we keep one shared global model aggregated by a central server, as the rest of this codebase does |
| PUFFLE | ECAI'24 | DP-SGD + a **momentum feedback controller** auto-tuning $\lambda\in[0,1]$ toward a target disparity $T$, plus a third DP channel sharing noised group-count statistics for group-imbalanced clients | the controller: `_puffle_step` in `src/federated/client.py` replaces the static `fairness_weight` with a per-round auto-tuned $\lambda$ driven by (DP-noised) local demographic-parity gap vs. `puffle_target_dpd`, combined with clip+noise DP-SGD | the third privacy channel (cross-client group-count sharing for clients missing a demographic group) is not implemented — every client here observes both sensitive-attribute groups, so it isn't needed; only one Gaussian noise channel (on the training gradient) is accounted, not three |
| FedFACT | NeurIPS'25 | joint global+local group-fairness-constrained Bayes risk, solved via a Lagrangian saddle point over global $\lambda$ (server-aggregated) and per-client local $\mu_k$ (never aggregated), with a general multiclass cost-matrix calibration | the exact closed-form special case for a **binary** demographic-parity target: a shared global offset (identical to FDP-Fair's) plus a per-client local offset computed only from that client's own validation split, summed at inference (`_fedfact_offsets` in `src/federated/trainer.py`) | the general multiclass confusion-matrix cost matrix and the iterative dual-ascent solver are not reproduced — for the linear/DP special case the paper's own optimum reduces to closed-form mean-matching, so this is an exact reduction for that case, not a heuristic, but it does not generalise to EOP/multiclass as the paper's method does |
| PoPETs'25 | PoPETs 2025(1), paper 20250044 | **FairFed** aggregation made homomorphically computable: replaces $\exp(-\beta|F_i-F_g|)$ with a degree-2 polynomial (FHE-friendly), aggregated under threshold multi-key CKKS, with the CKKS approximation noise analysed as an incidental $(\varepsilon,\delta)$-DP mechanism | the statistical weighting core only: the degree-2-polynomial FairFed weight (aggregator `popets_fairfed`) | the actual contribution — threshold-CKKS secure aggregation and its noise-as-DP analysis — is cryptographic/systems infrastructure with no effect on the cleartext numeric result once "computed in the clear," so it is not reimplemented; we do not add the CKKS-approximation Gaussian noise, so our numbers reflect the fairness-weighting idea without its privacy side-effect |

## Sources we could NOT retrieve automatically

- **Fairness-constrained optimisation attack**, arXiv **2510.12143** (Oct 2025)
  — strongest single-client fairness-poisoning threat; exact objective in PDF
  only (uploaded and read). Our `fairness_poison` attack is a faithful stand-in.
