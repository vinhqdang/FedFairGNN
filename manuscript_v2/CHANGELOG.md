# Changelog — Manuscript v2 (`FedFairGNN/manuscript_v2`)

All notable changes to the manuscript rewriting project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - 2026-09-21

### Added
- **Taxonomy of Baseline Paradigms**: Restructured the flat baseline palette in `sections/04_implementation_results.tex` into five principled research paradigms:
  1. *Paradigm I (Unconstrained Graph FL)*: `FedAvg-GCN` utility reference baseline.
  2. *Paradigm II (Client-Side Fair In-Processing)*: `FairGNN`, `FairSIN` (local topological regularizations, undefended aggregation).
  3. *Paradigm III (Client-Reported Disparity Coordination)*: `FairFed`, `FairGFL`, `FedGraph-Fair`, `BFWA`, `PUFFLE` (metadata honor system, vulnerable to spoofing in Act 1).
  4. *Paradigm IV (Spatial & Geometric Byzantine Filtering)*: `Median`, `Trimmed-Mean`, `Krum`, `Multi-Krum`, `FLAME` (Euclidean distance clustering, prone to Shielding Paradox in Act 2).
  5. *Paradigm V (Server-Anchored Directional Alignment)*: `FLTrust`, `CGSV` (gradient projection against root anchor, prone to trajectory variance in Act 3).
  - Seamlessly bridges baseline classification directly into empirical deductions across Act 1 (§4.2), Act 2 (§4.3), and Act 3 (§4.4).
- **Directory Architecture**: Created modular structure `manuscript_v2/` with `sections/`, `tables/`, `figures/`.
- **Modular TeX Files**: Split monolithic document into `main.tex` and independent section files:
  - `sections/00_preamble.tex`: Centralized packages, math operators, macros.
  - `sections/06_implementation.tex`: Main empirical section structured into 4 beats:
    1. *Experimental Setup and Benchmark Protocol* (Setup, 4 datasets, Master Matrix).
    2. *Vulnerability and Collapse of Client-Reported Fairness Interfaces* (Hồi 1).
    3. *Multi-Tier Adversarial Stress-Testing and Empirical Resilience* (Hồi 2).
    4. *Causal Attribution, Architectural Hygiene, and Auditable Governance* (Hồi 3).
  - Skeletons for `01_introduction.tex`, `02_related_work.tex`, `03_problem_formulation.tex`, `04_methodology.tex`, `05_theoretical_analysis.tex`, `07_limitations.tex`, `08_conclusion.tex`.
- **Ref.bib Audit & Expansion**:
  - Added missing SOTA entries from `docs/01_sota_taxonomy_and_gap_analysis.md`:
    - `nguyen2022flame` (FLAME, USENIX Security 2022).
    - `byitfl2024` (ByITFL, IEEE TIFS 2024).
    - `wen2025fltg` (FLTG, BlockSys 2025).
    - `li2025guardfed` (GuardFed, arXiv 2025).
    - `commey2026fedgraphvasp` (FedGraph-VASP, arXiv 2026).
    - `fung2018mitigating` (FoolsGold, 2018).
  - Added alias keys `zhou2026fairgfl`, `khan2026fedgraphfair`, `f2gnn2023`.
- **Table Suite (`tables/`)**:
  - Implemented float plan from `docs/review/plan_floats_CAU_CHUYEN.md`:
    - `tab_datasets.tex` (T0): Benchmark topologies & Master Config.
    - `tab_metadata_capture.tex` (T1): 10 rules metadata capture & syntactic immunity.
    - `tab_ldp_barrier.tex` (T2): Folded-normal bias & Minimax Le Cam slack.
    - `tab_two_tier.tex` (T4): Multi-tier defense & Rescale vs Median.
    - `tab_ablation_suite.tex` (T5): M1-M7 Ablation suite.
    - `tab_adaptive_poisoner.tex` (T6): Proximity stealth & Median backfire.
    - `tab_alignment_adversary.tex` (T7): White-box T1 adversary & Kerckhoffs boundary.
    - `tab_sota_main.tex` (T8): SOTA Pokec-z & Credit with FLTrust defensive concession.
    - `tab_cost.tex` (T9): Hardware overhead & communication footprint.
    - `tab_factorial_2x2.tex` (T10): Factorial 2x2 grid on $w_{\mathrm{adv}}$ AND $\text{DPD}_{\mathrm{hard}}$.
    - `tab_topology_stress.tex` (T11): Metis community partition & Dirichlet sweep.
    - `tab_trust_score_sensitivity.tex` (T12): Monte Carlo 2,000 configurations.

### Deprecated & Removed from Old Manuscript (`manuscript_neurocomputing/`)
- Completely excluded stale tables identified in `plan_floats_CAU_CHUYEN.md` §1:
  - `main_auc.tex`, `main_dpd.tex`, `main_eod.tex` (stale numbers from previous campaign).
  - `robustness.tex` (stale baseline labels identifying BFWA as ours).
  - `large_scale.tex` (stale AUC numbers).
  - `efficiency.tex` (placeholder zero energy columns).
  - Duplicate `shapley_fidelity.tex` / `privacy_attack.tex`.

- **Unified Section Restructuring**:
  - Combined Sections 03, 04, 05 into unified **`sections/03_methodology.tex`** (`\section{Methodology}`):
    - `\subsection{Problem Formulation and Threat Model}` (Graph setting, Kerckhoffs 3-tier threat model).
    - `\subsection{The \MethodName{} Framework}` (Holdout reference, FTGD orthogonal surgery, EMA coordinate gating).
    - `\subsection{Theoretical Analysis and Lower Bounds}` (Folded-normal perpetual bias, Le Cam minimax bound, linear Shapley decomposition).
  - Combined Sections 06 and 07 into unified **`sections/04_implementation_results.tex`** (`\section{Implementation and Empirical Results}`):
    - `\subsection{Experimental Setup and Benchmark Protocol}` (Setup, 4 benchmarks, master config, palette).
    - `\subsection{Vulnerability and Collapse of Client-Reported Fairness Interfaces}` (Hồi 1).
    - `\subsection{Multi-Tier Adversarial Stress-Testing and Empirical Resilience}` (Hồi 2).
    - `\subsection{Causal Attribution, Architectural Hygiene, and Auditable Governance}` (Hồi 3).
    - `\subsection{Operational Boundaries and Scope of Validity}` (5 operational boundaries & limitations).
  - Streamlined `main.tex` into 5 standard journal sections: Introduction, Related Work, Methodology, Implementation and Empirical Results, Conclusion.
  - Archived old fragmented files into `sections/_archive/`.

### Verified
- **Compilation**: Successfully compiled `manuscript_v2/main.tex` to 25-page PDF (`main.pdf`) using `pdflatex` + `bibtex` with zero undefined references or missing citations.
- **Blacklist Linter**: Ran `python3 FedFairGNN/scripts/lint_manuscript_blacklist.py` across all sections, tables, and documents: **0 violations detected (100% CLEAN)**.
- **Float Suite Audit & Provenance Verification (plan_floats_CAU_CHUYEN.md)**:
  - Added warning banner `% ⛔ STALE — chiến dịch cũ, KHÔNG input. Xem docs/review/plan_floats_CAU_CHUYEN.md §1` to all 8 archived legacy tables in `manuscript_neurocomputing/tables/_archive/`.
  - Harmonized `tab_sota_main.tex` caption edge count to exact invariant $|E|=617,958$ undirected edges.
  - Upgraded `tab_ldp_barrier.tex` (T2) to unify Differential Privacy Accounting, folded-normal noise scale $\tilde{\sigma} \in [0.2127, 7.3886]$, and Le Cam Minimax Testing Error lower bounds $\mathcal{R}^*$ ($\ge 31.4\%$ to $\ge 49.5\%$).
  - Created 6 audited supplementary tables in `manuscript_v2/tables/supplementary/`:
    1. `tab_weight_stability.tex`: Multi-regime reconciliation of $\Omega_w$ across Pokec-z ($7.37\times$ vs.\ FLTrust), German Credit ablation ($24.1\times$ vs.\ M7 w/o EMA), and historical dual-ascent BFWA ($27.4\times$).
    2. `tab_trust_fidelity.tex`: Limitation scope documenting that FU-Alignment fails 4/5 combinatorial Shapley criteria ($L_1$ distance $0.7453$), formalizing it as a first-order directional projection heuristic ($O(KP)$).
    3. `tab_proxy_sensitivity.tex`: Discloses sensitivity gap where FedAvg outperforms on topological degree hubs.
    4. `tab_centralized_sanity.tex`: Reports full-graph centralized anchors and federated partition generalization gaps.
    5. `tab_compliance.tex`: Maps regulatory governance principles (EU AI Act & NIST AI RMF) with objective, non-overclaiming language.
    6. `tab_update_attack.tex` (T3): Detailed attribute inference leakage across observation channels (Bail).
- **Figure Suite Generation & Publication-Ready Embedding (plan_floats_CAU_CHUYEN.md)**:
  - Created automated figure orchestrator `FedFairGNN/experiments/make_manuscript_v2_figures.py`.
  - Generated and embedded all 4 critical vector figures in `manuscript_v2/figures/`:
    1. `fig_interface_boundary.pdf` (F1): Slope / dumbbell chart mapping 10 rules across 3 interface classes, visually contrasting the 6 self-reported rules that collapse under strategic falsification with the 4 gradient-only rules that maintain $\Delta w = 0.0000$ bit-exact immunity.
    2. `fig_robustness_byz.pdf` (F3): Adversary share ratio $w_{\mathrm{adv}} / (f/K)$ across corruption ratios $f/K \in \{0.1, 0.2, 0.3, 0.4\}$, capturing the shielding paradox where coordinate median filtering inadvertently backfires by purging benign minority updates under proximity stealth.
    3. `fig_dynamics.pdf` (F4): 2-panel figure combining (a) 20-round non-monotonic convergence dynamics on Bail Recidivism and (b) round-by-round client aggregation weight trajectories, showcasing how EMA smoothing suppresses path turbulence ($\Omega_w = 0.0586$ vs.\ $16.5407$ in unsmoothed dual-ascent).
    4. `fig_pareto.pdf` (F5): Multi-objective Pareto frontier across Credit Default and Pokec-z containing all 11 evaluated algorithms without selective omission, explicitly including FLTrust (highest AUC $0.8057$) and FedGraph-Fair (lowest disparity $0.0050$).
  - Full manuscript recompilation: `main.pdf` compiled cleanly to 27 pages (915,276 bytes) with 0 blacklist violations.



