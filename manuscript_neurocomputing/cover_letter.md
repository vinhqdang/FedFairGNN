# Cover Letter for Elsevier Neurocomputing

**Date:** September 3, 2026  
**To:** The Editor-in-Chief, *Neurocomputing* (Elsevier)  

**Subject:** Submission of Original Research Article titled *"TrustFedGNN: A Byzantine-Robust and Differentially Private Federated Graph Neural Network with Fairness Constraints"*

Dear Editor-in-Chief,

I am pleased to submit our original research manuscript titled **"TrustFedGNN: A Byzantine-Robust and Differentially Private Federated Graph Neural Network with Fairness Constraints"** for consideration as a regular research paper in *Neurocomputing*.

### 1. Research Context & Motivation
Graph Neural Networks (GNNs) deployed in decentralized, cross-silo domains—such as financial fraud detection, anti-money laundering (AML), and risk scoring—face a severe tripartite bottleneck termed the **Trustworthy Trilemma**:
1. *Homophily vs. Fairness*: Structural message passing systematically amplifies demographic disparities across connected nodes.
2. *Fairness vs. Differential Privacy*: Traditional $(\epsilon, \delta)$-DP-SGD mechanisms add isotropic noise scaling with parameter dimension $O(\sqrt{|\theta|})$, devastating fraud-detection utility on compact GNNs.
3. *Fairness Steering vs. Byzantine Resilience*: Reliance on client-reported metadata enables malicious participants to execute deceptive poisoning by advertising zero disparity while steering the global model toward biased local objectives.

### 2. Methodological Innovation & Technical Highlights
To address these intertwined challenges, this manuscript presents **TrustFedGNN**, an integrated framework structured around verifiable trustworthiness guarantees:
- **Client-Side Targeted DP & Minimax Bounds**: Introduces statistic-level Differential Privacy via scalar group statistics, circumventing the $O(\sqrt{|\theta|})$ noise penalty of classical DP-SGD. We establish a minimax observability lower bound via Le Cam's lemma (Theorem 4), proving that local DP on client-side disparity statistics fundamentally limits group fairness observability.
- **Server-Side Byzantine-Resilient Reference Aggregation**: Implements server-referenced aggregation with holdout gradient scoring (`fu_shapley`), provably achieving bit-exact immunity against metadata falsification ($\Delta w = 0.0000$, Theorem 2). Under gradient scaling Byzantine attacks, payload-bounded gradient scoring maintains utility and fairness stability, while median-screened aggregation completely purges attacker weight ($w_{\text{adv}} = 0.0000$ across 5 seeds $\times$ 3 Byzantine ratios).
- **Empirical Rigor & Pre-Registered Audit**: Rather than asserting universal superiority across all fairness and utility metrics, we report complete per-seed evaluations across benchmark datasets (Pokec-z, Credit Default, German Credit, and Bail Recidivism) with rigorous Holm–Bonferroni corrections, explicitly documenting setting-specific trade-offs and structural scope conditions.

### 3. Relevance to *Neurocomputing*
*Neurocomputing* has long been a premier venue for advanced neural network architectures, robust optimization, and trustworthy learning paradigms. Our work directly advances the intersection of graph neural computation, federated optimization, and ethical AI governance (aligned with Articles 10 & 14 of the EU AI Act and the NIST AI RMF).

### 4. Declarations
- This manuscript represents original, unpublished work and is not under consideration elsewhere.
- The author has no competing interests to declare.
- Full source code, pre-registered experiment runners, and verification test suites are publicly accessible.

Thank you very much for your consideration of this work. I look forward to receiving the reviewers' feedback.

Sincerely,

**Ngoc-Son-An Nguyen**  
Industrial University of Ho Chi Minh City, Vietnam  
Corresponding Email: `nnsan@iuh.edu.vn` (or personal contact)
