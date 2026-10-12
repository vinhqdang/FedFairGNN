# Cover Letter for Elsevier Neurocomputing

**Date:** September 3, 2026  
**To:** The Editor-in-Chief, *Neurocomputing* (Elsevier)  

**Subject:** Submission of Original Research Article titled *"TrustFedGNN: A Byzantine-Robust and Differentially Private Federated Graph Neural Network with Fairness Constraints"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript titled **"TrustFedGNN: A Byzantine-Robust and Differentially Private Federated Graph Neural Network with Fairness Constraints"** for consideration as a regular research paper in *Neurocomputing*.

### 1. Research Context & Motivation
Graph Neural Networks (GNNs) deployed in decentralized, cross-silo domains—such as financial fraud detection, anti-money laundering (AML), and risk scoring—face a severe tripartite bottleneck termed the **Trustworthy Trilemma**:
1. *Homophily vs. Fairness*: Structural message passing systematically amplifies demographic disparities across connected nodes.
2. *Fairness vs. Differential Privacy*: Traditional $(\epsilon, \delta)$-DP-SGD mechanisms add isotropic noise scaling with parameter dimension $O(\sqrt{|\theta|})$, devastating fraud-detection utility on compact GNNs.
3. *Fairness Steering vs. Byzantine Resilience*: Reliance on client-reported metadata enables malicious participants to execute deceptive poisoning by advertising zero disparity while steering the global model toward biased local objectives.

### 2. Methodological Innovation & Technical Highlights
To address these intertwined challenges, this paper presents **TrustFedGNN**, an integrated framework structured around three cohesive pillars:
- **Pillar C1 (Client-Side Attention Debiasing & Targeted DP)**: Introduces *Fairness-Sensitive Edge Reweighting (FSER)*, a learned attention penalty that suppresses cross-group edge weights. Crucially, empirical ablation across 10 random seeds demonstrates that FSER acts as an inductive regularizer on natural homophilic graphs, improving AUC by **+0.0201 ($p=0.0059$)**. Concurrently, *Fairness-Targeted Gradient Decomposition (FTGD)* projects task updates orthogonal to fairness gradients ($g_{\text{task}}^\perp \perp g_{\text{fair}}$) and injects calibrated Gaussian noise solely into the 2D scalar disparity statistics, achieving certified statistic-level DP while reducing attack AUC on that released statistic from $1.000$ to $0.500$ (random guessing).
- **Pillar C2 (Server-Side Byzantine-Resilient Simplex Aggregation)**: Introduces *Bi-objective Frank–Wolfe Aggregation (BFWA)*, formulating federated model merging as a dual-constrained optimization over the probability simplex $\Delta_K$, solved via dual ascent that steers weights toward the fairness budget (feasibility reported as a constraint residual). Furthermore, coordinate-wise median aggregation's robustness to an omniscient adaptive stealth fairness adversary is structural---it consumes no self-reported metadata---up to the standard Byzantine breakdown fraction $f < 0.5$.
- **Pillar C3 (Empirical Rigor & Scalability)**: The theoretical foundations are backed by formal derivations of the privacy and non-leakage guarantees. Experiments across 6 benchmark datasets—ranging from social networks (Pokec-z, 1.24M edges) to transaction networks (Elliptic Bitcoin, 203k transactions) and massive commercial graphs (ogbn-products, 2.4M nodes)—confirm that TrustFedGNN consistently outperforms 16 state-of-the-art baselines.

### 3. Relevance to *Neurocomputing*
*Neurocomputing* has long been a premier venue for advanced neural network architectures, robust optimization, and trustworthy learning paradigms. Our work directly advances the intersection of graph neural computation, federated optimization, and ethical AI governance (aligned with Articles 10 & 14 of the EU AI Act and the NIST AI RMF).

### 4. Declarations
- This manuscript represents original, unpublished work and is not under consideration elsewhere.
- The authors have no competing interests to declare.
- Full source code, pre-registered experiment runners, and verification test suites are publicly accessible.

Thank you very much for your consideration of this work. We look forward to receiving the reviewers' feedback.

Sincerely,

**Ngoc-Son-An Nguyen** (corresponding author)
Industrial University of Ho Chi Minh City, Vietnam  
Corresponding Email: `annns25871@pgr.iuh.edu.vn`

**Quang-Vinh Dang**
British University Vietnam, Hung Yen, Vietnam
Email: `vinh.dq4@buv.edu.vn`
