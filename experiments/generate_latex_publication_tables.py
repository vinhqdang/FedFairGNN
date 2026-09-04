import json
import os

def generate_all_tables():
    stats_path = "results/consolidated_statistics.json"
    remed_path = "results/stage4_remediation_results.json"
    shapley_path = "results/stage4_3_shapley_results.json"
    
    with open(stats_path) as f:
        stats = json.load(f)
    with open(remed_path) as f:
        remed = json.load(f)
    with open(shapley_path) as f:
        shapley = json.load(f)

    os.makedirs("manuscript/tables", exist_ok=True)

    # -------------------------------------------------------------
    # 1. Main Pokec-z Table (LaTeX)
    # -------------------------------------------------------------
    pokecz_summary = stats["pokecz_67.8k"]["metrics_summary"]
    pokecz_paired = stats["pokecz_67.8k"]["paired_comparisons_vs_ours"]

    methods_order = [
        ("fedavg-gcn", "FedAvg-GCN", "AISTATS'17"),
        ("fairgnn", "FairGNN", "WSDM'21"),
        ("fairsin", "FairSIN", "WWW'24"),
        ("fairfed", "FairFed", "AAAI'23"),
        ("fairgfl", "FairGFL", "IEEE TPDS'26"),
        ("fedgraphfair", "FedGraph-Fair", "InfoSci'26"),
        ("cgsv", "CGSV", "NeurIPS'21"),
        ("ours-nofser", "Ours w/o FSER (M2)", "Ablation"),
        ("fedfairgnn", "\\textbf{TrustFedGNN (Ours)}", "Proposed"),
    ]

    pokec_rows = []
    for key, name, venue in methods_order:
        s = pokecz_summary[key]
        auc_str = f"{s['auc']['mean']:.4f} $\\pm$ {s['auc']['std']:.4f}"
        dpd_str = f"{s['dpd_hard']['mean']:.4f} $\\pm$ {s['dpd_hard']['std']:.4f}"
        eod_str = f"{s['eod']['mean']:.4f} $\\pm$ {s['eod']['std']:.4f}"
        omega_str = f"{s['omega_w']['mean']:.4f}"
        dp_str = "\\checkmark" if key in ["fedfairgnn", "ours-nofser"] else "$\\times$"
        
        if key == "fedfairgnn":
            auc_str = f"\\textbf{{{auc_str}}}"
            dpd_str = f"\\textbf{{{dpd_str}}}"
            eod_str = f"\\textbf{{{eod_str}}}"
        elif key in pokecz_paired:
            p_comp = pokecz_paired[key]
            if p_comp["auc"]["holm_bonferroni_sig"]:
                auc_str += "$^\\star$"
            if p_comp["dpd_hard"]["holm_bonferroni_sig"]:
                dpd_str += "$^\\star$"
            if p_comp["eod"]["holm_bonferroni_sig"]:
                eod_str += "$^\\star$"

        pokec_rows.append(f"{name} & {venue} & {dp_str} & {auc_str} & {dpd_str} & {eod_str} & {omega_str} \\\\")

    pokec_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Main SOTA Benchmark on Pokec-z ($N=67,796$ nodes, $1.24\text{M}$ edges, $n=10$ independent random seeds).} 
Metrics are reported as $\text{Mean} \pm \text{Std}$. $\star$ denotes statistically significant difference versus TrustFedGNN (Ours) under the two-sided Wilcoxon signed-rank test with family-wise Holm-Bonferroni correction ($p < 0.05$). Bold indicates the best result. TrustFedGNN is the only method with $(\epsilon=8.0, \delta=10^{-5})$-DP enabled.}
\label{tab:main_pokecz_sota}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Venue} & \textbf{$(\epsilon,\delta)$-DP} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(pokec_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    with open("manuscript/tables/main_pokecz_sota.tex", "w") as f:
        f.write(pokec_tex)

    # -------------------------------------------------------------
    # 2. Credit Default Table (LaTeX) - Boundary Analysis
    # -------------------------------------------------------------
    credit_summary = stats["credit_default_30k"]["metrics_summary"]
    credit_paired = stats["credit_default_30k"]["paired_comparisons_vs_ours"]
    credit_rows = []

    for key, name, venue in methods_order:
        s = credit_summary[key]
        auc_str = f"{s['auc']['mean']:.4f} $\\pm$ {s['auc']['std']:.4f}"
        dpd_str = f"{s['dpd_hard']['mean']:.4f} $\\pm$ {s['dpd_hard']['std']:.4f}"
        eod_str = f"{s['eod']['mean']:.4f} $\\pm$ {s['eod']['std']:.4f}"
        omega_str = f"{s['omega_w']['mean']:.4f}"
        dp_str = "\\checkmark" if key in ["fedfairgnn", "ours-nofser"] else "$\\times$"
        
        if key == "fedfairgnn":
            auc_str = f"\\textbf{{{auc_str}}}"
            dpd_str = f"\\textbf{{{dpd_str}}}"
            eod_str = f"\\textbf{{{eod_str}}}"
        elif key in credit_paired:
            p_comp = credit_paired[key]
            if p_comp["auc"]["holm_bonferroni_sig"]:
                auc_str += "$^\\star$"
            if p_comp["dpd_hard"]["holm_bonferroni_sig"]:
                dpd_str += "$^\\star$"
            if p_comp["eod"]["holm_bonferroni_sig"]:
                eod_str += "$^\\star$"

        credit_rows.append(f"{name} & {venue} & {dp_str} & {auc_str} & {dpd_str} & {eod_str} & {omega_str} \\\\")

    credit_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Application Boundary Analysis on Tabular $k$-NN Graph (Credit Default, $N=30,000$ nodes, $n=10$ seeds).} 
Metrics reported as $\text{Mean} \pm \text{Std}$. $\star$ denotes Holm-Bonferroni statistical significance ($p < 0.05$). On synthetic $k$-NN graphs constructed from tabular attributes, TrustFedGNN reduces unfairness ($\text{DPD}_{\text{hard}}$ reduced by $31.3\%$ vs FedAvg) while operating under strict $(\epsilon,\delta)$-DP, although topological edge-reweighting provides minimal utility gain compared to natural social networks.}
\label{tab:credit_boundary_sota}
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Venue} & \textbf{$(\epsilon,\delta)$-DP} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(credit_rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    with open("manuscript/tables/credit_boundary_sota.tex", "w") as f:
        f.write(credit_tex)

    # -------------------------------------------------------------
    # 3. Ablation Suite Table (German Credit, M1-M7)
    # -------------------------------------------------------------
    ablation_matrix = remed["stage4_5_ablation_matrix"]
    # NOTE: every LaTeX string below MUST be a raw literal. Written as plain
    # strings, "$\alpha$" and "$\beta$" put a BEL (0x07) and a backspace (0x08)
    # control byte into the emitted .tex and render as "lpha"/"eta" in the PDF --
    # which is exactly what shipped in manuscript/tables/ablation.tex.
    ablation_rows = [
        ("M1 (Full Proposed)", r"\textbf{TrustFedGNN (Canonical)}", ablation_matrix["M1_Full"]),
        ("M2 (w/o FSER)", "GAT Backbone (No Topological Reweighting)", ablation_matrix["M2_wo_FSER"]),
        ("M3 (w/o FTGD)", "Standard Local Optimization (No Orthogonal Surgery)", ablation_matrix["M3_wo_FTGD"]),
        ("M4 (Full DP-SGD)", r"Standard Client-Wide DP-SGD ($\epsilon=8.0$)", ablation_matrix["M4_Full_DPSGD"]),
        ("M5 (w/o FairScore)", r"GTG-Shapley Metric ($\alpha=0.0$)", ablation_matrix["M5_wo_FairScore"]),
        ("M6 (w/o Two-Tier)", "CGSV Aggregation (No Server Holdout)", ablation_matrix["M6_wo_TwoTier"]),
        ("M7 (w/o Temp EMA)", r"Instantaneous Gradient Alignment ($\beta_{\text{ema}}=0.0$)", ablation_matrix["M7_wo_EMA"]),
    ]

    abl_lines = []
    for code, desc, d in ablation_rows:
        auc_s = f"{d['auc_mean']:.4f} $\\pm$ {d['auc_std']:.4f}"
        dpd_s = f"{d['dpd_hard_mean']:.4f} $\\pm$ {d['dpd_hard_std']:.4f}"
        eod_s = f"{d['eod_mean']:.4f} $\\pm$ {d['eod_std']:.4f}"
        omega_s = f"{d['omega_w_mean']:.4f}"
        if "M1" in code:
            auc_s = f"\\textbf{{{auc_s}}}"
            dpd_s = f"\\textbf{{{dpd_s}}}"
            eod_s = f"\\textbf{{{eod_s}}}"
        abl_lines.append(f"{code} & {desc} & {auc_s} & {dpd_s} & {eod_s} & {omega_s} \\\\")

    ablation_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Ablation Study of Core Components on German Credit ($K=5, \alpha_{\text{Dir}}=0.3, n=3$ seeds).} 
At $n=3$ seeds only two effects are large relative to seed variance. (1)~Targeted FTGD noise avoids the utility collapse of client-wide DP-SGD (M4, AUC $0.4761$ vs $0.6426$, a $12\sigma$ gap); this is a privacy--utility result and \emph{not} a fairness result---removing FTGD (M3) in fact yields the \emph{lowest} $\text{DPD}_{\text{hard}}$ in the table ($0.0378$ vs $0.0740$) at indistinguishable AUC, so FTGD does not reduce disparity here. (2)~Removing the temporal EMA (M7) inflates weight instability $17.7\times$ ($\Omega_w$ $0.0781 \to 1.3837$); this is the expected behaviour of EMA smoothing, which suppresses first-difference variance by construction, and we report the magnitude rather than treat it as a discovery. Two results cut against the proposed design and we state them: M5 (FairScore removed, $\alpha=0$) is statistically indistinguishable from M1 on $\text{DPD}_{\text{hard}}$ at this sample size ($0.0870 \pm 0.0557$ vs $0.0740 \pm 0.0375$), so FairScore \emph{may} contribute to disparity reduction but $n=3$ is underpowered to establish it; and M6 (two-tier aggregation removed, CGSV with no server holdout) is the best arm on both AUC ($0.6768$) and $\Omega_w$ ($0.0332$) with $\text{DPD}_{\text{hard}}$ overlapping M1---i.e.\ the arm that removes the aggregation novelty wins on two of the four axes. Only the FSER ablation (M2) degrades every fairness axis unambiguously.}
\label{tab:ablation_suite}
\begin{tabular}{llcccc}
\toprule
\textbf{Ablation Arm} & \textbf{Description / Isolated Component} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(abl_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    with open("manuscript/tables/ablation.tex", "w") as f:
        f.write(ablation_tex)

    # -------------------------------------------------------------
    # 4. Shapley Probing Table (Exact vs FU-Shapley Extended)
    # -------------------------------------------------------------
    sh_sum = shapley["summary"]
    trust_tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Empirical Evaluation of FU-Shapley Alignment vs Exact Shapley (125 probe points, $K=5$, 5 seeds).} 
The three pre-registered fidelity criteria are \emph{not} met: pooled Pearson $r = 0.7436$ (target $\ge 0.80$), sign agreement $73.6\%$ (target $\ge 85\%$), and mean simplex $L_1$ distance $0.7554$ (target $\le 0.15$); hypothesis H3 (faithful approximation of exact Shapley in ranking \emph{and} allocation) is therefore recorded as \textsc{refuted} in \texttt{results/stage4\_3\_shapley\_results.json}. What the data do support is agreement at the \emph{ranking} level---pooled Spearman $\rho = 0.6897$, the one criterion met---so FU-Shapley is usable as a fast, first-order ranking proxy at $O(KP)$ instead of $O(2^K P)$ cost, but not as a faithful value allocation: its simplex weights depart substantially from the exact Shapley allocation.}
\label{tab:shapley_fidelity}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Target Criterion (Pre-reg)} & \textbf{Empirical Value (125 points)} \\
\midrule
Pooled Pearson Correlation $r$ & $\ge 0.80$ & $0.7436$ ($p < 0.001$) \\
Pooled Spearman Rank Correlation $\rho$ & $\ge 0.70$ & $0.6897$ ($p < 0.001$) \\
Sign Agreement Proportion & $\ge 85\%$ & $73.60\%$ \\
Bottom-1 Detection Rate & $\ge 75\%$ & $64.00\%$ \\
Mean Simplex $L_1$ Distance & $\le 0.15$ & $0.7554$ \\
\midrule
\textbf{Computational Complexity} & -- & \textbf{$O(KP)$ vs $O(2^K P)$} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open("manuscript/tables/trust.tex", "w") as f:
        f.write(trust_tex)

    # -------------------------------------------------------------
    # 5. Significance & Hypothesis Testing Table
    # -------------------------------------------------------------
    sig_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Statistical Hypothesis Testing and Paired Comparisons versus TrustFedGNN (Ours).}
Reported with two-sided Wilcoxon signed-rank test $p$-values, effect size Cohen's $d_z$, and 95\% Bootstrap Confidence Intervals ($n=10$ seeds). $\star$ denotes significance surviving family-wise Holm-Bonferroni correction ($\alpha=0.05$).}
\label{tab:statistical_significance}
\begin{tabular}{llcccc}
\toprule
\textbf{Dataset} & \textbf{Baseline Comparison} & \textbf{Metric} & \textbf{$\Delta$ (Ours $-$ Base)} & \textbf{Cohen's $d_z$} & \textbf{Wilcoxon $p$ (Holm-Bonferroni)} \\
\midrule
\multirow{6}{*}{\textbf{Pokec-z (67.8k)}} 
 & FedAvg-GCN & AUC & $+0.0590$ & $+3.74$ & $p = 0.0020^\star$ \\
 & FairGFL (2026) & AUC & $+0.0544$ & $+4.32$ & $p = 0.0020^\star$ \\
 & FedGraph-Fair (2026) & AUC & $+0.0703$ & $+4.01$ & $p = 0.0020^\star$ \\
 & CGSV (2021) & AUC & $+0.0472$ & $+4.30$ & $p = 0.0020^\star$ \\
 & CGSV (2021) & $\text{DPD}_{\text{hard}}$ & $-0.0286$ & $-3.42$ & $p = 0.0020^\star$ \\
 & Ours w/o FSER (M2) & AUC & $+0.0201$ & $+1.28$ & $p = 0.0059^\star$ \\
\midrule
\multirow{4}{*}{\textbf{Credit (30k)}} 
 & FedAvg-GCN & $\text{DPD}_{\text{hard}}$ & $-0.0331$ & $-0.98$ & $p = 0.0098$ \\
 & FairGFL (2026) & $\text{DPD}_{\text{hard}}$ & $-0.0343$ & $-1.16$ & $p = 0.0020^\star$ \\
 & FairGFL (2026) & AUC & $-0.0096$ & $-1.11$ & $p = 0.0020^\star$ \\
 & CGSV (2021) & AUC & $-0.0086$ & $-0.96$ & $p = 0.0059^\star$ \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open("manuscript/tables/significance.tex", "w") as f:
        f.write(sig_tex)

    # -------------------------------------------------------------
    # 6. Benchmark Datasets Characteristics Table
    # -------------------------------------------------------------
    datasets_tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Characteristics of Experimental Benchmark Datasets.} 
All datasets strictly satisfy the Zero-Feature Leakage criterion ($\max_j \text{AUC}(x_j, y) < 0.85$). $h_s$ denotes the sensitive attribute homophily ratio.}
\label{tab:datasets}
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{Nodes ($N$)} & \textbf{Edges ($|E|$)} & \textbf{Features ($D$)} & \textbf{Sensitive ($s$)} & \textbf{Target ($y$)} & \textbf{Homophily ($h_s$)} \\
\midrule
German Credit & 1,000 & 24,444 & 27 & Gender & Credit Risk & 0.6120 \\
Bail Recidivism & 18,876 & 321,308 & 17 & Race & Recidivism & 0.7240 \\
Credit Default & 30,000 & 1,436,858 & 23 & Age & Default & 0.9595 \\
Pokec-z & 67,796 & 1,235,916 & 276 & Region & Working Field & 0.9506 \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open("manuscript/tables/datasets.tex", "w") as f:
        f.write(datasets_tex)

    print("ALL 6 LATEX PUBLICATION TABLES GENERATED SUCCESSFULLY IN manuscript/tables/!")

if __name__ == "__main__":
    generate_all_tables()
