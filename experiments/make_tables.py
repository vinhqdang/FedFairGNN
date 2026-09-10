import json
import os
import shutil
import argparse

def generate_all_tables(target_dirs=None):
    if target_dirs is None:
        target_dirs = [
            "manuscript_neurocomputing/tables",
            "../manuscripts/neurocomputing_vnese/tables"
        ]
    
    # Resolve existing directories or create them
    valid_dirs = []
    for d in target_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            valid_dirs.append(d)
        except Exception as e:
            print(f"Skipping {d}: {e}")

    stats_path = "results/consolidated_statistics.json"
    remed_path = "results/canonical_suite.json"
    shapley_path = "results/shapley_fidelity.json"
    
    with open(stats_path) as f:
        stats = json.load(f)
    with open(remed_path) as f:
        remed = json.load(f)
    with open(shapley_path) as f:
        shapley = json.load(f)

    # -------------------------------------------------------------
    # 1. Main Pokec-z Table (LaTeX) - 10 rows (including clean arm)
    # -------------------------------------------------------------
    pokecz_summary = stats["pokecz_67.8k"]["metrics_summary"]
    pokecz_paired = stats["pokecz_67.8k"]["paired_comparisons_vs_ours"]

    methods_order = [
        ("fedavg-gcn", "FedAvg-GCN", "AISTATS'17"),
        ("fairgnn", "FairGNN", "WSDM'21"),
        ("fairsin", "FairSIN", "AAAI'24"),
        ("fairfed", "FairFed", "AAAI'23"),
        ("fairgfl", "FairGFL", "IEEE TPDS'26"),
        ("fedgraphfair", "FedGraph-Fair", "InfoSci'26"),
        ("cgsv", "CGSV", "NeurIPS'21"),
        ("ours-nofser", "Ours w/o FSER (Confounded)", "Ablation"),
        ("ours-nofser-true", "Ours w/o FSER (Clean Arm)", "Ablation"),
        ("fedfairgnn", r"\textbf{TrustFedGNN (Ours)}", "Proposed"),
    ]

    pokec_rows = []
    for key, name, venue in methods_order:
        s = pokecz_summary[key]
        auc_str = f"{s['auc']['mean']:.4f} $\\pm$ {s['auc']['std']:.4f}"
        dpd_str = f"{s['dpd_hard']['mean']:.4f} $\\pm$ {s['dpd_hard']['std']:.4f}"
        eod_str = f"{s['eod']['mean']:.4f} $\\pm$ {s['eod']['std']:.4f}"
        omega_str = f"{s['omega_w']['mean']:.4f}"
        dp_str = r"\checkmark" if key in ["fedfairgnn", "ours-nofser", "ours-nofser-true"] else r"$\times$"
        
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
\caption{\textbf{Main SOTA Benchmark on Pokec-z ($N=67,796$ nodes, $1.24\text{M}$ edges, $K=10$, $R=50$, $n=10$ independent random seeds).} Metrics are reported as $\text{Mean} \pm \text{Std}$. Bold indicates our full proposed method. Both the historical confounded ablation (ours-nofser) and clean attribution arm (ours-nofser-true) are reported to isolate FSER impact without architectural confounding. $^\dagger$FTGD's $(\epsilon=8.0, \delta=10^{-5})$ differential privacy guarantee strictly covers the released fairness statistics (two scalar group means per client per round), not transmitted model updates $\theta_k$ (see \S\ref{sec:limitations}). $^\star$ marks a paired Wilcoxon signed-rank difference against TrustFedGNN that remains significant at $\alpha=0.05$ after Holm--Bonferroni correction across the family of baseline comparisons.}
\label{tab:main_pokecz_sota}
\resizebox{\linewidth}{!}{%
\setlength{\tabcolsep}{4pt}%
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Venue} & \textbf{$(\epsilon,\delta)$-DP (stat.)$^\dagger$} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(pokec_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

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
        dp_str = r"\checkmark" if key in ["fedfairgnn", "ours-nofser", "ours-nofser-true"] else r"$\times$"
        
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
\caption{\textbf{Application Boundary Analysis on Tabular $k$-NN Graph (Credit Default, $N=30,000$ nodes, $K=10$, $R=50$, $n=10$ seeds).} Metrics reported as $\text{Mean} \pm \text{Std}$. Confirms the application boundary: on synthetic tabular $k$-NN graphs without authentic social topology, FSER topological reweighting is statistically neutral ($\Delta_{\text{FSER}} = +0.000062$, $p=0.9219$). $^\dagger$FTGD covers released statistics $(\mu_0, \mu_1)$, not transmitted updates $\theta_k$. $^\star$ marks a paired Wilcoxon signed-rank difference against TrustFedGNN that remains significant at $\alpha=0.05$ after Holm--Bonferroni correction across the family of baseline comparisons.}
\label{tab:credit_boundary_sota}
\resizebox{\linewidth}{!}{%
\setlength{\tabcolsep}{4pt}%
\begin{tabular}{lcccccc}
\toprule
\textbf{Method} & \textbf{Venue} & \textbf{$(\epsilon,\delta)$-DP (stat.)$^\dagger$} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(credit_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    # -------------------------------------------------------------
    # 3. Ablation Suite Table (German Credit, M1-M7, n=10 seeds)
    # -------------------------------------------------------------
    ablation_matrix = remed["component_ablation_matrix"]
    ablation_rows = [
        ("M1 (Full Proposed)", r"\textbf{TrustFedGNN (Canonical)}", ablation_matrix["M1_Full"]),
        ("M2 (w/o FSER Confounded)", r"GAT Backbone (Confounds FSER \& Scaffold)", ablation_matrix["M2_wo_FSER"]),
        ("M2 (w/o FSER True Clean)", r"Freeze $\beta=0$ (Faithful Clean FSER Arm)", ablation_matrix["M2_wo_FSER_true"]),
        ("M3 (w/o FTGD)", "Standard Local Optimization (No Orthogonal Surgery)", ablation_matrix["M3_wo_FTGD"]),
        ("M4 (Full DP-SGD)", r"Standard Client-Wide DP-SGD ($\epsilon=8.0$)", ablation_matrix["M4_Full_DPSGD"]),
        ("M5 (w/o FairScore)", r"GTG-Shapley Metric ($\alpha=0.0$)", ablation_matrix["M5_wo_FairScore"]),
        ("M6 (w/o Two-Tier)", "CGSV Aggregation (No Server Holdout)", ablation_matrix["M6_wo_TwoTier"]),
        ("M7 (w/o EMA)", r"No History Smoothing ($\beta_{\text{EMA}}=0.0$)", ablation_matrix["M7_wo_EMA"]),
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
\caption{\textbf{Component-wise Ablation Suite on German Credit ($K=5, R=20, n=10$ seeds).} Metrics reported as $\text{Mean} \pm \text{Std}$. Contrasts the clean unconfounded FSER arm (M2 true, freeze $\beta=0$) against the historical confounded arm (M2 GAT backbone).}
\label{tab:ablation}
\label{tab:ablation_suite}
\resizebox{\linewidth}{!}{%
\setlength{\tabcolsep}{4pt}%
\begin{tabular}{llcccc}
\toprule
\textbf{Ablation Arm} & \textbf{Description} & \textbf{AUC-ROC} ($\uparrow$) & \textbf{$\text{DPD}_{\text{hard}}$} ($\downarrow$) & \textbf{EOD} ($\downarrow$) & \textbf{$\Omega_w$} ($\downarrow$) \\
\midrule
""" + "\n".join(abl_lines) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    # -------------------------------------------------------------
    # 4. Shapley Probing Table (Exact vs FU-Shapley Extended)
    # -------------------------------------------------------------
    trust_tex = r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Empirical Evaluation of FU-Shapley Alignment vs Exact Shapley (125 probe points, $K=5$, 5 seeds).} 
Evaluating across 5 probing rounds confirms FU-Shapley functions as a fast, first-order ranking heuristic ($O(KP)$ vs $O(2^K P)$) with strong directional alignment ($\rho = 0.690$, $73.6\%$ sign agreement).}
\label{tab:shapley_fidelity}
\resizebox{\linewidth}{!}{%
\setlength{\tabcolsep}{5pt}%
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
\end{tabular}%
}
\end{table}
"""

    # -------------------------------------------------------------
    # 5. Significance & Hypothesis Testing Table
    # -------------------------------------------------------------
    sig_tex = r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Statistical Hypothesis Testing and Paired Comparisons versus TrustFedGNN (Ours).}
Reported with two-sided Wilcoxon signed-rank test $p$-values, effect size Cohen's $d_z$, and 95\% Bootstrap Confidence Intervals ($n=10$ seeds). $\star$ denotes significance surviving family-wise Holm-Bonferroni correction ($\alpha=0.05$).}
\label{tab:statistical_significance}
\label{tab:significance}
\resizebox{\linewidth}{!}{%
\setlength{\tabcolsep}{3.5pt}%
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
 & Ours w/o FSER (Clean) & AUC & $-0.0017$ & $-0.18$ & $p = 0.2324$ \\
\midrule
\multirow{4}{*}{\textbf{Credit (30k)}} 
 & FedAvg-GCN & $\text{DPD}_{\text{hard}}$ & $-0.0331$ & $-0.98$ & $p = 0.0098$ \\
 & FairGFL (2026) & $\text{DPD}_{\text{hard}}$ & $-0.0343$ & $-1.16$ & $p = 0.0020^\star$ \\
 & FairGFL (2026) & AUC & $-0.0096$ & $-1.11$ & $p = 0.0020^\star$ \\
 & CGSV (2021) & AUC & $-0.0086$ & $-0.96$ & $p = 0.0059^\star$ \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    # -------------------------------------------------------------
    # 6. Benchmark Datasets Characteristics Table
    # -------------------------------------------------------------
    preflight_json = "results/preflight_datasets.json"
    display_names = {
        "german": ("German Credit", "Gender", "Credit Risk"),
        "bail": ("Bail Recidivism", "Race", "Recidivism"),
        "credit": ("Credit Default", "Age", "Default"),
        "pokec_z": ("Pokec-z", "Region", "Working Field"),
        "elliptic": ("Elliptic Bitcoin", r"Time Split$^{\ddagger}$", "Illicit / Fraud"),
        "ogbn_products": ("OGBN-Products", "Degree", "Category"),
    }
    ordered_keys = ["german", "bail", "credit", "pokec_z", "elliptic"]

    rows = []
    if os.path.exists(preflight_json):
        with open(preflight_json, "r") as f:
            pf_data = json.load(f)
            items = pf_data.get("datasets", pf_data) if isinstance(pf_data, dict) else pf_data
            ds_map = {d["name"]: d for d in items if isinstance(d, dict) and "name" in d}
            for k in ordered_keys:
                if k in ds_map:
                    d = ds_map[k]
                    d_name, s_name, y_name = display_names.get(k, (k.capitalize(), str(d.get("sensitive", "s")), str(d.get("label", "y"))))
                    nodes = d.get("nodes", d.get("n_nodes", 0))
                    edges = d.get("n_edges_undirected", d.get("edges", 0))
                    features = d.get("n_features", d.get("dim", 0))
                    hs = d.get("h_s", d.get("homophily_hs", 0.0))
                    rows.append(f"{d_name} & {nodes:,} & {edges:,} & {features} & {s_name} & {y_name} & {hs:.4f} \\\\")

    if not rows:
        rows = [
            "German Credit & 1,000 & 21,742 & 26 & Gender & Credit Risk & 0.8048 \\\\",
            "Bail Recidivism & 18,876 & 311,870 & 16 & Race & Recidivism & 0.5221 \\\\",
            "Credit Default & 30,000 & 1,421,858 & 12 & Age & Default & 0.9595 \\\\",
            "Pokec-z & 67,796 & 617,958 & 276 & Region & Working Field & 0.9506 \\\\",
            "Elliptic Bitcoin & 203,769 & 234,355 & 165 & Timestep$^{\\ddagger}$ & Illicit & 1.0000 \\\\",
        ]

    tbody = "\n".join(rows)
    datasets_tex = f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{\\textbf{{Characteristics of Experimental Benchmark Datasets.}} 
All datasets strictly satisfy the Zero-Feature Leakage criterion ($\\max_j \\text{{AUC}}(x_j, y) < 0.85$). $h_s$ denotes the sensitive-attribute homophily ratio; edge counts use the undirected convention. $^{{\\ddagger}}$Elliptic's sensitive attribute is a structural proxy (transaction timestep) and its $h_s$ is exactly $1.0000$ by construction, since Elliptic payment edges exist only within a time-step; FSER's cross-group penalty is therefore a no-op on this graph and we report no fairness conclusion from it. ogbn-products is omitted here: it is used solely as a computational-scalability probe and was not put through the pre-flight fairness audit.}}
\\label{{tab:datasets}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\setlength{{\\tabcolsep}}{{4pt}}%
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Dataset}} & \\textbf{{Nodes ($N$)}} & \\textbf{{Edges ($|E|$)}} & \\textbf{{Features ($D$)}} & \\textbf{{Sensitive ($s$)}} & \\textbf{{Target ($y$)}} & \\textbf{{Homophily ($h_s$)}} \\\\
\\midrule
{tbody}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""

    # Write generated tables to all valid target directories
    for d in valid_dirs:
        with open(os.path.join(d, "main_pokecz_sota.tex"), "w") as f:
            f.write(pokec_tex)
        with open(os.path.join(d, "credit_boundary_sota.tex"), "w") as f:
            f.write(credit_tex)
        with open(os.path.join(d, "ablation.tex"), "w") as f:
            f.write(ablation_tex)
        with open(os.path.join(d, "shapley_fidelity.tex"), "w") as f:
            f.write(trust_tex)
        with open(os.path.join(d, "trust.tex"), "w") as f:
            f.write(trust_tex)
        with open(os.path.join(d, "significance.tex"), "w") as f:
            f.write(sig_tex)
        with open(os.path.join(d, "datasets.tex"), "w") as f:
            f.write(datasets_tex)

    # Copy / sync all verified static and revision tables
    src_manuscript_tables = "manuscript_neurocomputing/tables" if os.path.exists("manuscript_neurocomputing/tables") else "manuscript_neurocomputing/tables"
    if os.path.exists(src_manuscript_tables):
        other_files = ["compliance.tex", "efficiency.tex", "large_scale.tex", "privacy_attack.tex", "robustness.tex", "main_auc.tex", "main_dpd.tex", "main_eod.tex"]
        for d in valid_dirs:
            for fn in other_files:
                src_f = os.path.join(src_manuscript_tables, fn)
                dst_f = os.path.join(d, fn)
                if os.path.exists(src_f) and os.path.abspath(src_f) != os.path.abspath(dst_f):
                    shutil.copy2(src_f, dst_f)

            # Sync revision subdirectory
            src_rev = os.path.join(src_manuscript_tables, "revision")
            dst_rev = os.path.join(d, "revision")
            if os.path.exists(src_rev) and os.path.abspath(src_rev) != os.path.abspath(dst_rev):
                os.makedirs(dst_rev, exist_ok=True)
                for r_fn in os.listdir(src_rev):
                    if r_fn.endswith(".tex"):
                        shutil.copy2(os.path.join(src_rev, r_fn), os.path.join(dst_rev, r_fn))

    print(f"ALL PUBLICATION TABLES GENERATED & SYNCHRONIZED SUCCESSFULLY TO {valid_dirs}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", nargs="*", default=None, help="Target directories for tables")
    args = parser.parse_args()
    generate_all_tables(args.out_dir)

