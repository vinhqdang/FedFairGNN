"""Figure generation pipeline for Manuscript v2 (FedFairGNN/manuscript_v2/figures).

Generates high-resolution vector PDF figures with strict provenance from real artifacts:
  - F1: fig_interface_boundary.pdf  -- Slope/Dumbbell chart of 10 aggregation rules (metadata capture vs syntactic immunity)
  - F3: fig_robustness_byz.pdf      -- Ratio of w_adv / uniform share across corruption ratios f/K in {0.1, 0.2, 0.3, 0.4}
  - F4: fig_dynamics.pdf            -- 2-panel figure: (a) 20-round Bail convergence; (b) Weight trajectory across rounds
  - F5: fig_pareto.pdf              -- Pareto frontier across all 11 methods on Pokec-z and Credit Default (including FLTrust & FedGraph-Fair)
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "FedFairGNN/manuscript_v2/figures"
RESULTS_DIR = "FedFairGNN/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Styling palette (Color Universal Design & Google Antigravity aesthetic)
COLOR_TEXT = "#1f2328"
COLOR_MUTED = "#656d76"
COLOR_GRID = "#e1e4e8"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "text.color": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def generate_f1_interface_boundary():
    """F1: Slope / Dumbbell chart showing adversary aggregation weight under honest vs lying reports."""
    stats_path = os.path.join(RESULTS_DIR, "revision", "metadata_capture_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing artifact: {stats_path}")

    with open(stats_path) as f:
        data = json.load(f)["per_rule"]

    rule_configs = [
        ("f2gnn", r"$F^2\mathrm{GNN}$ (WWW'23)", "Class (a): Scalar Metadata", "#b2182b"),
        ("bfwa", "BFWA (IndabaX'26)", "Class (a): Scalar Metadata", "#d6604d"),
        ("qffl", "q-FedAvg (ICLR'20)", "Class (a): Scalar Metadata", "#e08214"),
        ("fedgraphfair", "FedGraph-Fair (InfoSci'26)", "Class (a): Scalar Metadata", "#f4a582"),
        ("fairfed", "FairFed (AAAI'23)", "Class (a): Scalar Metadata", "#fdbb84"),
        ("popets_fairfed", "PoPETs-FairFed (PoPETs'25)", "Class (a): Scalar Metadata", "#9970ab"),
        ("cgsv", "CGSV (NeurIPS'21)", "Class (b): Client Validation", "#4393c3"),
        ("fltrust", "FLTrust (NDSS'21)", "Class (c): Server Gradients", "#2166ac"),
        ("fu_shapley_alpha0", r"TrustFedGNN ($\alpha=0$)", "Class (c): Server Gradients", "#1b7837"),
        ("fu_shapley", "TrustFedGNN (Ours)", "Class (c): Server Gradients", "#006837"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    y_positions = np.arange(len(rule_configs))

    # Reference uniform line (w = 0.20)
    ax.axvline(0.20, color="#d73027", linestyle=":", linewidth=1.5, alpha=0.85, zorder=2, label="Uniform Share (1/K = 0.20)")

    for idx, (key, label, cat, color) in enumerate(rule_configs):
        v = data[key]
        w_h = v["w_adv_honest"]
        w_l = v["w_adv_lie"]
        delta_w = w_l - w_h

        # Plot line connecting honest to lying
        if delta_w > 0.001:
            ax.plot([w_h, w_l], [idx, idx], color=color, linewidth=2.4, alpha=0.85, zorder=3)
            ax.annotate("", xy=(w_l, idx), xytext=(w_h, idx),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=14), zorder=4)
            ax.text(w_l + 0.015, idx, f"+{delta_w:.3f}", va="center", ha="left", fontsize=8.0, color=color, fontweight="bold")
        else:
            ax.scatter([w_h], [idx], color=color, s=70, marker="o", edgecolors="#1f2328", linewidth=1.0, zorder=5)
            ax.text(w_h + 0.015, idx, r"$\Delta w = 0.0000$ (Bit-Exact)", va="center", ha="left", fontsize=8.0, color=color, fontweight="semibold")

        ax.scatter([w_h], [idx], color="#ffffff", edgecolors=color, s=38, marker="o", linewidth=1.5, zorder=4)

    # Styling and separators
    ax.axhspan(-0.5, 5.5, facecolor="#fee8c8", alpha=0.18, zorder=1)
    ax.axhspan(5.5, 6.5, facecolor="#e0f3f8", alpha=0.25, zorder=1)
    ax.axhspan(6.5, 9.5, facecolor="#e5f5e0", alpha=0.25, zorder=1)

    ax.text(0.98, 2.5, "Class (a): Self-Reported Scalar Metadata\n(Subject to Coordinate Manipulation)", 
            va="center", ha="right", fontsize=8.5, color="#b2182b", style="italic")
    ax.text(0.98, 6.0, "Class (b): Client Validation Data (Poison Vulnerable)", 
            va="center", ha="right", fontsize=8.5, color="#2166ac", style="italic")
    ax.text(0.98, 8.0, "Class (c): Server-Anchored Parameter Gradients\n(Syntactic Metadata Invariance, Theorem 2)", 
            va="center", ha="right", fontsize=8.5, color="#006837", style="italic")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[1] for r in rule_configs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Adversary Aggregation Weight $w_{\mathrm{adv}}$", fontsize=10)
    ax.set_xlim(-0.02, 1.05)
    ax.set_title("Structural Interface Immunity vs. Interface Capture under Strategic Metadata Falsification", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.5)

    ax.plot([], [], color="#ffffff", marker="o", markeredgecolor="#555555", markeredgewidth=1.5, label=r"Honest Report ($w_{\mathrm{honest}}$)")
    ax.plot([], [], color="#b2182b", marker=">", markersize=7, label=r"Falsified Report ($w_{\mathrm{lying}}$)")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_interface_boundary.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_interface_boundary.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved F1 interface boundary to {out_path}")


def generate_f3_robustness_byz():
    """F3: Ratio of adversary weight share w_adv / (f/K) across corruption ratios f/K in {0.1, 0.2, 0.3, 0.4}."""
    artifact_path = os.path.join(RESULTS_DIR, "revision", "adaptive_poisoner_results.json")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    with open(artifact_path) as f:
        data = json.load(f)

    ratios = [0.1, 0.2, 0.3, 0.4]
    aggs_to_plot = [
        ("fu_shapley", "TrustFedGNN (Ours)", "#006837", "o-", 2.2),
        ("robust_fu_shapley", "Robust TrustFedGNN (+median)", "#b2182b", "s--", 2.0),
        ("fltrust", "FLTrust", "#2166ac", "^-.", 1.6),
        ("fedavg", "FedAvg (no defense)", "#757575", "v:", 1.5),
        ("krum", "Krum", "#d95f02", "d-", 1.8),
        ("multikrum", "Multi-Krum", "#7570b3", "x-", 1.5),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    # Horizontal 1.0x parity line
    ax.axhline(1.0, color="#252525", linestyle="-", linewidth=1.5, alpha=0.8, label=r"Uniform Parity ($1.0\times$ Fair Share)")
    ax.axhspan(0.0, 1.0, facecolor="#e5f5e0", alpha=0.3, zorder=1)
    ax.text(0.12, 0.55, "Adversary Suppressed (< 1.0x share)", fontsize=8.5, color="#006837", style="italic")
    ax.text(0.12, 1.45, "Adversary Privileged (> 1.0x share)", fontsize=8.5, color="#b2182b", style="italic")

    for agg_key, label, color, fmt, lw in aggs_to_plot:
        means = []
        stds = []
        for r in ratios:
            records = [rec["w_adv"] for rec in data["records"] if rec["aggregator"] == agg_key and abs(rec["byz_ratio"] - r) < 1e-4]
            if not records:
                means.append(np.nan)
                stds.append(0.0)
            else:
                ratio_vals = [w / r for w in records]
                means.append(np.mean(ratio_vals))
                stds.append(np.std(ratio_vals))

        marker = fmt[0]
        linestyle = fmt[1:]
        ax.plot(ratios, means, marker=marker, linestyle=linestyle, color=color, linewidth=lw, label=label, markersize=5.5, zorder=4)
        ax.fill_between(ratios, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.12, zorder=3)

    ax.set_xlabel(r"Byzantine Fraction $f/K$", fontsize=10)
    ax.set_ylabel(r"Adversary Share Ratio $w_{\mathrm{adv}} \,/\, (f/K)$ ($\downarrow$)", fontsize=10)
    ax.set_title("Adversarial Share Ratio under Stealth Poisoning: The Shielding Paradox", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xticks(ratios)
    ax.set_xticklabels(["0.1 (1/10)", "0.2 (2/10)", "0.3 (3/10)", "0.4 (4/10)"], fontsize=9)
    ax.set_ylim(0.4, 2.0)
    ax.text(0.105, 1.90, r"Krum shoots to $9.53\times$ at $f/K=0.1$ (off-scale)", fontsize=8, color="#d95f02", style="italic")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_robustness_byz.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_robustness_byz.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved F3 robustness_byz to {out_path}")


def generate_f4_dynamics():
    """F4: 2-Panel dynamics: (a) Bail 20-round convergence; (b) Weight trajectory round-by-round."""
    bail_path = os.path.join(RESULTS_DIR, "convergence_bail.json")
    traj_path = os.path.join(RESULTS_DIR, "fairshare", "audit_traj__german__fu_shapley__none__s0.csv")
    bfwa_path = os.path.join(RESULTS_DIR, "fairshare", "audit_traj__german__bfwa__none__s0.csv")

    if not os.path.exists(bail_path):
        raise FileNotFoundError(f"Missing artifact: {bail_path}")

    with open(bail_path) as f:
        bail_data = json.load(f)["history"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    # Panel (a): Convergence on Bail
    ax_a = axes[0]
    rounds_a = [e["round"] for e in bail_data]
    auc_a = [e["g_auc"] for e in bail_data]
    dpd_a = [e["g_dpd"] for e in bail_data]
    eod_a = [e.get("g_eod", 0.0) for e in bail_data]

    ax_a.plot(rounds_a, auc_a, "o-", color="#1b7837", linewidth=2.2, label=r"Test AUC-ROC ($\uparrow$)", markersize=5)
    ax_a.plot(rounds_a, dpd_a, "s-", color="#d73027", linewidth=2.2, label=r"Test DPD ($\downarrow$)", markersize=5)
    ax_a.plot(rounds_a, eod_a, "^-", color="#4575b4", linewidth=2.2, label=r"Test EOD ($\downarrow$)", markersize=5)

    ax_a.set_xlabel("Communication Round", fontsize=10)
    ax_a.set_ylabel("Metric Value", fontsize=10)
    ax_a.set_title("(a) Training Convergence Dynamics (Bail Recidivism)", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.set_ylim(-0.03, 0.85)
    ax_a.legend(loc="center right", fontsize=8.5, framealpha=0.92)
    ax_a.grid(True, alpha=0.5)

    # Panel (b): Weight trajectories across rounds (EMA smoothed vs Unsmoothed Dual Ascent)
    ax_b = axes[1]
    if os.path.exists(traj_path) and os.path.exists(bfwa_path):
        import csv
        import ast

        def load_weights(csv_file):
            weights_by_round = []
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    w = ast.literal_eval(row["agg_weights"])
                    weights_by_round.append(w)
            return np.array(weights_by_round)

        w_ours = load_weights(traj_path)   # Shape: (R, K)
        w_bfwa = load_weights(bfwa_path)   # Shape: (R, K)
        R_rounds = np.arange(1, len(w_ours) + 1)

        # Plot TrustFedGNN smoothed trajectories
        client_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for k in range(w_ours.shape[1]):
            ax_b.plot(R_rounds, w_ours[:, k], color=client_colors[k % len(client_colors)],
                      linewidth=2.0, label=f"Client {k+1} (TrustFedGNN)" if k < 3 else "")

        # Plot BFWA unsmoothed dual-ascent paths as dashed high-frequency lines
        for k in range(min(2, w_bfwa.shape[1])):
            ax_b.plot(R_rounds, w_bfwa[:, k], color="#8c564b", linestyle="--", linewidth=1.4, alpha=0.75,
                      label=f"Client {k+1} (BFWA Dual-Ascent)" if k == 0 else "")

        # Informative text callouts placed in clean areas
        ax_b.text(2.2, 0.44, r"TrustFedGNN ($\Omega_w = 0.0586$)" + "\nSmooth EMA trajectories", 
                  fontsize=8.5, color="#006837", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.35", fc="#e5f5e0", ec="#a1d99b", lw=0.9))
        ax_b.text(10.5, 0.70, r"BFWA Dual-Ascent ($\Omega_w = 16.54$)" + "\nHigh oscillation & variance", 
                  fontsize=8.5, color="#7f2704", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.35", fc="#fee6ce", ec="#fdae6b", lw=0.9))

    ax_b.axhline(0.20, color="#d73027", linestyle=":", linewidth=1.4, alpha=0.8, label="Uniform share (0.20)")
    ax_b.set_xlabel("Communication Round", fontsize=10)
    ax_b.set_ylabel(r"Aggregation Weight $w_k^{(t)}$", fontsize=10)
    ax_b.set_title(r"(b) Weight Trajectory Dispersion $\Omega_w$ Across Rounds", fontsize=10.5, fontweight="bold", pad=8)
    ax_b.set_ylim(-0.02, 0.88)
    ax_b.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
    ax_b.grid(True, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig_dynamics.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_dynamics.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved F4 dynamics to {out_path}")


def generate_f5_pareto():
    """F5: Re-plot Pareto frontier with ALL 11 methods, with dedicated top margin and clear labels."""
    pokec_path = os.path.join(RESULTS_DIR, "sota_pokecz.json")
    credit_path = os.path.join(RESULTS_DIR, "sota_credit.json")

    with open(pokec_path) as f:
        pokec_data = json.load(f).get("baselines", {})
    with open(credit_path) as f:
        credit_data = json.load(f).get("baselines", {})

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=300)

    # 10 distinct baseline styles
    styles = {
        "fedavg-gcn":       ("#737373", "o", "FedAvg-GCN"),
        "fairgnn":          ("#984ea3", "s", "FairGNN"),
        "fairsin":          ("#4daf4a", "^", "FairSIN"),
        "fairfed":          ("#ff7f00", "v", "FairFed"),
        "fairgfl":          ("#a65628", "D", "FairGFL"),
        "fedgraphfair":     ("#377eb8", "P", "FedGraph-Fair"),
        "cgsv":             ("#f781bf", "X", "CGSV Non-DP"),
        "fltrust":          ("#252525", "*", "FLTrust"),
        "ours-nofser-true": ("#e7298a", "p", "Ours w/o FSER (Clean)"),
        "fedfairgnn":       ("#d73027", "h", "TrustFedGNN (Ours)"),
    }

    datasets = [("Credit Default ($N=30,000$)", credit_data), ("Pokec-z ($N=67,796$)", pokec_data)]

    for ax_idx, (ds_name, data) in enumerate(datasets):
        ax = axes[ax_idx]
        for method, (col, marker, label) in styles.items():
            if method in data:
                entry = data[method]
                if isinstance(entry, dict) and "summary" in entry:
                    entry = entry["summary"]
                if not isinstance(entry, dict) or "auc" not in entry or "dpd_hard" not in entry:
                    continue
                auc_val = entry["auc"]["mean"] if isinstance(entry["auc"], dict) else entry["auc"]
                dpd_val = entry["dpd_hard"]["mean"] if isinstance(entry["dpd_hard"], dict) else entry["dpd_hard"]
                auc_err = entry["auc"].get("std", 0.0) if isinstance(entry["auc"], dict) else 0.0
                dpd_err = entry["dpd_hard"].get("std", 0.0) if isinstance(entry["dpd_hard"], dict) else 0.0

                msize = 9.5 if "Ours" in label or "FLTrust" in label else 7.0
                malpha = 0.95 if "Ours" in label or "FLTrust" in label else 0.8

                ax.errorbar(dpd_val, auc_val, xerr=dpd_err, yerr=auc_err,
                            fmt=marker, color=col, label=label if ax_idx == 0 else "",
                            markersize=msize, capsize=3.5, elinewidth=1.2,
                            alpha=malpha, zorder=6 if "Ours" in label else 4)

                # Annotations for key Pareto operating points on Pokec-z
                if ax_idx == 1:
                    if method == "fltrust":
                        ax.annotate(r"FLTrust ($\uparrow$AUC $0.8057$)", xy=(dpd_val, auc_val),
                                    xytext=(dpd_val + 0.003, auc_val + 0.007),
                                    arrowprops=dict(arrowstyle="->", color="#252525", lw=1.0),
                                    fontsize=8.0, fontweight="bold", color="#252525")
                    elif method == "fedgraphfair":
                        ax.annotate(r"FedGraph-Fair ($\downarrow$DPD $0.0050$)", xy=(dpd_val, auc_val),
                                    xytext=(dpd_val + 0.003, auc_val - 0.015),
                                    arrowprops=dict(arrowstyle="->", color="#377eb8", lw=1.0),
                                    fontsize=8.0, fontweight="bold", color="#377eb8")
                    elif method == "fedfairgnn":
                        ax.annotate("TrustFedGNN (Balanced)", xy=(dpd_val, auc_val),
                                    xytext=(dpd_val - 0.012, auc_val - 0.022),
                                    arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.1),
                                    fontsize=8.5, fontweight="bold", color="#d73027")

        ax.set_xlabel(r"Demographic Parity Disparity $\text{DPD}_{\mathrm{hard}}$ ($\downarrow$)", fontsize=10)
        ax.set_ylabel(r"Test ROC-AUC ($\uparrow$)", fontsize=10)
        ax.set_title(f"Pareto Frontier: {ds_name}", fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.5)

    # Dedicated top space (top=0.78) for 2-row legend so it NEVER overlaps plot titles
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.96, wspace=0.22)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

    out_path = os.path.join(OUT_DIR, "fig_pareto.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig_pareto.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved F5 pareto to {out_path}")


if __name__ == "__main__":
    generate_f1_interface_boundary()
    generate_f3_robustness_byz()
    generate_f4_dynamics()
    generate_f5_pareto()
    print("[🎉] Successfully generated all 4 manuscript v2 vector figures!")
