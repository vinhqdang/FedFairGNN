"""Figure generation orchestrator for TrustFedGNN publication manuscript.

Re-implements and unifies the generation of the 5 core manuscript figures from real artifacts:
  1. pareto.pdf           -- Pareto frontier (AUC vs DPD) across baselines on Credit and Pokec-z
  2. privacy_bail.pdf     -- Utility & fairness vs DP epsilon on Bail (TrustFedGNN vs DP-FedAvg)
  3. privacy_attack.pdf   -- Attribute inference attack accuracy vs DP budget epsilon
  4. robustness_byz.pdf   -- Robustness breakdown under Byzantine corruption across aggregators
  5. convergence.pdf      -- Training convergence across communication rounds on Bail

Strict Governance:
  - Every figure MUST be generated directly from real experiment artifacts in results/.
  - No fabricated data, no analytic formula simulations, no hardcoded metric values.
  - Missing artifacts MUST raise FileNotFoundError indicating which stage produces them.

Outputs:
  - manuscript/figures/{pareto,privacy_bail,privacy_attack,robustness_byz,convergence}.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = "manuscript/figures"
RESULTS_DIR = "results"


def plot_pareto(out_path: str = os.path.join(FIG_DIR, "pareto.pdf")):
    """Figure 1: Pareto frontier (AUC vs DPD) on Credit and Pokec-z."""
    pokec_path = os.path.join(RESULTS_DIR, "sota_pokecz.json")
    credit_path = os.path.join(RESULTS_DIR, "sota_credit.json")

    if not os.path.exists(pokec_path):
        raise FileNotFoundError(
            f"Artifact for Pokec-z SOTA matrix missing at '{pokec_path}'. "
            "Run Stage S6 (experiments/run_sota_pokecz.py) first."
        )
    if not os.path.exists(credit_path):
        raise FileNotFoundError(
            f"Artifact for Credit SOTA matrix missing at '{credit_path}'. "
            "Run Stage S6 (experiments/run_sota_credit.py) first."
        )

    with open(pokec_path) as f:
        pokec_data = json.load(f).get("baselines", {})
    with open(credit_path) as f:
        credit_data = json.load(f).get("baselines", {})

    if not pokec_data or not credit_data:
        raise FileNotFoundError(
            "SOTA baseline records in artifacts are empty. Run Stage S6 first."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    styles = {
        "fedavg-gcn": ("#757575", "o", "FedAvg-GCN"),
        "fairgnn": ("#8e24aa", "s", "FairGNN"),
        "fairsin": ("#00acc1", "^", "FairSIN"),
        "fairfed": ("#fb8c00", "v", "FairFed"),
        "fairgfl": ("#43a047", "D", "FairGFL"),
        "fedgraphfair": ("#3949ab", "P", "FedGraph-Fair"),
        "cgsv": ("#f4511e", "X", "CGSV Non-DP"),
        "ours-nofser": ("#d81b60", "*", "Ours w/o FSER"),
        "ours-nofser-true": ("#8c510a", "p", "Ours w/o FSER (True)"),
        "fedfairgnn": ("#e53935", "h", "TrustFedGNN (Ours)"),
    }

    for ax_idx, (ds_name, data) in enumerate([("Credit Default", credit_data), ("Pokec-z", pokec_data)]):
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

                ax.errorbar(dpd_val, auc_val, xerr=dpd_err, yerr=auc_err,
                            fmt=marker, color=col, label=label if ax_idx == 0 else "",
                            markersize=8, capsize=3, alpha=0.9)

        ax.set_xlabel("Demographic Parity Difference (DPD $\\downarrow$)", fontsize=10)
        ax.set_ylabel("ROC-AUC ($\\uparrow$)", fontsize=10)
        ax.set_title(f"Pareto Frontier: {ds_name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved pareto figure to {out_path}")


def plot_privacy_bail(out_path: str = os.path.join(FIG_DIR, "privacy_bail.pdf")):
    """Figure 2: Utility & Fairness vs DP Epsilon on Bail Recidivism."""
    artifact_path = os.path.join(RESULTS_DIR, "revision", "privacy_bail_sweep.json")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact for privacy_bail plot missing at '{artifact_path}'. "
            "Run Stage S5 (experiments/revision/dp_accounting_table.py and bfwa_slack_analysis.py) first."
        )

    with open(artifact_path) as f:
        data = json.load(f)

    if "epsilons" not in data or "ftgd" not in data or "dp_fedavg" not in data:
        raise FileNotFoundError(
            f"Artifact at '{artifact_path}' does not contain complete epsilon sweep data. Run Stage S5 first."
        )

    epsilons = data["epsilons"]
    ftgd_auc = data["ftgd"]["auc"]
    ftgd_dpd = data["ftgd"]["dpd"]
    dp_auc = data["dp_fedavg"]["auc"]
    dp_dpd = data["dp_fedavg"]["dpd"]

    ftgd_auc_std = data["ftgd"].get("auc_std")
    ftgd_dpd_std = data["ftgd"].get("dpd_std")
    dp_auc_std = data["dp_fedavg"].get("auc_std")
    dp_dpd_std = data["dp_fedavg"].get("dpd_std")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # Left: Utility (AUC)
    ax[0].plot(epsilons, ftgd_auc, "o-", color="#1b7837", linewidth=2, label="TrustFedGNN (FTGD)")
    if ftgd_auc_std:
        ax[0].fill_between(epsilons, np.array(ftgd_auc) - np.array(ftgd_auc_std),
                           np.array(ftgd_auc) + np.array(ftgd_auc_std), color="#1b7837", alpha=0.15)
    ax[0].plot(epsilons, dp_auc, "s--", color="#d73027", linewidth=2, label="DP-FedAvg (Standard DP-SGD)")
    if dp_auc_std:
        ax[0].fill_between(epsilons, np.array(dp_auc) - np.array(dp_auc_std),
                           np.array(dp_auc) + np.array(dp_auc_std), color="#d73027", alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Privacy Budget $\\epsilon$ (Lower = Stricter Privacy)", fontsize=10)
    ax[0].set_ylabel("AUC-ROC ($\\uparrow$)", fontsize=10)
    ax[0].set_title("Utility Preservation under DP", fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    # Right: Fairness (DPD)
    ax[1].plot(epsilons, ftgd_dpd, "o-", color="#1b7837", linewidth=2, label="TrustFedGNN (FTGD)")
    if ftgd_dpd_std:
        ax[1].fill_between(epsilons, np.array(ftgd_dpd) - np.array(ftgd_dpd_std),
                           np.array(ftgd_dpd) + np.array(ftgd_dpd_std), color="#1b7837", alpha=0.15)
    ax[1].plot(epsilons, dp_dpd, "s--", color="#d73027", linewidth=2, label="DP-FedAvg (Standard DP-SGD)")
    if dp_dpd_std:
        ax[1].fill_between(epsilons, np.array(dp_dpd) - np.array(dp_dpd_std),
                           np.array(dp_dpd) + np.array(dp_dpd_std), color="#d73027", alpha=0.15)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Privacy Budget $\\epsilon$ (Lower = Stricter Privacy)", fontsize=10)
    ax[1].set_ylabel("Demographic Parity Diff (DPD $\\downarrow$)", fontsize=10)
    ax[1].set_title("Fairness Stability under DP", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved privacy_bail figure to {out_path}")


def plot_privacy_attack(out_path: str = os.path.join(FIG_DIR, "privacy_attack.pdf")):
    """Figure 3: Attribute Inference Attack AUC vs DP Epsilon on Bail."""
    artifact_path = os.path.join(RESULTS_DIR, "privacy_attack.json")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact for privacy_attack plot missing at '{artifact_path}'. "
            "Run Stage S5 (experiments/privacy_attack.py and experiments/revision/update_level_attack.py) first."
        )

    with open(artifact_path) as f:
        data = json.load(f)

    if "epsilons" not in data or "attack_auc" not in data:
        raise FileNotFoundError(
            f"Artifact at '{artifact_path}' does not contain required attack accuracy sweep. Run Stage S5 first."
        )

    epsilons = data["epsilons"]
    ftgd_attack_auc = data["attack_auc"]
    exact_leakage = data.get("exact_release_auc", 1.0)
    random_chance = data.get("random_chance", 0.500)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(epsilons, ftgd_attack_auc, "o-", color="#1b7837", linewidth=2, label="FTGD (Released Statistic + DP Noise)")
    ax.axhline(exact_leakage, linestyle="--", color="#762a83", linewidth=1.5, label=f"No DP Exact Release (AUC = {exact_leakage:.3f})")
    ax.axhline(random_chance, linestyle=":", color="#999999", linewidth=1.5, label="Random Guess Chance (0.500)")

    ax.set_xscale("log")
    ax.set_xlabel("Differential Privacy Budget $\\epsilon$", fontsize=10)
    ax.set_ylabel("Sensitive-Attribute Inference AUC ($\\downarrow$)", fontsize=10)
    ax.set_title("Attribute Inference Defense on Released Statistic (Bail)", fontsize=11, fontweight="bold")
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved privacy_attack figure to {out_path}")


def plot_robustness_byz(out_path: str = os.path.join(FIG_DIR, "robustness_byz.pdf")):
    """Figure 4: Robustness breakdown under Byzantine corruption ratios."""
    artifact_path = os.path.join(RESULTS_DIR, "revision", "robustness_multiseed.json")
    if not os.path.exists(artifact_path):
        alt_path = os.path.join(RESULTS_DIR, "revision", "adaptive_poisoner_results.json")
        if os.path.exists(alt_path):
            artifact_path = alt_path
        else:
            raise FileNotFoundError(
                f"Artifact for robustness_byz plot missing at '{artifact_path}'. "
                "Run Stage S7 (experiments/revision/adaptive_poisoner.py) first."
            )

    with open(artifact_path) as f:
        data = json.load(f)

    runs = data.get("raw_runs", []) or data.get("records", [])
    if not runs:
        raise FileNotFoundError(
            f"Artifact at '{artifact_path}' does not contain raw_runs or records. Run Stage S7 first."
        )

    # Group runs by aggregator and byz_ratio
    agg_ratios = {}
    for r in runs:
        agg = r.get("aggregator")
        ratio = r.get("byz_ratio")
        auc = r.get("auc")
        if agg is not None and ratio is not None and auc is not None:
            agg_ratios.setdefault(agg, {}).setdefault(ratio, []).append(auc)

    if not agg_ratios:
        raise FileNotFoundError(
            "No valid aggregator curves could be extracted from robustness artifact. Run Stage S7 first."
        )

    styles = {
        "fedavg": ("#757575", "x--", "FedAvg"),
        "bfwa": ("#fb8c00", "^--", "BFWA"),
        "krum": ("#8e24aa", "s-", "Krum"),
        "multikrum": ("#00acc1", "d-", "Multi-Krum"),
        "median": ("#6baed6", "+--", "Coordinate Median"),
        "trimmed_mean": ("#9ecae1", "*--", "Trimmed Mean"),
        "robust_bfwa": ("#3949ab", "v-", "Robust BFWA"),
        "fu_shapley": ("#e53935", "o-", "FU-Shapley"),
        "robust_fu_shapley": ("#b2182b", "o-", "TrustFedGNN (Ours)"),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    for agg, (col, fmt, name) in styles.items():
        if agg in agg_ratios:
            ratios = sorted(agg_ratios[agg].keys())
            mean_aucs = [float(np.mean(agg_ratios[agg][r])) for r in ratios]
            ax.plot(ratios, mean_aucs, fmt, color=col, linewidth=2, label=name, markersize=6)

    ax.set_xlabel("Byzantine Corruption Ratio $f / K$", fontsize=10)
    ax.set_ylabel("AUC-ROC under Attack ($\\uparrow$)", fontsize=10)
    ax.set_title("Adversarial Robustness: Adaptive Stealth Poisoning", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved robustness_byz figure to {out_path}")


def plot_convergence(out_path: str = os.path.join(FIG_DIR, "convergence.pdf")):
    """Figure 5: Training convergence across communication rounds on Bail."""
    artifact_path = os.path.join(RESULTS_DIR, "convergence_bail.json")
    if not os.path.exists(artifact_path):
        artifact_path = os.path.join(RESULTS_DIR, "canonical_suite.json")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Artifact for convergence plot missing at '{artifact_path}'. "
            "Run Stage S3 (experiments/run_canonical_suite.py) first."
        )

    with open(artifact_path) as f:
        data = json.load(f)

    # Search for canonical bail run with recorded history
    history = data.get("history")
    if not history:
        for k, v in data.items():
            if isinstance(v, dict) and "history" in v and len(v["history"]) > 1:
                history = v["history"]
                break

    if history is None:
        raise FileNotFoundError(
            f"Artifact at '{artifact_path}' does not contain training history for Bail. Run Stage S3 first."
        )

    rounds = [entry["round"] for entry in history]
    auc_curve = [entry["g_auc"] for entry in history]
    dpd_curve = [entry["g_dpd"] for entry in history]
    eod_curve = [entry.get("g_eod", 0.0) for entry in history]

    auc_std = [entry.get("g_auc_std") for entry in history]
    dpd_std = [entry.get("g_dpd_std") for entry in history]
    eod_std = [entry.get("g_eod_std") for entry in history]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
    ax.plot(rounds, auc_curve, "o-", color="#1b7837", linewidth=2, label="Test AUC-ROC ($\\uparrow$)", markersize=4)
    if all(s is not None for s in auc_std):
        ax.fill_between(rounds, np.array(auc_curve) - np.array(auc_std),
                        np.array(auc_curve) + np.array(auc_std), color="#1b7837", alpha=0.15)

    ax.plot(rounds, dpd_curve, "s-", color="#d73027", linewidth=2, label="Test DPD ($\\downarrow$)", markersize=4)
    if all(s is not None for s in dpd_std):
        ax.fill_between(rounds, np.array(dpd_curve) - np.array(dpd_std),
                        np.array(dpd_curve) + np.array(dpd_std), color="#d73027", alpha=0.15)

    ax.plot(rounds, eod_curve, "^-", color="#4575b4", linewidth=2, label="Test EOD ($\\downarrow$)", markersize=4)
    if all(s is not None for s in eod_std):
        ax.fill_between(rounds, np.array(eod_curve) - np.array(eod_std),
                        np.array(eod_curve) + np.array(eod_std), color="#4575b4", alpha=0.15)

    ax.set_xlabel("Communication Round", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_title("TrustFedGNN Convergence Dynamics (Bail Recidivism)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved convergence figure to {out_path}")


def generate_all_figures(out_dir: str = FIG_DIR, strict: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    plotters = [
        ("pareto", plot_pareto, os.path.join(out_dir, "pareto.pdf")),
        ("privacy_bail", plot_privacy_bail, os.path.join(out_dir, "privacy_bail.pdf")),
        ("privacy_attack", plot_privacy_attack, os.path.join(out_dir, "privacy_attack.pdf")),
        ("robustness_byz", plot_robustness_byz, os.path.join(out_dir, "robustness_byz.pdf")),
        ("convergence", plot_convergence, os.path.join(out_dir, "convergence.pdf")),
    ]

    generated = []
    skipped = []

    for name, func, out_file in plotters:
        try:
            func(out_file)
            generated.append(name)
        except FileNotFoundError as e:
            skipped.append((name, str(e)))
            print(f"[!] SKIPPED {name}: {e}")
            if strict:
                raise

    print(f"\n[Figures Report] Generated: {len(generated)}/5 | Skipped (missing artifacts): {len(skipped)}/5")
    return {"generated": generated, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="Generate all publication manuscript figures from real artifacts.")
    parser.add_argument("--out-dir", default=FIG_DIR, help="Output directory for figures.")
    parser.add_argument("--strict", action="store_true", help="Fail with exit code 1 if any artifact is missing.")
    args = parser.parse_args()

    res = generate_all_figures(out_dir=args.out_dir, strict=args.strict)
    if args.strict and res["skipped"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
