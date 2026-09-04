"""BFWA Disparity Constraint Slack Analysis under DP noise (Issue I4, Stanford Q4-Q6).

Evaluates the empirical slack between the DP-noised disparity reported by clients
and the true underlying disparity under Frank-Wolfe aggregation:
    Slack_t = | sum_k w_k * DPD_hat_k - sum_k w_k * DPD_true_k |
Relative to the fairness budget tau = 0.05.

Evaluates on Bail and Credit across eps in {2.0, 4.0, 8.0} over 5 seeds.

Outputs:
  - results/revision/bfwa_slack.json
  - manuscript/tables/revision/bfwa_slack.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import _soft_dpd


def analyze_bfwa_slack(dataset="bail", seeds=(42, 43, 44), epsilons=(2.0, 4.0, 8.0),
                       rounds=25, num_clients=10, tau=0.05):
    summary_by_eps = {}

    for eps in epsilons:
        slacks_all_rounds = []
        true_dpd_aggregates = []
        noisy_dpd_aggregates = []

        for s in seeds:
            cfg = ExperimentConfig.canonical(
                dataset=dataset, seed=s, rounds=rounds, num_clients=num_clients,
                model="trustfedgnn", aggregator="bfwa", fairness_budget=tau,
                local_fairness=True, dp_enabled=True, dp_epsilon=eps, dp_delta=1e-5
            )
            trainer = FederatedTrainer(cfg)

            for t in range(rounds):
                # Run round
                rec = trainer._round(t)
                w = rec.get("agg_weights")
                if w is None or len(w) != num_clients:
                    w = [1.0 / num_clients] * num_clients
                w = np.array(w)

                # Compute true and reported DPD for each client
                dpd_true_list = []
                dpd_noisy_list = []

                for c in trainer.clients:
                    c.model.eval()
                    d = c.data
                    with torch.no_grad():
                        out = c.model(d.x, d.edge_index, d.sensitive_attr)[d.train_mask]
                        true_val = float(_soft_dpd(out, d.sensitive_attr[d.train_mask]).item())
                    dpd_true_list.append(true_val)
                    # Noisy value reported by client with DP noise
                    # Noise on scalar mean difference
                    noise_mag = c.dp_sigma * math.sqrt(2.0 / max(1, len(d.train_mask)))
                    noisy_val = max(0.0, true_val + np.random.normal(0, noise_mag))
                    dpd_noisy_list.append(noisy_val)

                dpd_true_arr = np.array(dpd_true_list)
                dpd_noisy_arr = np.array(dpd_noisy_list)

                agg_true = float(np.sum(w * dpd_true_arr))
                agg_noisy = float(np.sum(w * dpd_noisy_arr))
                slack = abs(agg_noisy - agg_true)

                slacks_all_rounds.append(slack)
                true_dpd_aggregates.append(agg_true)
                noisy_dpd_aggregates.append(agg_noisy)

        mean_slack = float(np.mean(slacks_all_rounds))
        std_slack = float(np.std(slacks_all_rounds))
        slack_ratio = mean_slack / tau  # fraction of tau

        summary_by_eps[str(eps)] = {
            "mean_slack": mean_slack,
            "std_slack": std_slack,
            "slack_ratio_of_tau": slack_ratio,
            "mean_true_dpd_agg": float(np.mean(true_dpd_aggregates)),
            "mean_noisy_dpd_agg": float(np.mean(noisy_dpd_aggregates)),
            "tau": tau,
        }
        print(f"[*] eps={eps} -> Slack = {mean_slack:.4f} +/- {std_slack:.4f} ({slack_ratio * 100:.1f}% of tau={tau})", flush=True)

    return summary_by_eps



# --------------------------------------------------------------------------- #
# Caption text, derived from the measurement
# --------------------------------------------------------------------------- #
# The caption used to assert "a negligible slack of $<0.003$ ($<6\%$ of $\tau$)"
# as a fixed string, printed verbatim whatever the run produced -- i.e. a
# conclusion written before the experiment. Everything below is computed from
# the results dict instead. Cut-offs, stated so a reader can disagree with them
# rather than having to reverse-engineer them:
#
#   slack / tau  <  10%  -> "negligible"  (well inside the budget's own slack)
#              <  50%  -> "moderate"    (eats a noticeable share of the budget)
#              >= 50%  -> "substantial" (the reported constraint is no longer
#                                        a reliable proxy for the true one)
SLACK_BUCKETS = ((10.0, "negligible"), (50.0, "moderate"), (float("inf"), "substantial"))


def slack_bucket(pct_of_tau: float) -> str:
    """Qualitative label for a measured slack, as a percentage of tau."""
    for hi, name in SLACK_BUCKETS:
        if pct_of_tau < hi:
            return name
    return "substantial"                                  # pragma: no cover


def _slack_caption_sentence(results: dict, tau: float) -> str:
    """Sentence describing the slack at the *largest* epsilon tested.

    The largest epsilon is the weakest-privacy / lowest-noise setting, i.e. the
    deployed operating point and the most favourable case for the method; if
    the slack is not negligible there it is not negligible anywhere.
    """
    if not results:
        return "No slack measurements were produced by this run."
    eps_key = max(results, key=lambda k: float(k))
    v = results[eps_key]
    pct = 100.0 * v["slack_ratio_of_tau"]
    word = slack_bucket(pct)
    if word == "negligible":
        implication = ("so FTGD privacy does not destabilise the BFWA dual "
                       "iteration at this operating point")
    elif word == "moderate":
        implication = ("so the reported constraint value carries a non-trivial "
                       "DP-induced error that the budget $\\tau$ must absorb")
    else:
        implication = ("so at this noise level the reported disparity is no longer "
                       "a reliable stand-in for the true one, and $\\tau$ must be "
                       "tightened (or the privacy budget loosened) for the "
                       "constraint to mean what it says")
    return (f"At the weakest-privacy point tested ($\\epsilon = {eps_key}$), DP noise "
            f"induces a {word} slack of ${v['mean_slack']:.4f} \\pm "
            f"{v['std_slack']:.4f}$ (${pct:.1f}\\%$ of $\\tau = {tau}$), "
            f"{implication}.")


def run_bfwa_slack_experiment(out_json="results/revision/bfwa_slack.json",
                              out_tex="manuscript/tables/revision/bfwa_slack.tex",
                              dataset="bail", seeds=(42, 43),
                              epsilons=(2.0, 4.0, 8.0), rounds=20,
                              num_clients=10, tau=0.05):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    print(f"[*] Running BFWA DP-induced disparity slack analysis on {dataset}...", flush=True)
    results = analyze_bfwa_slack(dataset=dataset, seeds=seeds, epsilons=epsilons,
                                 rounds=rounds, num_clients=num_clients, tau=tau)
    for v in results.values():
        v["slack_bucket"] = slack_bucket(100.0 * v["slack_ratio_of_tau"])

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved BFWA slack JSON to {out_json}")

    # Generate LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{BFWA Disparity Constraint Slack Induced by Differential Privacy Noise.}",
        "Empirical slack between reported noisy disparity and true underlying disparity "
        "$\\Delta_{\\tau} = |\\sum_k w_k \\widehat{\\text{DPD}}_k - \\sum_k w_k \\text{DPD}_k|$ "
        f"under budget $\\tau = {tau}$ ({dataset.capitalize()}, $n={num_clients}$ clients, "
        f"{rounds} rounds, {len(seeds)} seed" + ("s" if len(seeds) != 1 else "") + ").",
        _slack_caption_sentence(results, tau) + "}",
        "\\label{tab:bfwa_slack}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{DP Target $\\epsilon$} & \\textbf{Reported $\\sum w_k \\widehat{\\text{DPD}}_k$} & \\textbf{True $\\sum w_k \\text{DPD}_k$} & \\textbf{Slack $|\\Delta_{\\tau}|$} & \\textbf{Slack / $\\tau$ (\\%)} \\\\",
        "\\midrule",
    ]

    for eps_str, v in results.items():
        line = (
            f"$\\epsilon = {eps_str}$ & {v['mean_noisy_dpd_agg']:.4f} & {v['mean_true_dpd_agg']:.4f} & "
            f"{v['mean_slack']:.4f} $\\pm$ {v['std_slack']:.4f} & {v['slack_ratio_of_tau'] * 100:.1f}\\% \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX BFWA slack table to {out_tex}")


def main():
    ap = argparse.ArgumentParser(description="BFWA DP-induced constraint slack.")
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--epsilons", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out-json", default="results/revision/bfwa_slack.json")
    ap.add_argument("--out-tex", default="manuscript/tables/revision/bfwa_slack.tex")
    a = ap.parse_args()
    run_bfwa_slack_experiment(out_json=a.out_json, out_tex=a.out_tex,
                              dataset=a.dataset, seeds=tuple(a.seeds),
                              epsilons=tuple(a.epsilons), rounds=a.rounds,
                              num_clients=a.num_clients, tau=a.tau)


if __name__ == "__main__":
    main()
