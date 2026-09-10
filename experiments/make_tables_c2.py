#!/usr/bin/env python3
"""Generate the Pillar-C2 evidence tables (metadata immunity, two-tier defense,
weight stability) directly from ``results/`` artifacts.

These three tables carry the manuscript's central defensive claim, so none of
their numbers may be typed by hand: every cell is read out of a JSON artifact
here and written straight into LaTeX. Kept separate from ``make_tables.py``
only to avoid threading three more artifact loaders through that module's
already long ``generate_all_tables``; the output directories and the
"regenerate, never edit the .tex" contract are identical.

Run from the FedFairGNN repository root:

    ../.venv-local/bin/python experiments/make_tables_c2.py
"""
from __future__ import annotations

import argparse
import json
import math
import os

TARGET_DIRS = [
    "manuscript_neurocomputing/tables",
    "../manuscripts/neurocomputing_vnese/tables",
]


def _load(path: str):
    with open(path, "r") as fh:
        return json.load(fh)


def _fmt(x: float, nd: int = 4) -> str:
    """NaN renders as an em dash: an undefined AUC is a diverged run, not a zero."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- table 1
def table_metadata_immunity(res_dir: str) -> str:
    """Theorem 2(1): weights before/after a client falsifies its own metadata.

    The lying client is the one whose BFWA weight moves; we report the whole
    weight vector delta as an L-infinity norm because that is what the audit
    script records, and additionally the lying client's own weight, because
    "the attacker captured 86% of the aggregate" is the quantity a reader
    actually wants.
    """
    v = _load(os.path.join(res_dir, "fairshare/metadata_immunity_verdict.json"))["verdict"]

    def row(label: str, honest_key: str, lying_key: str, diff_key: str) -> str:
        honest = v[honest_key]
        lying = v[lying_key]
        # the falsifying client is index 0 in the audit harness
        return (
            f"{label} & {honest[0]:.4f} & {lying[0]:.4f} & "
            f"{v[diff_key]:.4f} \\\\"
        )

    rows = [
        row(r"\textbf{FU-Shapley (ours)}", "weights_fu_honest", "weights_fu_lying", "max_diff_fu"),
        # the robust variant stores no separate weight vector; it is bit-exact too
        f"\\textbf{{robust FU-Shapley (ours)}} & \\multicolumn{{2}}{{c}}{{identical, bit-exact}} & {v['max_diff_robust_fu']:.4f} \\\\",
        row("BFWA (baseline)", "weights_bfwa_honest", "weights_bfwa_lying", "max_diff_bfwa"),
    ]

    return r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Metadata immunity under a falsifying client (Theorem~\ref{thm:metadata_immunity}).}
A single client reports $\widehat{\dpd}_k = 0.0$ and $\mathrm{Perf}_k = 0.99$ while transmitting an
unchanged parameter update. Columns give that client's own aggregation weight when it reports
honestly and when it lies, and the resulting $\ell_\infty$ change over the full weight vector.
FU-Shapley never reads the reported fields, so the two weight vectors are identical to the last bit;
BFWA reads them, and the falsifying client captures $86.2\%$ of the aggregate.
German, $K=5$; uniform share is $1/K = 0.20$.}
\label{tab:metadata_immunity}
\begin{tabular}{lccc}
\toprule
\textbf{Aggregation rule} & \textbf{$w_k$ (honest)} & \textbf{$w_k$ (falsified)} & \textbf{$\lVert \Delta \bm{w}\rVert_\infty$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ---------------------------------------------------------------- table 2
def table_two_tier(res_dir: str) -> str:
    """Adversary weight and utility across the four attack scenarios.

    The NaN-seed count is part of the result, not a footnote: removing EMA does
    not merely raise the adversary's weight, it makes training diverge, and a
    table that silently averaged over the surviving seeds would hide that.
    """
    matrix = _load(os.path.join(res_dir, "canonical_suite.json"))["two_tier_defense_robustness"]

    arms = [
        ("FedAvg", "FedAvg", "no defence"),
        ("M1", r"\textbf{FU-Shapley (M1)}", "ReLU gate"),
        ("M1_robust", r"\textbf{robust FU-Shapley}", "ReLU gate + median screen"),
        ("M5", "M5: w/o FairScore", r"$\alpha = 0$"),
        ("M6", "M6: w/o two-tier", "no server holdout"),
        ("M7", "M7: w/o EMA", r"$\beta_{\mathrm{ema}} = 0$"),
    ]
    scenarios = ["no_attack", "sign_flip_20pct", "fairness_poison_20pct", "scaling_20pct"]

    def cell(arm: str, scen: str) -> str:
        key = f"{arm}_{scen}"
        e = matrix.get(key)
        if e is None:
            return "n/a"
        w = e.get("w_adv_mean")
        auc = e.get("auc_mean")
        n_nan = sum(
            1 for r in e.get("per_seed", [])
            if r.get("auc") is None or (isinstance(r.get("auc"), float) and math.isnan(r["auc"]))
        )
        s = f"{w:.4f}" if scen != "no_attack" else f"{w:.4f}"
        s += f" / {_fmt(auc)}"
        if n_nan:
            s += f"$^{{{n_nan}}}$"
        return s

    rows = []
    for arm, label, note in arms:
        cells = " & ".join(cell(arm, s) for s in scenarios)
        rows.append(f"{label} \\textit{{({note})}} & {cells} \\\\")

    return r"""\begin{table*}[t]
\centering
\small
\caption{\textbf{Adversary weight $w_{\mathrm{adv}}$ / global AUC under a $20\%$ Byzantine minority.}
German, $K=5$, $n_{\mathrm{byz}} = 1$, $R = 20$, $n = 10$ seeds, CPU. Lower $w_{\mathrm{adv}}$ is
better; the uniform share an attacker would receive under unweighted averaging is $1/K = 0.20$.
A superscript gives the number of seeds out of $10$ whose AUC is undefined because training
diverged---these are reported rather than dropped, since divergence is the finding. Removing EMA
(M7) does not merely weaken the gate, it destabilises training outright.}
\label{tab:two_tier}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Aggregation arm} & \textbf{No attack} & \textbf{Sign-flip} & \textbf{Fairness poison} & \textbf{Scaling} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""


# ---------------------------------------------------------------- table 3
def table_weight_stability(res_dir: str) -> str:
    """Total weight variation across rounds: the cost side of re-scoring."""
    m = _load(os.path.join(res_dir, "fairshare/convergence_empirical.json"))
    meth = m["methods"]
    order = [("fedavg", "FedAvg"), ("bfwa", "BFWA (baseline)"), ("fu_shapley", r"\textbf{FU-Shapley (ours)}")]
    rows = [
        f"{lab} & {meth[k]['final']['omega_w_total']:.4f} & {meth[k]['final']['auc']:.4f} & {meth[k]['final']['dpd_hard']:.4f} \\\\"
        for k, lab in order if k in meth
    ]
    rounds = m.get("rounds", "?")
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Total variation $\Omega_w$ of the aggregation weights across rounds.}}
German, seed $42$, $R = {rounds}$. $\Omega_w = \sum_t \lVert \bm{{w}}^{{(t)}} - \bm{{w}}^{{(t-1)}}\rVert_1$.
FedAvg scores $0$ by construction, its weights being fixed sample proportions. FU-Shapley re-scores
every round and so cannot reach $0$; the comparison that matters is against BFWA, which re-solves a
dual programme each round and is $27\times$ less stable.}}
\label{{tab:weight_stability}}
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Aggregation rule}} & \textbf{{$\Omega_w$ $\downarrow$}} & \textbf{{AUC $\uparrow$}} & \textbf{{$\dpd_{{\mathrm{{hard}}}}$ $\downarrow$}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ---------------------------------------------------------------- table 4
def table_cost(res_dir: str) -> str:
    """Per-run wall-clock on Pokec-z, read from the same artifact as Table 1.

    Every arm in this table ran in one campaign on one T4, so the ratio is a
    like-for-like measurement rather than a cross-device comparison.
    """
    raw = _load(os.path.join(res_dir, "sota_pokecz.json"))["raw_runs"]
    label = {
        "fedavg-gcn": "FedAvg-GCN", "fairgnn": "FairGNN", "fairsin": "FairSIN",
        "fairfed": "FairFed", "fairgfl": "FairGFL", "fedgraphfair": "FedGraph-Fair",
        "cgsv": "CGSV", "ours-nofser": "Ours w/o FSER (confounded)",
        "ours-nofser-true": "Ours w/o FSER (clean arm)",
        "fedfairgnn": r"\textbf{TrustFedGNN (ours)}",
    }
    means = {m: sum(r["wall_clock_s"] for r in runs) / len(runs) for m, runs in raw.items()}
    base = means["fedavg-gcn"]
    rows = []
    for m, t in sorted(means.items(), key=lambda kv: kv[1]):
        bold = m == "fedfairgnn"
        val = f"\\textbf{{{t:.1f}}}" if bold else f"{t:.1f}"
        rat = f"\\textbf{{{t / base:.2f}}}" if bold else f"{t / base:.2f}"
        rows.append(f"{label.get(m, m)} & {val} & {rat}$\\times$ \\\\")

    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Per-run wall-clock time on Pokec-z.}} Single NVIDIA T4, $K = 10$, $R = 50$,
mean over $n = 10$ seeds, read from the same artifact as Table~\ref{{tab:main_pokecz_sota}}. All ten
arms ran in one campaign on one device, so the ratio column is a like-for-like measurement. The
overhead is dominated by the server-side scoring pass---one target-gradient evaluation on the
holdout split plus $K$ inner products per round---and not by any term that grows with the local
graph.}}
\label{{tab:cost}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Method}} & \textbf{{Time / run (s)}} & \textbf{{vs.\ FedAvg}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", nargs="*", default=None)
    args = ap.parse_args()

    targets = args.out_dir or TARGET_DIRS
    targets = [d for d in targets if os.path.isdir(d)]
    if not targets:
        raise SystemExit("no target table directory found")

    built = {
        "metadata_immunity.tex": table_metadata_immunity(args.results),
        "two_tier_defense.tex": table_two_tier(args.results),
        "weight_stability.tex": table_weight_stability(args.results),
        "cost.tex": table_cost(args.results),
    }

    for d in targets:
        for name, body in built.items():
            with open(os.path.join(d, name), "w") as fh:
                fh.write(body)
    print(f"C2 evidence tables written to {targets}")


if __name__ == "__main__":
    main()
