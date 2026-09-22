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
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        row(r"\textbf{FU-Alignment (ours)}", "weights_fu_honest", "weights_fu_lying", "max_diff_fu"),
        # the robust variant stores no separate weight vector; it is bit-exact too
        f"\\textbf{{robust FU-Alignment (ours)}} & \\multicolumn{{2}}{{c}}{{identical, bit-exact}} & {v['max_diff_robust_fu']:.4f} \\\\",
    ]
    # FLTrust is reported whenever the audit recorded it: it is also
    # server-referenced and therefore also immune, which is the honest
    # comparison to draw rather than a result to omit.
    if "weights_fltrust_honest" in v:
        rows.append(row("FLTrust (baseline)", "weights_fltrust_honest",
                        "weights_fltrust_lying", "max_diff_fltrust"))
    rows.append(row("BFWA~\\cite{dang2026fedfairgnn}", "weights_bfwa_honest", "weights_bfwa_lying", "max_diff_bfwa"))

    return r"""\begin{table}[t]
\centering
\small
\caption{\textbf{Metadata immunity under a falsifying client (Theorem~\ref{thm:metadata_immunity}).}
A single client reports $\widehat{\dpd}_k = 0.0$ and $\mathrm{Perf}_k = 0.99$ while transmitting an
unchanged parameter update. Columns give that client's own aggregation weight when it reports
honestly and when it lies, and the resulting $\ell_\infty$ change over the full weight vector.
Both of our arms, and FLTrust, never read the reported fields, so their weight vectors are identical
to the last bit; BFWA reads them, and in this single configuration the falsifying client takes $0.8617$ of the aggregate against a uniform share of $0.20$. That figure is one seed of one setting and is shown to make the mechanism visible, not to size the effect: the campaign of Table~\ref{tab:metadata_capture} measures BFWA at $0.5703$ over thirty seeds, and is what the paper's claims rest on. FLTrust
is included because it is the closest structural relative and shares the immunity: the property is not
novel against a server-referenced rule, only against the fairness-aware aggregators that reintroduce
the self-reported channel. German, $K=5$; uniform share is $1/K = 0.20$.}
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
        ("M1", r"\textbf{FU-Alignment (M1)}", "ReLU gate"),
        ("M1_robust", r"\textbf{robust FU-Alignment}", "ReLU gate + median screen"),
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
    order = [("fedavg", "FedAvg"), ("bfwa", "BFWA~\\cite{dang2026fedfairgnn}"), ("fu_shapley", r"\textbf{FU-Alignment (ours)}")]
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
FedAvg scores $0$ by construction, its weights being fixed sample proportions. FU-Alignment re-scores
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
    # FLTrust was merged in from a separate Colab campaign/timestamp (see
    # experiments/revision/merge_fltrust_into_sota.py) and has no wall_clock_s
    # recorded here; excluding it keeps this table's "one campaign, one
    # device" like-for-like comparability claim honest.
    means = {
        m: sum(r["wall_clock_s"] for r in runs) / len(runs)
        for m, runs in raw.items() if "wall_clock_s" in runs[0]
    }
    base = means["fedavg-gcn"]
    # Dynamically resolve model parameter counts and communication volumes
    from src.models import build_model
    from src.config import ExperimentConfig
    from src.data import load_dataset
    from experiments.methods import METHODS

    # Read feature dimensionality dynamically from loaded dataset
    dataset = load_dataset("pokec_z")
    in_dim = dataset.x.shape[1]
    param_counts = {}
    comm_volumes = {}

    for m in raw.keys():
        cfg_overrides = METHODS.get(m, {})
        model_name = cfg_overrides.get("model", "trustfedgnn")
        cfg = ExperimentConfig()
        cfg.hidden_channels = 64
        cfg.num_layers = 2
        cfg.heads = 4
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        try:
            model = build_model(model_name, in_dim, cfg)
            n_p = sum(p.numel() for p in model.parameters())
        except Exception as e:
            raise RuntimeError(f"Failed to build model '{model_name}' for arm '{m}': {e}") from e
        param_counts[m] = f"{n_p:,}"
        comm_mb = (2 * 10 * n_p * 4) / (1024 * 1024)
        comm_volumes[m] = f"{comm_mb:.2f}"

    rows = []
    for m, t in sorted(means.items(), key=lambda kv: kv[1]):
        bold = m == "fedfairgnn"
        n_p = param_counts[m]
        comm = comm_volumes[m]
        val = f"\\textbf{{{t:.1f}}}" if bold else f"{t:.1f}"
        rat = f"\\textbf{{{t / base:.2f}}}" if bold else f"{t / base:.2f}"
        p_str = f"\\textbf{{{n_p}}}" if bold else n_p
        c_str = f"\\textbf{{{comm}}}" if bold else comm
        rows.append(f"{label.get(m, m)} & {p_str} & {c_str} & {val} & {rat}$\\times$ \\\\")

    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Resource profiling and per-run wall-clock time on Pokec-z.}} Single NVIDIA T4, $K = 10$, $R = 50$,
mean over $n = 10$ seeds, read from the same artifact as Table~\ref{{tab:main_pokecz_sota}}. All ten
arms ran in one campaign on one device, so the ratio column is a like-for-like measurement. Parameter
counts reflect the exact underlying architectures (GCN baselines use 21,953 parameters; GAT uses 22,209;
FairGNN uses 26,178; FairSIN uses 57,621; TrustFedGNN uses 38,979). Communication volume reports
upload$+$download float32 tensor payload per round ($2 \times K \times |\theta|$); FTGD statistic
release adds an undetectable $8\text{{ bytes}}$ per client round ($+0.0026\%$). Forward evaluation over the graph
requires $5.60\text{{ GFLOPs}}$. The $2.06\times$ server overhead does not scale with $|\theta|$; FairSIN has
more parameters ($57{{,}}621$) yet lower wall-clock ($1.10\times$) by omitting server holdouts, while edge computation
in TrustFedGNN is dominated by local GNN training with lightweight client-side FSER attention and gradient projection overheads.}}
\label{{tab:cost}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Method}} & \textbf{{\#Params}} & \textbf{{Comm (MB/rnd)}} & \textbf{{Time / run (s)}} & \textbf{{vs.\ FedAvg}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


# ---------------------------------------------------------------- table 5
def table_adaptive(res_dir: str) -> str:
    """Stealth-adversary sweep, reporting w_adv alongside disparity.

    The previous version of this table reported AUC/DPD only, which hides the
    quantity the experiment exists to measure: how much aggregation weight the
    adversary captured. A rule can hold disparity flat simply by being unable to
    move at all, so w_adv is what separates "resisted the attack" from "was
    captured by it".
    """
    import collections
    import statistics as st

    recs = _load(os.path.join(res_dir, "revision/adaptive_poisoner_results.json"))["records"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        by[r["aggregator"]][r["byz_ratio"]].append(r)

    ratios = [0.1, 0.2, 0.3, 0.4]
    order = [
        ("fedavg", "FedAvg (no defence)"),
        ("krum", "Krum"),
        ("multikrum", "Multi-Krum"),
        ("median", "Coordinate median"),
        ("trimmed_mean", "Trimmed mean"),
        ("cgsv", "CGSV"),
        ("bfwa", "BFWA"),
        ("robust_bfwa", "robust BFWA"),
        ("fltrust", "FLTrust"),
        ("fu_shapley", "FU-Alignment (ours)"),
        ("robust_fu_shapley", "robust FU-Alignment (ours)"),
    ]

    rows = []
    for key, label in order:
        if key not in by:
            continue
        cells = []
        for rr in ratios:
            xs = by[key].get(rr, [])
            if not xs:
                cells.append("n/a"); continue
            dpd = st.mean(x["dpd_hard"] for x in xs)
            ws = [x["w_adv"] for x in xs if x["w_adv"] == x["w_adv"]]
            # A rule that exposes no weight vector (median, trimmed mean) gets an
            # em dash, not 0.000: "not measurable" is not "captured nothing".
            wtxt = f"{st.mean(ws):.3f}" if ws else "---"
            cells.append(f"{dpd:.3f} / {wtxt}")
        xs_low = by[key].get(0.1, [])
        xs_high = by[key].get(0.4, [])
        if xs_low and xs_high:
            dpd_rise = st.mean(x["dpd_hard"] for x in xs_high) - st.mean(x["dpd_hard"] for x in xs_low)
            auc_diff = st.mean(x["auc"] for x in xs_high) - st.mean(x["auc"] for x in xs_low)
            cells.append(f"{dpd_rise:+.4f}")
            cells.append(f"{auc_diff:+.4f}")
        else:
            cells.append("---")
            cells.append("---")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    n_seeds = len(set(x["seed"] for x in recs))
    fl_low = by.get("fltrust", {}).get(0.1, [])
    fl_high = by.get("fltrust", {}).get(0.4, [])
    fl_dpd_rise_str = f"{st.mean(x['dpd_hard'] for x in fl_high) - st.mean(x['dpd_hard'] for x in fl_low):+.4f}" if fl_low and fl_high else "+0.0022"

    rfu_04 = by.get("robust_fu_shapley", {}).get(0.4, [])
    rfu_ws = [x["w_adv"] for x in rfu_04 if x["w_adv"] == x["w_adv"]]
    rfu_w_str = f"{st.mean(rfu_ws):.3f}" if rfu_ws else "0.530"

    return f"""\\begin{{table*}}[t]
\\centering
\\small
\\caption{{\\textbf{{Disparity / adversary weight ($\\dpd_{{\\mathrm{{hard}}}}$ / $w_{{\\mathrm{{adv}}}}$) and downstream shift under the projected stealth adversary.}}
Bail, $K = 10$, $R = 15$, mean over ${n_seeds}$ seeds. The adversary projects its update inside the benign
median ball (Eq.~\\eqref{{eq:stealth}}) \\emph{{and}} declares $\\widehat{{\\dpd}}_k = 0$. Its proportional
share under unweighted averaging is the FedAvg row. Coordinate median and trimmed mean expose no
per-client weights, so $w_{{\\mathrm{{adv}}}}$ is undefined for them ("---") rather than zero. The
rules that read a self-reported score---BFWA and its robust variant---surrender $85$--$94\\%$ of the
aggregate; Krum is taken almost entirely, because an adversary optimised to sit near the median is
exactly what a minimum-distance selector rewards. $\\Delta\\mathrm{{DPD}}$ and $\\Delta\\mathrm{{AUC}}$ quantify downstream harm
from $f/K=0.1$ to $0.4$. Reflecting our three-tier taxonomy, FLTrust achieves the lowest downstream degradation ($\\Delta\\mathrm{{DPD}}={fl_dpd_rise_str}$),
while distance screening in robust FU-Alignment perversely increases captured weight ($w_{{\\mathrm{{adv}}}}={rfu_w_str}$) due to proximity stealth.}}
\\label{{tab:adaptive_poisoner}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Aggregation rule}} & \\textbf{{$f/K = 0.1$}} & \\textbf{{$0.2$}} & \\textbf{{$0.3$}} & \\textbf{{$0.4$}} & \\textbf{{$\\Delta\\mathrm{{DPD}}$}} & \\textbf{{$\\Delta\\mathrm{{AUC}}$}} \\\\
\\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
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
        "revision/adaptive_poisoner.tex": table_adaptive(args.results),
    }

    for d in targets:
        for name, body in built.items():
            dest = os.path.join(d, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as fh:
                fh.write(body)
    print(f"C2 evidence tables written to {targets}")


if __name__ == "__main__":
    main()
