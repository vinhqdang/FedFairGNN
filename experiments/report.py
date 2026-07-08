"""Generate LaTeX tables and PDF figures for the manuscript from logged results.

Reads results/summary.jsonl (+ per-run history) and writes to
manuscript/tables/*.tex and manuscript/figures/*.pdf. Robust to partial
results: each table/figure is skipped if its runs are absent, so this can be
run repeatedly while the experiment matrix is still filling in.

    python -m experiments.report
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
TAB = "manuscript/tables"
FIG = "manuscript/figures"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# consistent, colourblind-safe palette
C = {"ours": "#1b7837", "dp": "#762a83", "base": "#2166ac",
     "b1": "#d73027", "b2": "#fc8d59", "b3": "#4575b4", "b4": "#91bfdb",
     "b5": "#999999", "b6": "#e08214"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

PRETTY = {
    "fedavg-gcn": "FedAvg-GCN", "fedavg-gat": "FedAvg-GAT", "fairgnn": "FairGNN",
    "fairsin": "FairSIN", "fairfed": "FairFed", "qffl": "q-FedAvg", "fedfb": "FedFB",
    "f2gnn": "F$^2$GNN", "dp-fedavg": "DP-FedAvg", "fedfairgnn-nodp": "FedFairGNN (no DP)",
    "fedfairgnn": "\\textbf{FedFairGNN}", "ours-robust": "FedFairGNN-Robust",
    "ours-nofser": "w/o FSER", "ours-nobfwa": "w/o BFWA",
    "favgnn": "FaVGNN$^{\\ast}$ (2026)", "fdp-fair": "FDP-Fair (2026)",
}
DS_PRETTY = {"german": "German", "credit": "Credit", "bail": "Bail",
             "elliptic": "Elliptic", "pokec_z": "Pokec-z"}


def load():
    rows = []
    p = os.path.join(RESULTS, "summary.jsonl")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(run_id):
    p = os.path.join(RESULTS, "runs", f"{run_id}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def agg(rows, key_fields, metric):
    """mean,std of metric grouped by key_fields tuple."""
    buckets = defaultdict(list)
    for r in rows:
        v = r.get(f"final_{metric}")
        if v is None:
            continue
        buckets[tuple(r.get(k) for k in key_fields)].append(v)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in buckets.items()}


def fmt(mean, std, bold=False, prec=3):
    s = f"{mean:.{prec}f}\\,$\\pm$\\,{std:.{prec}f}"
    return f"\\textbf{{{s}}}" if bold else s


# --------------------------------------------------------------------------- #
def table_main(rows):
    rows = [r for r in rows if not r["run_id"].endswith(("abl",)) and "__abl" not in r["run_id"]
            and "eps" not in r["run_id"].split("__")[-1] and "lam" not in r["run_id"].split("__")[-1]
            and "rob_" not in r["run_id"] and "byz_" not in r["run_id"]
            and "K" not in r["run_id"].split("__")[-1] and "part_" not in r["run_id"]]
    methods = ["fedavg-gcn", "fedavg-gat", "fairgnn", "fairsin", "fairfed", "qffl",
               "f2gnn", "favgnn", "fdp-fair", "dp-fedavg",
               "fedfairgnn-nodp", "fedfairgnn", "ours-robust"]
    datasets = ["german", "bail", "credit", "pokec_z", "elliptic"]
    for metric, lower in [("auc", False), ("dpd", True), ("eod", True)]:
        A = agg(rows, ("exp_name", "dataset"), metric)
        present_ds = [d for d in datasets if any((m, d) in A for m in methods)]
        if not present_ds:
            continue
        lines = ["\\begin{tabular}{l" + "c" * len(present_ds) + "}", "\\toprule",
                 "Method & " + " & ".join(DS_PRETTY[d] for d in present_ds) + " \\\\", "\\midrule"]
        # best per column
        best = {}
        for d in present_ds:
            vals = [(m, A[(m, d)][0]) for m in methods if (m, d) in A]
            if vals:
                best[d] = (min if lower else max)(vals, key=lambda x: x[1])[0]
        for m in methods:
            if not any((m, d) in A for d in present_ds):
                continue
            cells = []
            for d in present_ds:
                if (m, d) in A:
                    mean, std = A[(m, d)]
                    cells.append(fmt(mean, std, bold=(best.get(d) == m)))
                else:
                    cells.append("--")
            lines.append(f"{PRETTY.get(m, m)} & " + " & ".join(cells) + " \\\\")
            if m == "fdp-fair":
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(TAB, f"main_{metric}.tex"), "w") as f:
            f.write("\n".join(lines))
        print(f"[table] main_{metric}.tex ({len(present_ds)} datasets)")


def table_ablation(rows):
    rows = [r for r in rows if "__abl" in r["run_id"]]
    if not rows:
        return
    order = ["fedavg-gat", "ours-nofser", "ours-nobfwa", "fedfairgnn-nodp"]
    datasets = sorted({r["dataset"] for r in rows})
    lines = ["\\begin{tabular}{ll" + "cc" * len(datasets) + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_PRETTY.get(d,d)}}}" for d in datasets) + " \\\\",
             "Config & & " + " & ".join("AUC & DPD" for _ in datasets) + " \\\\", "\\midrule"]
    Aa = agg(rows, ("exp_name", "dataset"), "auc")
    Ad = agg(rows, ("exp_name", "dataset"), "dpd")
    label = {"fedavg-gat": "Base GAT", "ours-nofser": "+ BFWA (no FSER)",
             "ours-nobfwa": "+ FSER (no BFWA)", "fedfairgnn-nodp": "+ FSER + BFWA"}
    for m in order:
        cells = []
        for d in datasets:
            if (m, d) in Aa:
                cells.append(f"{Aa[(m,d)][0]:.3f}")
                cells.append(f"{Ad[(m,d)][0]:.3f}")
            else:
                cells += ["--", "--"]
        lines.append(f"{label.get(m,m)} & & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "ablation.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] ablation.tex")


def fig_privacy(rows):
    rows = [r for r in rows if "eps" in r["run_id"].split("__")[-1]]
    if not rows:
        return
    for ds in sorted({r["dataset"] for r in rows}):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for method, col, lab in [("fedfairgnn", C["ours"], "FedFairGNN (FTGD)"),
                                 ("dp-fedavg", C["dp"], "DP-FedAvg")]:
            pts = sorted([(r["dp_epsilon"], r.get("final_auc"), r.get("final_dpd"))
                          for r in rows if r["exp_name"] == method and r["dataset"] == ds])
            if not pts:
                continue
            e, auc, dpd = zip(*pts)
            ax[0].plot(e, auc, "o-", color=col, label=lab)
            ax[1].plot(e, dpd, "o-", color=col, label=lab)
        ax[0].set_xlabel("privacy budget $\\epsilon$"); ax[0].set_ylabel("AUC-ROC")
        ax[0].set_xscale("log"); ax[1].set_xscale("log")
        ax[1].set_xlabel("privacy budget $\\epsilon$"); ax[1].set_ylabel("DPD")
        ax[0].legend(); ax[0].set_title(f"Utility vs privacy ({DS_PRETTY.get(ds,ds)})")
        ax[1].set_title("Fairness vs privacy")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"privacy_{ds}.pdf")); plt.close(fig)
        print(f"[fig] privacy_{ds}.pdf")


def fig_pareto(rows):
    rows = [r for r in rows if "lam" in r["run_id"].split("__")[-1]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for ds, col in zip(sorted({r["dataset"] for r in rows}), [C["ours"], C["dp"], C["base"]]):
        pts = sorted([(r.get("final_dpd"), r.get("final_auc"), r["fairness_weight"])
                      for r in rows if r["dataset"] == ds])
        if not pts:
            continue
        dpd, auc, lam = zip(*pts)
        ax.plot(dpd, auc, "o-", color=col, label=DS_PRETTY.get(ds, ds))
    ax.set_xlabel("DPD (lower = fairer)"); ax.set_ylabel("AUC-ROC")
    ax.set_title("Fairness--utility trade-off (varying $\\lambda$)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pareto.pdf")); plt.close(fig)
    print("[fig] pareto.pdf")


def fig_robustness(rows):
    byz = [r for r in rows if "byz_" in r["run_id"]]
    if byz:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        cols = {"fedavg": C["b1"], "bfwa": C["b2"], "krum": C["b3"], "robust_bfwa": C["ours"]}
        for agg_name, col in cols.items():
            pts = sorted([(r["num_byzantine"], r.get("final_auc"))
                          for r in byz if f"byz_{agg_name}_" in r["run_id"]])
            if not pts:
                continue
            b, auc = zip(*pts)
            ax.plot(b, auc, "o-", color=col, label=agg_name)
        ax.set_xlabel("# Byzantine clients (of 10)"); ax.set_ylabel("AUC-ROC under attack")
        ax.set_title("Robustness to Gaussian attack"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness_byz.pdf")); plt.close(fig)
        print("[fig] robustness_byz.pdf")


def fig_convergence(rows):
    # representative FedFairGNN run on bail
    cand = [r for r in rows if r["exp_name"] == "fedfairgnn" and r["dataset"] == "bail"]
    if not cand:
        cand = [r for r in rows if r["exp_name"] == "fedfairgnn-nodp" and r["dataset"] == "bail"]
    if not cand:
        return
    run = load_run(cand[0]["run_id"])
    if not run or not run.get("history"):
        return
    h = run["history"]
    rounds = [x["round"] for x in h]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(rounds, [x.get("g_auc") for x in h], color=C["ours"], label="AUC")
    ax.plot(rounds, [x.get("g_dpd") for x in h], color=C["b1"], label="DPD")
    ax.plot(rounds, [x.get("g_eod") for x in h], color=C["b3"], label="EOD")
    ax.set_xlabel("communication round"); ax.set_ylabel("metric")
    ax.set_title("FedFairGNN convergence (Bail)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergence.pdf")); plt.close(fig)
    print("[fig] convergence.pdf")


DATASETS_STATIC = [
    ("German", "1{,}000", "43{,}484", "26", "Gender", "0.70", "credit risk"),
    ("Bail", "18{,}876", "623{,}740", "17", "Race (WHITE)", "0.38", "recidivism"),
    ("Credit", "30{,}000", "2{,}843{,}716", "12", "Age", "0.78", "default"),
    ("Pokec-z", "67{,}796", "1{,}235{,}916", "276", "Region", "0.08", "working field"),
    ("Elliptic", "203{,}769", "234{,}355", "165", "Time period$^\\dagger$", "0.02", "illicit (crypto)"),
]


def table_datasets():
    lines = ["\\begin{tabular}{lrrrlcl}", "\\toprule",
             "Dataset & Nodes & Edges & Feat. & Sensitive attr. & Pos.\\ rate & Task \\\\",
             "\\midrule"]
    for name, n, e, d, s, pr, task in DATASETS_STATIC:
        lines.append(f"{name} & {n} & {e} & {d} & {s} & {pr} & {task} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "datasets.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] datasets.tex")


def table_robustness(rows):
    rows = [r for r in rows if "rob_" in r["run_id"]]
    if not rows:
        return
    aggs = ["fedavg", "bfwa", "krum", "multikrum", "median", "trimmed_mean", "robust_bfwa"]
    attacks = ["gaussian", "alie", "fairness_poison"]
    A = {}
    for r in rows:
        for agg in aggs:
            for atk in attacks:
                if f"rob_{agg}_{atk}" in r["run_id"]:
                    A[(agg, atk)] = r.get("final_auc")
    if not A:
        return
    pretty_atk = {"gaussian": "Gaussian", "alie": "ALIE", "fairness_poison": "Fair-poison"}
    lines = ["\\begin{tabular}{l" + "c" * len(attacks) + "}", "\\toprule",
             "Aggregator & " + " & ".join(pretty_atk[a] for a in attacks) + " \\\\", "\\midrule"]
    # best (highest retained AUC) per attack column
    best = {a: max((A[(g, a)] for g in aggs if (g, a) in A), default=None) for a in attacks}
    for agg in aggs:
        cells = []
        for a in attacks:
            v = A.get((agg, a))
            if v is None:
                cells.append("--")
            else:
                cells.append(f"\\textbf{{{v:.3f}}}" if best[a] and abs(v - best[a]) < 1e-9 else f"{v:.3f}")
        name = "\\textbf{robust\\_bfwa (ours)}" if agg == "robust_bfwa" else agg.replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "robustness.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] robustness.tex")


def table_trust(rows):
    """Composite trust score per method on Bail from logged metrics."""
    from src.trust.trust_score import trust_score, sub_scores
    ds = "bail"
    methods = ["fedavg-gat", "fairgnn", "fairsin", "fairfed", "f2gnn",
               "dp-fedavg", "fedfairgnn-nodp", "fedfairgnn"]
    A = agg([r for r in rows if r["dataset"] == ds and "__abl" not in r["run_id"]
             and "rob_" not in r["run_id"] and "byz_" not in r["run_id"]],
            ("exp_name",), "auc")
    D = agg([r for r in rows if r["dataset"] == ds], ("exp_name",), "dpd")
    E = agg([r for r in rows if r["dataset"] == ds], ("exp_name",), "eod")
    # epsilon per method (from summary config)
    eps_map = {}
    for r in rows:
        if r["dataset"] == ds and r.get("dp_enabled"):
            eps_map[r["exp_name"]] = r.get("dp_epsilon")
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Method & AUC & DPD & EOD & $\\epsilon$ & Trust \\\\", "\\midrule"]
    best_trust, best_m = -1, None
    trust_vals = {}
    for m in methods:
        if (m,) not in A:
            continue
        met = {"auc": A[(m,)][0], "dpd": D.get((m,), (0,))[0], "eod": E.get((m,), (0,))[0]}
        eps = eps_map.get(m)
        t = trust_score(met, p=1.0, epsilon=eps)
        trust_vals[m] = (met, eps, t)
        if t > best_trust:
            best_trust, best_m = t, m
    for m in methods:
        if m not in trust_vals:
            continue
        met, eps, t = trust_vals[m]
        name = PRETTY.get(m, m)
        tt = f"\\textbf{{{t:.3f}}}" if m == best_m else f"{t:.3f}"
        epss = f"{eps:.0f}" if eps else "--"
        lines.append(f"{name} & {met['auc']:.3f} & {met['dpd']:.3f} & {met['eod']:.3f} & {epss} & {tt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "trust.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] trust.tex")


def _esc(s):
    return str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def table_compliance():
    from src.trust.compliance import compliance_matrix
    cm = compliance_matrix()
    lines = ["\\begin{tabular}{p{0.30\\textwidth}p{0.62\\textwidth}}", "\\toprule",
             "Requirement & Addressed by \\\\", "\\midrule",
             "\\multicolumn{2}{l}{\\emph{EU AI Act}} \\\\"]
    for r in cm["eu_ai_act"]:
        lines.append(f"{_esc(r['requirement'])} & {_esc(r['how_addressed'])} \\\\")
    lines.append("\\midrule\n\\multicolumn{2}{l}{\\emph{NIST AI RMF}} \\\\")
    for r in cm["nist_ai_rmf"]:
        lines.append(f"{_esc(r['requirement'])} & {_esc(r['how_addressed'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "compliance.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] compliance.tex")


def main():
    rows = load()
    print(f"[report] {len(rows)} runs loaded")
    table_datasets()
    table_compliance()
    if not rows:
        return
    table_main(rows)
    table_ablation(rows)
    table_robustness(rows)
    table_trust(rows)
    fig_privacy(rows)
    fig_pareto(rows)
    fig_robustness(rows)
    fig_convergence(rows)


if __name__ == "__main__":
    main()
