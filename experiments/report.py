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
    "f2gnn": "F$^2$GNN", "dp-fedavg": "DP-FedAvg", "fedfairgnn-nodp": "TrustFedGNN (no DP)",
    "fedfairgnn": "\\textbf{TrustFedGNN}", "ours-robust": "TrustFedGNN-Robust",
    "trustfedgnn-plus": "TrustFedGNN+Calib",
    "ours-nofser": "w/o FSER", "ours-nobfwa": "w/o BFWA",
    "favgnn": "FaVGNN$^{\\ast}$ (2026)", "fdp-fair": "FDP-Fair (2026)",
    "fairgfl": "FairGFL (2025)", "fedgraphfair": "FedGraph-Fair (2026)",
    "puffle": "PUFFLE", "fedfact": "FedFACT (2025)",
    "popets-fairfed": "PoPETs'25 FairFed",
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
               "f2gnn", "favgnn", "fdp-fair", "fairgfl", "fedgraphfair", "puffle",
               "fedfact", "popets-fairfed", "dp-fedavg",
               "fedfairgnn-nodp", "fedfairgnn", "ours-robust", "trustfedgnn-plus"]
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
            if m == "popets-fairfed":
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
        for method, col, lab in [("fedfairgnn", C["ours"], "TrustFedGNN (FTGD)"),
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
    # representative TrustFedGNN run on bail
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
    ax.set_title("TrustFedGNN convergence (Bail)"); ax.legend()
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
    """Utility AND fairness retained under attack: AUC / DPD / EOD per aggregator.

    A fairness-poisoning attack aims to raise disparity while preserving
    accuracy, so an AUC-only table would hide whether the defence protected
    fairness. We therefore report DPD and EOD under attack alongside AUC.
    A near-chance model (AUC < COLLAPSE_AUC) reports spuriously low disparity
    because it predicts near-constant scores; we mark such cells with a dagger
    and exclude them from the fairness "best" comparison.
    """
    rows = [r for r in rows if "rob_" in r["run_id"]]
    if not rows:
        return
    aggs = ["fedavg", "bfwa", "krum", "multikrum", "median", "trimmed_mean", "robust_bfwa"]
    attacks = ["gaussian", "alie", "fairness_poison"]
    COLLAPSE_AUC = 0.6
    A = {}
    for r in rows:
        for agg in aggs:
            for atk in attacks:
                if f"rob_{agg}_{atk}" in r["run_id"]:
                    A[(agg, atk)] = dict(auc=r.get("final_auc"), dpd=r.get("final_dpd"),
                                         eod=r.get("final_eod"))
    if not A:
        return
    pretty_atk = {"gaussian": "Gaussian", "alie": "ALIE", "fairness_poison": "Fair-poison"}
    # best per (attack, metric): highest AUC; lowest DPD/EOD among non-collapsed models
    best = {}
    for a in attacks:
        present = [g for g in aggs if (g, a) in A]
        if not present:
            continue
        best[(a, "auc")] = max(present, key=lambda g: A[(g, a)]["auc"])
        healthy = [g for g in present if A[(g, a)]["auc"] >= COLLAPSE_AUC]
        for met in ("dpd", "eod"):
            if healthy:
                best[(a, met)] = min(healthy, key=lambda g: A[(g, a)][met])
    header2 = " & ".join("AUC $\\uparrow$ & DPD $\\downarrow$ & EOD $\\downarrow$" for _ in attacks)
    lines = ["\\begin{tabular}{l" + "ccc" * len(attacks) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{pretty_atk[a]}}}" for a in attacks) + " \\\\",
             "Aggregator & " + header2 + " \\\\", "\\midrule"]
    for agg in aggs:
        cells = []
        for a in attacks:
            m = A.get((agg, a))
            if m is None:
                cells += ["--", "--", "--"]
                continue
            collapsed = m["auc"] < COLLAPSE_AUC
            for met in ("auc", "dpd", "eod"):
                v = m[met]
                s = f"{v:.3f}"
                if best.get((a, met)) == agg and not (met != "auc" and collapsed):
                    s = f"\\textbf{{{s}}}"
                if met != "auc" and collapsed:
                    s += "$^{\\dagger}$"
                cells.append(s)
        name = "\\textbf{robust\\_bfwa (ours)}" if agg == "robust_bfwa" else agg.replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "robustness.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] robustness.tex")


def _canonical(rows):
    """The same filtered set the main results tables use: exclude ablation,
    robustness/byzantine, and hyper-parameter-sweep runs so every table reads
    from one consistent configuration."""
    out = []
    for r in rows:
        rid = r["run_id"]
        last = rid.split("__")[-1]
        if ("__abl" in rid or "rob_" in rid or "byz_" in rid or "part_" in rid
                or "eps" in last or "lam" in last or last.startswith("K")):
            continue
        out.append(r)
    return out


def table_trust(rows):
    """Composite trust score per method on Bail from logged metrics.

    Reads from the same canonical run set as the main tables so AUC/DPD/EOD
    match Tables 3--5 exactly, and reports the protocol privacy budget.
    """
    from src.trust.trust_score import trust_score, sub_scores
    ds = "bail"
    methods = ["fedavg-gat", "fairgnn", "fairsin", "fairfed", "f2gnn",
               "dp-fedavg", "fedfairgnn-nodp", "fedfairgnn"]
    crows = [r for r in _canonical(rows) if r["dataset"] == ds]
    A = agg(crows, ("exp_name",), "auc")
    D = agg(crows, ("exp_name",), "dpd")
    E = agg(crows, ("exp_name",), "eod")
    # epsilon per method from the canonical (protocol) runs, not sweep runs
    eps_map = {}
    for r in crows:
        if r.get("dp_enabled"):
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


def per_seed(rows, key_fields, metric):
    """Return {key_tuple: {seed: value}} for paired significance testing."""
    out = defaultdict(dict)
    for r in rows:
        v = r.get(f"final_{metric}")
        if v is None:
            continue
        out[tuple(r.get(k) for k in key_fields)][r.get("seed")] = v
    return out


def significance_report(rows):
    """Paired Wilcoxon signed-rank tests between TrustFedGNN and each baseline,
    per dataset and metric, over shared seeds. Writes a compact LaTeX table and
    prints a summary. Addresses the reviewer's request for significance testing
    rather than boldfacing means from few seeds."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        print("[sig] scipy unavailable; skipping significance report")
        return
    crows = _canonical(rows)
    datasets = ["german", "bail", "credit", "pokec_z", "elliptic"]
    ours = "fedfairgnn"
    baselines = ["fedavg-gat", "fairgnn", "fairsin", "fairfed", "f2gnn",
                 "fdp-fair", "puffle", "dp-fedavg"]
    lines = ["\\begin{tabular}{ll" + "c" * len(datasets) + "}", "\\toprule",
             "Metric & Baseline & " + " & ".join(DS_PRETTY.get(d, d) for d in datasets) + " \\\\",
             "\\midrule"]
    printed = []
    for metric in ["auc", "dpd"]:
        P = per_seed(crows, ("exp_name", "dataset"), metric)
        for b in baselines:
            cells = []
            for d in datasets:
                ov = P.get((ours, d), {})
                bv = P.get((b, d), {})
                shared = sorted(set(ov) & set(bv))
                if len(shared) < 5:
                    cells.append("--")
                    continue
                a = [ov[s] for s in shared]; c = [bv[s] for s in shared]
                if all(abs(x - y) < 1e-12 for x, y in zip(a, c)):
                    cells.append("--"); continue
                try:
                    stat, p = wilcoxon(a, c)
                except Exception:
                    cells.append("--"); continue
                diff = float(np.mean(a) - np.mean(c))
                star = "$^{*}$" if p < 0.05 else ""
                arrow = "" if abs(diff) < 1e-9 else ("$\\uparrow$" if diff > 0 else "$\\downarrow$")
                cells.append(f"{p:.2f}{star}")
                printed.append((metric, b, d, len(shared), diff, p))
            lines.append(f"{metric.upper()} & {PRETTY.get(b, b)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(os.path.join(TAB, "significance.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] significance.tex")
    for metric, b, d, n, diff, p in printed:
        print(f"  {metric} ours vs {b:14s} on {d:9s}: n={n} dmean={diff:+.3f} p={p:.3f}"
              + ("  *sig*" if p < 0.05 else ""))


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


def table_large_scale(rows):
    """Large-scale ogbn-products results: AUC / DPD / EOD / wall-clock per method."""
    rows = [r for r in rows if r.get("dataset") == "ogbn_products"]
    if not rows:
        return
    methods = ["fedavg-gat", "fairsin", "favgnn", "dp-fedavg",
               "fedfairgnn-nodp", "fedfairgnn"]
    nseeds = defaultdict(set)
    for r in rows:
        nseeds[r["exp_name"]].add(r.get("seed"))
    single_seed = all(len(s) <= 1 for s in nseeds.values())
    Aa = agg(rows, ("exp_name",), "auc")
    Ad = agg(rows, ("exp_name",), "dpd")
    Ae = agg(rows, ("exp_name",), "eod")
    Aw = agg(rows, ("exp_name",), "wall_s")
    present = [m for m in methods if (m,) in Aa]
    if not present:
        return
    # A collapsed (near-chance) model reports spuriously perfect DPD/EOD --
    # it predicts (near-)constant scores, so its group gap is trivially zero.
    # Exclude such runs from the fairness "best" comparison so the table does
    # not reward divergence.
    COLLAPSE_AUC = 0.6
    healthy = [m for m in present if Aa[(m,)][0] >= COLLAPSE_AUC]
    best_auc = max(present, key=lambda m: Aa[(m,)][0])
    best_dpd = (min(healthy, key=lambda m: Ad[(m,)][0])
                if healthy and all((m,) in Ad for m in healthy) else None)
    best_eod = (min(healthy, key=lambda m: Ae[(m,)][0])
                if healthy and all((m,) in Ae for m in healthy) else None)
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Method & AUC $\\uparrow$ & DPD $\\downarrow$ & EOD $\\downarrow$ & Time/run (s) \\\\",
             "\\midrule"]
    def cell(pair, bold=False):
        # single-seed study: report a bare scalar, not a misleading "+/- 0.000"
        if single_seed:
            s = f"{pair[0]:.3f}"
            return f"\\textbf{{{s}}}" if bold else s
        return fmt(*pair, bold=bold)
    for m in present:
        collapsed = Aa[(m,)][0] < COLLAPSE_AUC
        auc = cell(Aa[(m,)], bold=(m == best_auc))
        dpd = cell(Ad[(m,)], bold=(m == best_dpd)) if (m,) in Ad else "--"
        eod = cell(Ae[(m,)], bold=(m == best_eod)) if (m,) in Ae else "--"
        if collapsed:
            dpd += "$^{\\dagger}$"; eod += "$^{\\dagger}$"
        wall = f"{Aw[(m,)][0]:.0f}" if (m,) in Aw else "--"
        lines.append(f"{PRETTY.get(m, m)} & {auc} & {dpd} & {eod} & {wall} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "large_scale.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"[table] large_scale.tex ({len(present)} methods)")


def main():
    rows = load()
    print(f"[report] {len(rows)} runs loaded")
    table_datasets()
    table_compliance()
    if not rows:
        return
    table_main(rows)
    table_large_scale(rows)
    table_ablation(rows)
    table_robustness(rows)
    table_trust(rows)
    significance_report(rows)
    fig_privacy(rows)
    fig_pareto(rows)
    fig_robustness(rows)
    fig_convergence(rows)


if __name__ == "__main__":
    main()
