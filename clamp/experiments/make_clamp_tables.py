"""Tables and figures for the CLAMP paper, from results/clamp/<group>/runs/*.json.

Each table is written twice: a CSV (plain text, for inspection) and a LaTeX
fragment (booktabs, no unicode, math in $...$) under results/clamp/tables/. The
privacy tables are written by experiments/clamp_privacy.py from its own CSVs
(render path: ``python -m experiments.clamp_privacy --render-only``). A missing
group produces a placeholder instead of a failure.

Usage:
    python -m experiments.make_clamp_tables [--root results/clamp]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np

from experiments.methods import METHODS
from experiments.run_clamp_suite import E1_METHODS, E3_ATTACKS, E3_ROWS, MAIN

# --------------------------------------------------------------------------- #
# Names and constants
# --------------------------------------------------------------------------- #
DISPLAY = {
    "clamp-f": "CLAMP",
    "clamp-gcn-f": "CLAMP-GCN",
    "clamp-select": "CLAMP-Select (no controller)",
    "clamp-clipmean": "ClipMean (no controller, last iterate)",
    "clamp": "CLAMP (last iterate)",
    "clamp-gcn": "CLAMP-GCN (last iterate)",
    "clamp-p": "CLAMP-P (proportional direction)",
    "fedavg-gcn": "FedAvg-GCN", "fedavg-gat": "FedAvg-GAT", "fairfed": "FairFed",
    "fairgnn": "FairGNN", "fairsin": "FairSIN", "f2gnn": "F2GNN", "fairgfl": "FairGFL",
    "fedgraphfair": "FedGraph-Fair", "puffle": "PUFFLE", "fedfact": "FedFACT",
    "fedfdp": "FedFDP",
}
# Rows of the compact paper table (order as in the paper).
COMPACT_METHODS = ["fedavg-gat", "fairfed", "f2gnn", "fairgfl", "fedgraphfair", "fedfact",
                   "puffle", "fedfdp", "fairgnn", "fairsin", "clamp-clipmean", "clamp",
                   "clamp-select", "clamp-f", "clamp-gcn-f"]

DATASETS = ["german", "bail", "credit", "pokec_z", "pokec_n", "aminer_s", "nba", "facebook"]
DS_NAME = {"german": "German", "bail": "Bail", "credit": "Credit", "pokec_z": "Pokec-z",
           "pokec_n": "Pokec-n", "aminer_s": "AMiner-S", "nba": "NBA", "facebook": "Facebook"}
ATTACK_NAME = {"none": "none", "stat_falsify": "swap", "stat_falsify_mid": "mid"}
UPDATE_ATTACK_NAME = {"sign_flip": "Sign-flip", "scaling": "Scaling", "alie": "ALIE",
                      "ipm": "IPM", "fairness_poison": "Fairness-poison", "none": "None"}
TAU_REPORT = 0.03        # budget used for the "DPD <= tau (seeds)" count
BETA = 0.05              # failure probability of the statistics-noise bound
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Decimal places: two for AUC, three for DPD/EOD (paper convention).
DIG = {"auc": 2, "auc3": 3, "dpd": 3, "eod": 3, "acc": 3}
DEGENERATE_AUC = 0.60   # methods with mean AUC below this on a dataset are not bolded


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
W_FRAC = 0.2   # r2_select_warmup default; the warm-up rule is applied post hoc to old runs


def select_warmup_round(rounds: int, frac: float = W_FRAC) -> int:
    """W = ceil(frac * T), rounded first so that 0.2 * 100 gives 20, not 21."""
    return math.ceil(round(frac * rounds, 9))


def apply_warmup(final: Dict, hist: List[Dict], cfg: Dict):
    """Post-hoc warm-up rule for feasible-output runs made before the warm-up existed.

    The old rule could select round 1 (untrained, trivially feasible). The new rule only
    accepts rounds >= W, so a selection below W is replaced by the last model, whose
    metrics are history[-1] (g_* keys, the same evaluation as final). Exact for the
    selection; the metric values agree with final to about 1e-6 (see the run report).
    Returns (final_metrics, selected_round, changed).
    """
    sel = final.get("selected_round")
    if cfg.get("r2_output") != "feasible" or sel is None:
        return final, sel, False
    if sel >= select_warmup_round(cfg["rounds"]):
        return final, sel, False
    last = hist[-1]
    new = dict(final)
    for k in ("auc", "dpd", "eod"):
        new[k] = last["g_" + k]
    return new, None, True


def load_group(root: str, group: str) -> List[Dict]:
    rows = []
    for path in glob.glob(os.path.join(root, group, "runs", "*.json")):
        with open(path) as f:
            rec = json.load(f)
        cfg, final, hist = rec["config"], rec.get("final", {}), rec.get("history", [])
        final_old = final
        final, sel, changed = apply_warmup(final, hist, cfg)
        if changed:
            CHANGED.append(dict(group=group, method=cfg["exp_name"], dataset=cfg["dataset"],
                                attack=cfg["attack"], b=cfg["num_byzantine"],
                                seed=cfg["seed"], old_sel=final_old.get("selected_round"),
                                old_auc=final_old["auc"], old_dpd=final_old["dpd"],
                                new_auc=final["auc"], new_dpd=final["dpd"]))
        eta = cfg.get("r2_lambda_step")
        lam_T = next((h["lambda"] for h in reversed(hist) if h.get("lambda") is not None), None)
        # CLAMP certificate: lambda^T / (eta T), T = number of rounds.
        cert = (lam_T / (eta * len(hist))) if (lam_T is not None and eta and hist) else math.nan
        rows.append(dict(
            method=cfg["exp_name"], dataset=cfg["dataset"], seed=cfg["seed"],
            aggregator=cfg["aggregator"], attack=cfg["attack"], b=cfg["num_byzantine"],
            tau=cfg["fairness_budget"], eps=cfg["dp_epsilon"], P=cfg.get("r2_stat_period", 1),
            stat_dp=cfg.get("r2_stat_dp", True), clip=cfg.get("r2_clip_norm", 1.0),
            step=cfg.get("r2_lambda_step", 0.5), noise=cfg.get("r2_update_noise", 0.0),
            auc=final.get("auc", math.nan), dpd=final.get("dpd", math.nan),
            dpd_hard=final.get("dpd_hard", math.nan),
            eod=final.get("eod", math.nan), cert=cert, hist=hist,
            selected=sel, cfg=cfg,
            byz=rec.get("byzantine_ids", [])))
    return rows


CHANGED: List[Dict] = []   # runs whose feasible selection the warm-up rule changed


def _series(rows, metric, **match) -> Dict[int, float]:
    """seed -> metric value for rows matching `match`."""
    out = {}
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            out[r["seed"]] = r[metric]
    return out


def _ms(x: List[float]) -> Optional[tuple]:
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return None
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, len(x)


def holm(pvals: List[float]) -> List[float]:
    """Holm step-down adjusted p-values (nan entries are passed through)."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    m = len(idx)
    adj = list(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * pvals[i]))
        adj[i] = running
    return adj


def _wilcoxon(x: Dict[int, float], y: Dict[int, float]) -> float:
    from scipy.stats import wilcoxon
    seeds = sorted(set(x) & set(y))
    a = [x[s] for s in seeds if np.isfinite(x[s]) and np.isfinite(y[s])]
    b = [y[s] for s in seeds if np.isfinite(x[s]) and np.isfinite(y[s])]
    if len(a) < 5 or np.allclose(np.array(a) - np.array(b), 0):
        return math.nan
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return math.nan


# --------------------------------------------------------------------------- #
# LaTeX-safe formatting
# --------------------------------------------------------------------------- #
def pm(ms, metric: str = "auc") -> str:
    """mean$\\pm$sd, with the paper's decimals for the metric."""
    if ms is None:
        return "--"
    d = DIG.get(metric, 3)
    return f"{ms[0]:.{d}f}$\\pm${ms[1]:.{d}f}"


def num(v, d=3) -> str:
    return "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def plain(s: str) -> str:
    """LaTeX cell -> plain text for the CSV."""
    s = re.sub(r"\\textbf\{(.*?)\}", r"\1", s)
    s = s.replace("$\\pm$", "±").replace("$^\\dagger$", "†").replace("\\_", "_")
    s = s.replace("$\\infty$", "inf").replace("$\\varepsilon$", "epsilon")
    s = re.sub(r"\\[a-zA-Z]+", "", s)          # \widehat, \max, \mathrm, ...
    return s.replace("{", "").replace("}", "").replace("$", "").strip()


def _missing(outdir: str, name: str, group: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.csv"), "w") as f:
        f.write(f"note\nno results for group {group}\n")
    print(f"{name}: no results for group {group}; wrote a placeholder")


def write_table(outdir: str, name: str, header, rows, caption: str,
                colspec: Optional[str] = None, size: str = "small",
                resize: bool = False, tabsep: Optional[str] = None) -> None:
    """header: list of LaTeX strings (CSV uses their plain form).
    rows: list of rows, each cell (latex_text, bold). colspec defaults to l then c."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([plain(h) for h in header])
        for r in rows:
            w.writerow([plain(c[0]) for c in r])
    spec = colspec or ("l" + "c" * (len(header) - 1))
    body = []
    for r in rows:
        cells = [(f"\\textbf{{{t}}}" if b else t) for t, b in r]
        body.append(" & ".join(cells) + r" \\")
    sep = [f"\\setlength{{\\tabcolsep}}{{{tabsep}}}"] if tabsep else []
    tab = sep + [f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    if resize:
        tab = [r"\resizebox{\textwidth}{!}{%"] + tab + ["}"]
    lines = [r"\begin{table}[tbp]", r"\centering", f"\\{size}",
             f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}"] + tab + [r"\end{table}", ""]
    with open(os.path.join(outdir, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {name}.csv / {name}.tex ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# E1: main comparison
# --------------------------------------------------------------------------- #
def _e1_stats(rows: List[Dict]) -> Dict:
    """Per dataset: per-method summaries, seed count with DPD <= TAU_REPORT, and the
    Holm-adjusted Wilcoxon p-values of each method against MAIN (AUC and DPD)."""
    res = {}
    for ds in DATASETS:
        dsr = [r for r in rows if r["dataset"] == ds and r["method"] in E1_METHODS]
        if not dsr:
            continue
        methods = [m for m in E1_METHODS if any(r["method"] == m for r in dsr)]
        others = [m for m in methods if m != MAIN]
        p_auc = {m: _wilcoxon(_series(dsr, "auc", method=m), _series(dsr, "auc", method=MAIN))
                 for m in others}
        p_dpd = {m: _wilcoxon(_series(dsr, "dpd", method=m), _series(dsr, "dpd", method=MAIN))
                 for m in others}
        h_auc = dict(zip(others, holm([p_auc[m] for m in others])))
        h_dpd = dict(zip(others, holm([p_dpd[m] for m in others])))
        per = {}
        for m in methods:
            sel = [r for r in dsr if r["method"] == m]
            per[m] = dict(
                auc=_ms([r["auc"] for r in sel]), dpd=_ms([r["dpd"] for r in sel]),
                eod=_ms([r["eod"] for r in sel]), cert=_ms([r["cert"] for r in sel]),
                n=len(sel), tau_ok=sum(1 for r in sel if r["dpd"] <= TAU_REPORT + 1e-12),
                selected=[r["selected"] for r in sel],
                p_auc=h_auc.get(m, math.nan), p_dpd=h_dpd.get(m, math.nan))
        res[ds] = per
    return res


def table_e1(rows: List[Dict], outdir: str) -> None:
    """Full main comparison (not in the manuscript; the compact table is)."""
    rows = [r for r in rows if r["method"] in E1_METHODS]
    stats = _e1_stats(rows)
    if not stats:
        return _missing(outdir, "e1_main", "e1")
    header = ["Dataset", "Method", "AUC", "DPD", "EOD", "Sel. round", "$p_{\\mathrm{AUC}}$",
              "$p_{\\mathrm{DPD}}$", "Cert.", f"DPD $\\le {TAU_REPORT:g}$"]
    out = []
    for ds, per in stats.items():
        best = {k: (max if k == "auc" else min)(
            (per[m][k][0] for m in per if per[m][k] is not None), default=None)
            for k in ("auc", "dpd", "eod")}
        for m, st in per.items():
            cell = lambda k: (pm(st[k], k), st[k] is not None and best[k] is not None
                              and abs(st[k][0] - best[k]) < 1e-12)
            if METHODS.get(m, {}).get("r2_output") == "feasible":
                picked = [s for s in st["selected"] if s is not None]
                sel_txt = (f"{np.mean(picked):.1f} ({len(picked)}/{len(st['selected'])})"
                           if picked else "fallback")
            else:
                sel_txt = "--"
            pa = "--" if m == MAIN else num(st["p_auc"], 3)
            pd = "--" if m == MAIN else num(st["p_dpd"], 3)
            cert = "--" if st["cert"] is None else f"{st['cert'][0]:.4f}"
            out.append([(DS_NAME[ds], False), (DISPLAY.get(m, m), False), cell("auc"),
                        cell("dpd"), cell("eod"), (sel_txt, False), (pa, False), (pd, False),
                        (cert, False), (f"{st['tau_ok']}/{st['n']}", False)])
    write_table(outdir, "e1_main", header, out,
                "Main comparison, mean$\\pm$sd over seeds. Best value per dataset in bold. "
                "Sel.\\ round: mean over seeds with a feasible release (feasible/total); "
                "fallback: no feasible release. $p$: Wilcoxon signed-rank against CLAMP, "
                "paired by seed, Holm-adjusted within dataset.",
                colspec="ll" + "c" * 8, size="footnotesize", resize=True)


def write_e1_compact(rows: List[Dict], outdir: str) -> None:
    """Paper table: two stacked panels of four datasets, AUC and DPD per dataset, dagger
    where the Holm-adjusted Wilcoxon p-value against CLAMP is below 0.05."""
    rows = [r for r in rows if r["method"] in E1_METHODS]
    stats = _e1_stats(rows)
    if not stats:
        return _missing(outdir, "e1_main_compact", "e1")
    os.makedirs(outdir, exist_ok=True)

    def degenerate(ds, m):
        st = stats.get(ds, {}).get(m)
        return st is None or st["auc"] is None or st["auc"][0] < DEGENERATE_AUC

    best = {}
    for ds in stats:
        for k in ("auc", "dpd"):
            vals = [stats[ds][m][k][0] for m in COMPACT_METHODS
                    if m in stats[ds] and stats[ds][m][k] is not None
                    and not degenerate(ds, m)]
            if vals:
                best[(ds, k)] = max(vals) if k == "auc" else min(vals)

    def cell(ds, m, k):
        st = stats.get(ds, {}).get(m)
        if st is None or st[k] is None:
            return "--"
        txt = pm(st[k], k)
        if not degenerate(ds, m) and abs(st[k][0] - best.get((ds, k), math.inf)) < 1e-12:
            txt = f"\\textbf{{{txt}}}"
        p = st["p_auc" if k == "auc" else "p_dpd"]
        if m != MAIN and p == p and p < 0.05:
            txt += "$^\\dagger$"
        return txt

    def panel(dss):
        spec = "p{2.4cm}" + " cc" * len(dss)
        lines = [f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
                 " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in dss)
                 + r" \\",
                 " & " + " & ".join("AUC & DPD" for _ in dss) + r" \\", r"\midrule"]
        for m in COMPACT_METHODS:
            name = DISPLAY.get(m, m)
            if m == MAIN:
                name = f"\\textbf{{{name}}}"
            cells = []
            for d in dss:
                cells += [cell(d, m, "auc"), cell(d, m, "dpd")]
            lines.append(f"\\raggedright {name} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}"]
        lines[0] = r"\resizebox{\textwidth}{!}{%" + "\n" + lines[0]
        return "\n".join(lines)

    # CSV: the same numbers in plain text, one row per method and dataset.
    header = ["Dataset", "Method", "AUC", "DPD"]
    csv_rows = []
    for ds in DATASETS:
        for m in COMPACT_METHODS:
            if m in stats.get(ds, {}):
                csv_rows.append([(DS_NAME[ds], False), (DISPLAY.get(m, m), False),
                                 (cell(ds, m, "auc"), False), (cell(ds, m, "dpd"), False)])
    with open(os.path.join(outdir, "e1_main_compact.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([plain(h) for h in header])
        for r in csv_rows:
            w.writerow([plain(c[0]) for c in r])

    caption = ("Main results: test AUC and DPD, mean$\\pm$sd over seeds. Best per column in "
               "bold; methods with mean AUC below 0.60 on a dataset (degenerate) are not "
               "bolded. $^\\dagger$: Holm-adjusted Wilcoxon signed-rank $p<0.05$ against CLAMP, "
               "paired by seed.")
    body = "\n".join([
        r"\begin{table}[tbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{1.5pt}",
        f"\\caption{{{caption}}}", r"\label{tab:e1_main_compact}",
        panel(DATASETS[:4]), r"\vspace{4pt}", panel(DATASETS[4:]),
        r"\end{table}", ""])
    with open(os.path.join(outdir, "e1_main_compact.tex"), "w") as f:
        f.write(body)
    print("wrote e1_main_compact.tex")


BACKBONE = {"gcn": "GCN", "gat": "GAT"}


def backbone(method: str) -> str:
    return BACKBONE.get(METHODS.get(method, {}).get("model", ""), "own")


def write_e1_appendix(rows: List[Dict], outdir: str) -> None:
    """All datasets and all main-comparison methods, with hard DPD and the backbone.
    A longtable (the appendix table runs over several pages)."""
    rows = [r for r in rows if r["method"] in E1_METHODS]
    if not rows:
        return _missing(outdir, "e1_full_appendix", "e1")
    os.makedirs(outdir, exist_ok=True)
    header = ["Dataset", "Method", "Backbone", "AUC", "DPD (soft)", "DPD (hard)", "EOD",
              "Cert.", f"DPD $\\le {TAU_REPORT:g}$"]
    body, csv_rows = [], []
    for ds in DATASETS:
        for m in E1_METHODS:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            cells = ["--"] * 6
            if sel:
                cells = [pm(_ms([r["auc"] for r in sel]), "auc"),
                         pm(_ms([r["dpd"] for r in sel]), "dpd"),
                         pm(_ms([r["dpd_hard"] for r in sel]), "dpd"),
                         pm(_ms([r["eod"] for r in sel]), "eod")]
                cert = _ms([r["cert"] for r in sel]) if m.startswith("clamp") else None
                cells.append("--" if cert is None else f"{cert[0]:.4f}")
                ok = sum(1 for r in sel if r["dpd"] <= TAU_REPORT + 1e-12)
                cells.append(f"{ok}/{len(sel)}")
            line = [DS_NAME[ds], DISPLAY.get(m, m), backbone(m)] + cells
            body.append(line)
            csv_rows.append([plain(c) for c in line])
    with open(os.path.join(outdir, "e1_full_appendix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([plain(h) for h in header])
        w.writerows(csv_rows)
    caption = ("Full main comparison, mean$\\pm$sd over seeds, for all datasets and methods. "
               "DPD (soft) is the demographic-parity gap of the scores; DPD (hard) uses a 0.5 "
               "threshold. Cert.: $\\lambda^{R}/(\\eta R)$ for the CLAMP variants. "
               "Backbone: GCN or GAT as in the method definition, own: the baseline's own "
               "architecture. Degenerate runs (AUC near chance) are included.")
    spec = "lp{2.2cm}l" + "c" * 6
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{1pt}",
             f"\\begin{{longtable}}{{{spec}}}",
             f"\\caption{{{caption}}}\\label{{tab:e1_full}}\\\\",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule", r"\endfirsthead",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule", r"\endhead",
             r"\midrule", r"\multicolumn{9}{r}{\textit{continued on next page}} \\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    for line in body:
        lines.append(" & ".join(line) + r" \\")
    lines += [r"\end{longtable}", ""]
    with open(os.path.join(outdir, "e1_full_appendix.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote e1_full_appendix.csv / .tex ({len(body)} rows)")


# --------------------------------------------------------------------------- #
# E2: tau sweep
# --------------------------------------------------------------------------- #
def figure_e2(rows: List[Dict], outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = [d for d in ["bail", "credit", "pokec_z"] if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, max(1, len(dss)), figsize=(3.2 * max(1, len(dss)), 3.0),
                             squeeze=False)
    for ax, ds in zip(axes[0], dss):
        taus = sorted({r["tau"] for r in rows if r["dataset"] == ds})
        ms = [_ms([r["dpd"] for r in rows if r["dataset"] == ds and r["tau"] == t]) for t in taus]
        ok = [(t, m) for t, m in zip(taus, ms) if m is not None]
        ax.errorbar([t for t, _ in ok], [m[0] for _, m in ok], yerr=[m[1] for _, m in ok],
                    marker="o", capsize=3, label="CLAMP")
        lim = max([t for t, _ in ok] + [m[0] for _, m in ok] + [0.01]) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
        ax.set_xlabel(r"fairness budget $\tau$")
        ax.set_ylabel("test DPD")
        ax.set_title(DS_NAME.get(ds, ds))
        ax.grid(alpha=0.3)
    if dss:
        axes[0][0].legend(frameon=False)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "e2_tau_pareto.pdf"))
    plt.close(fig)
    print("wrote e2_tau_pareto.pdf")


def table_e2(rows: List[Dict], outdir: str) -> None:
    rows = [r for r in rows if r["method"] == MAIN]
    if not rows:
        return _missing(outdir, "e2_tau", "e2")
    header = ["Dataset", "$\\tau$", "AUC", "DPD", "EOD", "Seeds"]
    out = []
    for ds in ["bail", "credit", "pokec_z"]:
        for tau in sorted({r["tau"] for r in rows if r["dataset"] == ds}):
            sel = [r for r in rows if r["dataset"] == ds and r["tau"] == tau]
            out.append([(DS_NAME[ds], False), (f"{tau:g}", False),
                        (pm(_ms([r["auc"] for r in sel]), "auc3"), False),
                        (pm(_ms([r["dpd"] for r in sel]), "dpd"), False),
                        (pm(_ms([r["eod"] for r in sel]), "eod"), False),
                        (str(len(sel)), False)])
    write_table(outdir, "e2_tau", header, out,
                "Fairness budget sweep for CLAMP, mean$\\pm$sd over seeds.")
    figure_e2(rows, outdir)


# --------------------------------------------------------------------------- #
# E3: update attacks
# --------------------------------------------------------------------------- #
def _label(r: Dict) -> str:
    own = METHODS.get(r["method"], {}).get("aggregator")
    return r["method"] if r["aggregator"] == own else f"{r['method']}[{r['aggregator']}]"


def _agg_display(label: str) -> str:
    if "[" not in label:
        return DISPLAY.get(label, label)
    base, agg = label.split("[", 1)
    return f"{DISPLAY.get(base, base)} ({agg.rstrip(']').replace('_', ' ')})"


def table_e3(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e3_attacks", "e3")
    header = ["Dataset", "Attack", "Aggregator", "$f$", "AUC", "DPD"]
    out = []
    labels = [lab for _, _, lab in E3_ROWS]
    for ds in ["bail", "pokec_z"]:
        for attack in ["none"] + E3_ATTACKS:
            for lab in labels:
                for b in ([0] if attack == "none" else [1, 2, 3, 4]):
                    sel = [r for r in rows if r["dataset"] == ds and r["attack"] == attack
                           and r["b"] == b and _label(r) == lab]
                    if not sel:
                        continue
                    out.append([(DS_NAME[ds], False), (UPDATE_ATTACK_NAME[attack], False),
                                (_agg_display(lab), False), (str(b), False),
                                (pm(_ms([r["auc"] for r in sel]), "auc"), False),
                                (pm(_ms([r["dpd"] for r in sel]), "dpd"), False)])
    write_table(outdir, "e3_attacks", header, out,
                "Update attacks: utility and fairness against the number $f$ of Byzantine "
                "clients out of $K=10$; mean$\\pm$sd over seeds.")


E3_AGG = [("fedavg-gat", "FedAvg", False), ("fedavg-gat[median]", "Median", True),
          ("fedavg-gat[trimmed_mean]", "Trimmed mean", True),
          ("fedavg-gat[krum]", "Krum", True), ("clamp-f", "CLAMP", False),
          ("clamp-clipmean", "ClipMean", False)]
E3_ATTACK_ROWS = [("none", "None"), ("sign_flip", "Sign-flip"), ("scaling", "Scaling"),
                  ("alie", "ALIE"), ("ipm", "IPM"), ("fairness_poison", "Fairness-poison")]


def _e3_sel(rows, ds, attack, f, label):
    return [r for r in rows if r["dataset"] == ds and r["attack"] == attack
            and r["b"] == f and _label(r) == label]


def write_e3_compact(rows: List[Dict], outdir: str) -> None:
    """Two floats, one per dataset: attack x aggregator, f=0 (no attack, None row only),
    f=2 and f=4, mean$\\pm$sd in small type. Best AUC per attack and f in bold."""
    rows = [r for r in rows if r["method"] in ("fedavg-gat", "clamp-f", "clamp-clipmean")]
    if not rows:
        return _missing(outdir, "e3_compact_bail", "e3")
    os.makedirs(outdir, exist_ok=True)
    groups = [(0, "$f=0$"), (2, "$f=2$"), (4, "$f=4$")]
    short = {"clamp-clipmean": "ClipMean"}

    def summ(ds, attack, f, label, metric):
        return _ms([r[metric] for r in _e3_sel(rows, ds, attack, f, label)])

    def float_for(ds, name, label_tex, dname):
        lines = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
                 r"\setlength{\tabcolsep}{2pt}",
                 f"\\caption{{Update attacks on {dname}: mean$\\pm$sd over 3 seeds, $K=10$ "
                 "clients. AUC (two decimals) and DPD (three decimals) at $f=0$ (no attack, None "
                 "row only), $f=2$ and $f=4$ Byzantine clients. Best AUC per attack and $f$ in "
                 "bold. $^\\dagger$: median, trimmed mean and Krum inspect individual client "
                 "updates and cannot run under secure aggregation. ClipMean: clipped mean with "
                 "no fairness controller, last iterate.}",
                 f"\\label{{{label_tex}}}",
                 "\\begin{tabular}{ll" + "cc" * len(groups) + "}", r"\toprule",
                 " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{g}}}" for _, g in groups)
                 + r" \\",
                 " & & " + " & ".join("AUC & DPD" for _ in groups) + r" \\", r"\midrule"]
        for attack, aname in E3_ATTACK_ROWS:
            fs = [0] if attack == "none" else [2, 4]
            best = {}
            for f in (0, 2, 4):
                vals = [summ(ds, attack, f, lab, "auc") for lab, _, _ in E3_AGG]
                vals = [v[0] for v in vals if v is not None]
                best[f] = max(vals) if vals else None
            for k, (lab, disp, dag) in enumerate(E3_AGG):
                name_cell = short.get(lab, disp) + ("$^\\dagger$" if dag else "")
                cells = []
                for f, _ in groups:
                    if f not in fs:
                        cells += ["--", "--"]
                        continue
                    a = summ(ds, attack, f, lab, "auc")
                    d = summ(ds, attack, f, lab, "dpd")
                    if a is None:
                        cells += ["--", "--"]
                        continue
                    atxt = pm(a, "auc")
                    if best[f] is not None and abs(a[0] - best[f]) < 1e-12:
                        atxt = f"\\textbf{{{atxt}}}"
                    cells += [atxt, pm(d, "dpd")]
                lead = (f"\\multirow{{{len(E3_AGG)}}}{{*}}{{{aname}}}" if k == 0 else "")
                lines.append(f"{lead} & {name_cell} & " + " & ".join(cells) + r" \\")
            if attack != E3_ATTACK_ROWS[-1][0]:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        with open(os.path.join(outdir, f"{name}.tex"), "w") as fh:
            fh.write("\n".join(lines))
        print(f"wrote {name}.tex")

    float_for("bail", "e3_compact_bail", "tab:e3_bail", "Bail")
    float_for("pokec_z", "e3_compact_pokec", "tab:e3_pokec", "Pokec-z")


def write_e3_appendix(rows: List[Dict], outdir: str) -> None:
    """Full E3: every f, mean and sd, both datasets, as a longtable."""
    rows = [r for r in rows if r["method"] in ("fedavg-gat", "clamp-f", "clamp-clipmean")]
    if not rows:
        return _missing(outdir, "e3_full_appendix", "e3")
    os.makedirs(outdir, exist_ok=True)
    header = ["Dataset", "Attack", "$f$", "Aggregator", "AUC", "DPD"]
    body = []
    for ds in ["bail", "pokec_z"]:
        for attack, aname in E3_ATTACK_ROWS:
            fs = [0] if attack == "none" else [1, 2, 3, 4]
            for f in fs:
                for lab, disp, dag in E3_AGG:
                    sel = _e3_sel(rows, ds, attack, f, lab)
                    if not sel:
                        continue
                    body.append([DS_NAME[ds], aname, str(f),
                                 disp + ("$^\\dagger$" if dag else ""),
                                 pm(_ms([r["auc"] for r in sel]), "auc"),
                                 pm(_ms([r["dpd"] for r in sel]), "dpd")])
    with open(os.path.join(outdir, "e3_full_appendix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([plain(h) for h in header])
        w.writerows([[plain(c) for c in r] for r in body])
    caption = ("Full update-attack results, mean$\\pm$sd over 3 seeds, K=10 clients, for every "
               "number $f$ of Byzantine clients. $^\\dagger$: median, trimmed mean and Krum "
               "inspect individual client updates and cannot run under secure aggregation.")
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", "\\begin{longtable}{llclcc}",
             f"\\caption{{{caption}}}\\label{{tab:e3_full}}\\\\",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule", r"\endfirsthead",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule", r"\endhead",
             r"\midrule", r"\multicolumn{6}{r}{\textit{continued on next page}} \\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    for r in body:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\end{longtable}", ""]
    with open(os.path.join(outdir, "e3_full_appendix.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote e3_full_appendix.csv / .tex ({len(body)} rows)")


# --------------------------------------------------------------------------- #
# E4: statistics attacks against the Theorem 3 bound
# --------------------------------------------------------------------------- #
_PART_CACHE: Dict = {}


def _honest_setup(cfg: Dict, byz: List[int]):
    """(N^H_0, N^H_1), and the public cap of the trainer (largest client validation
    size over all clients, rounded up to a multiple of 50; see public_n_cap)."""
    counts, max_val = _partition_counts(cfg)
    honest = [k for k in range(len(counts)) if k not in set(byz)]
    nh = counts[honest].sum(0).astype(float)
    cap = int(math.ceil(max_val / 50.0) * 50)
    return nh, cap


def _honest_counts(cfg: Dict, byz: List[int]) -> np.ndarray:
    """N^H_a = validation counts per group, summed over the honest clients.

    These counts do not change across rounds (they depend only on the partition), so
    they are recomputed here from the same partition the trainer uses instead of
    being read from the round logs, which do not record them. The trainer's per-client
    cap r2_n_cap is derived from the same counts, so it never cuts an honest group.
    """
    return _honest_setup(cfg, byz)[0]


def _partition_counts(cfg: Dict):
    """Per-client (N_{k,0}, N_{k,1}) on the validation split, and the largest client
    validation size, for the partition the trainer uses (cached)."""
    from src.data import load_dataset, partition_graph
    if cfg.get("fu_val_source", "pooled") != "pooled":
        raise NotImplementedError("honest counts assume the pooled validation split")
    key = (cfg["dataset"], cfg["seed"], cfg["num_clients"], cfg["partition"],
           cfg["dirichlet_alpha"], cfg["partition_by"])
    if key not in _PART_CACHE:
        data = load_dataset(cfg["dataset"], root=os.path.join(_REPO, "data"), seed=cfg["seed"])
        parts = partition_graph(data, cfg["num_clients"], method=cfg["partition"],
                                alpha=cfg["dirichlet_alpha"], by=cfg["partition_by"],
                                seed=cfg["seed"])
        counts = np.array(
            [[int((d.sensitive_attr[d.val_mask] == a).sum()) for a in (0, 1)] for d in parts])
        max_val = max(int(d.val_mask.sum()) for d in parts)
        _PART_CACHE[key] = (counts, max_val)
    return _PART_CACHE[key]


_SIGMA_CACHE: Dict = {}


def _sigma_s(cfg: Dict) -> float:
    """sigma_s = 2 z(eps, ceil(R/P), delta), as in the trainer; 0 if stats noise is off."""
    if not cfg.get("r2_stat_dp", True):
        return 0.0
    from src.trust.privacy import calibrate_noise_multiplier
    steps = math.ceil(cfg["rounds"] / max(1, cfg.get("r2_stat_period", 1)))
    key = (cfg["dp_epsilon"], steps, cfg["dp_delta"])
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = 2.0 * calibrate_noise_multiplier(
            cfg["dp_epsilon"], steps=steps, delta=cfg["dp_delta"])
    return _SIGMA_CACHE[key]


def _noise_term(sigma_s: float, n_h: float, beta: float = BETA) -> float:
    """Per-group noise contribution 2e / (N^H_a - e), with e = sigma_s sqrt(2 ln(8/beta)),
    so that |S/N - S^H/N^H| from the statistics noise is covered with probability 1-beta."""
    if sigma_s <= 0:
        return 0.0
    e = sigma_s * math.sqrt(2.0 * math.log(8.0 / beta))
    return 2.0 * e / (n_h - e) if n_h > e else math.inf


def table_e4(rows: List[Dict], outdir: str) -> None:
    """Deviation of the server's estimate from the honest-only gap, against the full
    Theorem 3 bound: b_0 + b_1 (the logged t3_bound) plus the statistics-noise terms.
    The f = 0 row therefore reports the noise term alone."""
    rows = [r for r in rows if r["method"] == MAIN]
    if not rows:
        return _missing(outdir, "e4_stat_attacks", "e4")
    header = ["Dataset", "Attack", "$f$", "$|\\widehat{D}-D^H|_{\\max}$", "Bound", "Viol.",
              "AUC", "DPD"]
    out, checked_all, runs_all = [], set(), set()
    for ds in ["bail", "pokec_z", "credit"]:
        for attack in ["none", "stat_falsify", "stat_falsify_mid"]:
            for b in ([0] if attack == "none" else [1, 2, 3, 4]):
                sel = [r for r in rows if r["dataset"] == ds and r["attack"] == attack
                       and r["b"] == b]
                if not sel:
                    continue
                devs, bounds, viol, checked = [], [], 0, 0
                for r in sel:
                    rr = [h for h in r["hist"] if h.get("dhat") is not None
                          and h.get("dhat_honest") is not None]
                    if not rr:
                        continue
                    nh, cap_public = _honest_setup(r["cfg"], r["byz"])
                    # The run's own cap if it logged one, else the public cap. Either way
                    # this is the cap the attack ran with, so the bound is consistent.
                    cap = next((h["n_cap"] for h in rr if h.get("n_cap") is not None),
                               cap_public)
                    f = len(r["byz"])
                    bterm = sum(f * cap / (nh[a] + f * cap) for a in (0, 1)) if f else 0.0
                    sig = _sigma_s(r["cfg"])
                    noise = sum(_noise_term(sig, nh[a]) for a in (0, 1))
                    d = [abs(h["dhat"] - h["dhat_honest"]) for h in rr]
                    full = [bterm + noise for h in rr]
                    devs.append(max(d)); bounds.append(max(full))
                    viol += sum(1 for x, y in zip(d, full) if x > y + 1e-12)
                    checked += len(rr)
                checked_all.add(checked)
                runs_all.add(len(sel))
                out.append([(DS_NAME[ds], False), (ATTACK_NAME[attack], False), (str(b), False),
                            (num(np.mean(devs)) if devs else "--", False),
                            (num(np.mean(bounds)) if bounds else "--", False),
                            (str(viol), False),
                            (pm(_ms([r["auc"] for r in sel]), "auc"), False),
                            (pm(_ms([r["dpd"] for r in sel]), "dpd"), False)])
    if not out:
        return _missing(outdir, "e4_stat_attacks", "e4")
    if len(checked_all) == 1 and len(runs_all) == 1:
        per_row = (f"{max(checked_all)} release rounds ({max(runs_all)} seeds "
                   f"of {max(checked_all) // max(runs_all)} rounds)")
    else:
        per_row = "varying numbers of release rounds and seeds"
    write_table(outdir, "e4_stat_attacks", header, out,
                "Statistics attacks. $|\\widehat{D}-D^H|_{\\max}$: largest deviation of the "
                "server's estimate from the honest-only gap over the release rounds; Bound: "
                "largest Theorem 3 bound (honest-mass term plus statistics-noise term, "
                f"$\\beta={BETA:g}$) over the release rounds; Viol.: release rounds with the "
                f"deviation above the bound. Each row covers {per_row}; $f$ is the number of "
                "falsifying clients out of $K=10$. The bound uses the public cap (the cap the "
                "attack ran with), so it is computable without any group-level count. "
                "AUC and DPD: mean$\\pm$sd over seeds.",
                colspec="llc" + "c" * 5, size="footnotesize")


# --------------------------------------------------------------------------- #
# E5 and E5b: privacy sweeps
# --------------------------------------------------------------------------- #
def write_stacked(outdir: str, name: str, out_rows, dss, caption: str) -> None:
    """One table, stacked panels of two datasets each. Rows are (eps, P, then three
    cells per dataset: AUC, DPD, fallback)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        head = ["epsilon", "P"]
        for d in dss:
            head += [f"{DS_NAME[d]} AUC", f"{DS_NAME[d]} DPD", f"{DS_NAME[d]} fallback"]
        w.writerow(head)
        for r in out_rows:
            w.writerow([plain(c[0]) for c in r])
    panels = []
    for k in range(0, len(dss), 2):
        chunk = dss[k:k + 2]
        cols = 2 + 3 * len(chunk)
        head = ["$\\varepsilon$", "$P$"]
        for d in chunk:
            head += [f"{DS_NAME[d]} AUC", f"{DS_NAME[d]} DPD", "fallback"]
        lines = [f"\\begin{{tabular}}{{{'cc' + 'ccc' * len(chunk)}}}", r"\toprule",
                 " & ".join(head) + r" \\", r"\midrule"]
        for r in out_rows:
            cells = [r[0][0], r[1][0]]
            base = 2 + 3 * k  # column offset of this panel's first dataset
            for j in range(3 * len(chunk)):
                cells.append(r[base + j][0])
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        panels.append("\n".join(lines))
    body = "\n".join([r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
                      r"\setlength{\tabcolsep}{3pt}",
                      f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
                      panels[0], r"\vspace{4pt}", panels[1], r"\end{table}", ""])
    with open(os.path.join(outdir, f"{name}.tex"), "w") as f:
        f.write(body)
    print(f"wrote {name}.csv / {name}.tex (stacked, {len(out_rows)} rows)")


def table_e5(rows: List[Dict], outdir: str, group: str = "e5", tau: Optional[float] = None,
             datasets: Optional[List[str]] = None, name: Optional[str] = None,
             default_p: str = "P=10 for German, P=1 for Bail") -> None:
    """Rows are (epsilon, P); columns AUC, DPD and fallback per dataset (the number of
    seeds with no eligible feasible release after the warm-up). The statistics-noise-off
    run is merged in as epsilon = infinity. For e5b, the unconstrained ClipMean
    reference is a final row."""
    name = name or ("e5_privacy" if group == "e5" else f"{group}_privacy")
    dss = datasets or ["german", "bail"]
    if tau is not None:
        rows = [r for r in rows if r["tau"] == tau]
    rows = [r for r in rows if r["method"] == MAIN or r["method"] == "clamp-clipmean"]
    if not rows:
        return _missing(outdir, name, group)
    header = ["$\\varepsilon$", "$P$"]
    for ds in dss:
        header += [f"{DS_NAME[ds]} AUC", f"{DS_NAME[ds]} DPD", "fallback"]

    def cells(sel):
        """AUC, DPD and the fallback count (seeds with selected_round None)."""
        return [(pm(_ms([r["auc"] for r in sel]), "auc"), False),
                (pm(_ms([r["dpd"] for r in sel]), "dpd"), False),
                (str(sum(1 for r in sel if r["selected"] is None)), False)]
    keys = sorted({(r["eps"], r["P"]) for r in rows
                   if r["method"] == MAIN and r["stat_dp"]})
    out = []
    for eps, P in keys:
        line = [(f"{eps:g}", False), (str(P), False)]
        for ds in dss:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == MAIN
                   and r["stat_dp"] and r["eps"] == eps and r["P"] == P]
            line += cells(sel)
        out.append(line)
    off = [r for r in rows if r["method"] == MAIN and not r["stat_dp"]]
    if off:
        line = [("$\\infty$", False), ("--", False)]
        for ds in dss:
            line += cells([r for r in off if r["dataset"] == ds])
        out.append(line)
    ref = [r for r in rows if r["method"] == "clamp-clipmean"]
    if ref and group == "e5b":
        line = [("ClipMean", False), ("--", False)]
        for ds in dss:
            sel = [r for r in ref if r["dataset"] == ds]
            line += [(pm(_ms([r["auc"] for r in sel]), "auc"), False),
                     (pm(_ms([r["dpd"] for r in sel]), "dpd"), False), ("--", False)]
        out.append(line)
    budget = (f", fairness budget $\\tau={tau:g}$" if tau is not None else "")
    n_seeds = len({r["seed"] for r in rows if r["method"] == MAIN and r["dataset"] in dss})
    seeds_txt = f"{n_seeds} seeds"
    caption = ("Privacy: utility and fairness against the statistics budget $\\varepsilon$ and "
               "release period $P$" + budget + ". $\\varepsilon=\\infty$: statistics noise "
               f"off, default period ({default_p}). mean$\\pm$sd over {seeds_txt}. "
               "Fallback: number of "
               "seeds with no eligible feasible release after the warm-up, so the last model "
               "is reported.")
    if group == "e5b":
        caption += " The ClipMean row is the unconstrained reference (no Lagrangian term)."
    if len(dss) <= 2:
        write_table(outdir, name, header, out, caption, size="footnotesize", tabsep="3pt")
        return
    # More than two datasets: stacked tabulars of two datasets each (same rows), one table.
    write_stacked(outdir, name, out, dss, caption)


# --------------------------------------------------------------------------- #
# E6: ablations
# --------------------------------------------------------------------------- #
def table_e6(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e6_ablation", "e6")
    header = ["Dataset", "Method", "$C$", "$\\eta$", "$\\sigma_u$", "AUC", "DPD"]
    out = []
    for ds in ["bail", "credit"]:
        keys = sorted({(r["method"], r["clip"], r["step"], r["noise"]) for r in rows
                       if r["dataset"] == ds})
        for m, c, s, n in keys:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m
                   and (r["clip"], r["step"], r["noise"]) == (c, s, n)]
            out.append([(DS_NAME[ds], False), (DISPLAY.get(m, m), False), (f"{c:g}", False),
                        (f"{s:g}", False), (f"{n:g}", False),
                        (pm(_ms([r["auc"] for r in sel]), "auc"), False),
                        (pm(_ms([r["dpd"] for r in sel]), "dpd"), False)])
    write_table(outdir, "e6_ablation", header, out,
                "CLAMP ablations, one factor at a time around $C=1$, $\\eta=0.5$, "
                "$\\sigma_u=0$, mean$\\pm$sd over seeds. The last two rows at the default "
                "setting are the output-rule (last iterate) and direction-rule "
                "(proportional) ablations.", size="footnotesize", tabsep="3pt",
                colspec="lp{3.6cm}ccccc")


# --------------------------------------------------------------------------- #
# Privacy tables (also called by experiments/clamp_privacy.py)
# --------------------------------------------------------------------------- #
def write_privacy_tables(attack_rows: List[Dict], probe_rows: List[Dict], out_dir: str) -> None:
    """attack_rows: per-seed records with eps, bound, R, and the four accuracies.
    probe_rows: per-seed records with dataset, method, auc."""
    os.makedirs(out_dir, exist_ok=True)
    dss = [d for d in ["german", "bail"] if any(r["dataset"] == d for r in attack_rows)]
    if attack_rows and dss:
        eps_list = sorted({r["eps"] for r in attack_rows})
        header = ["$\\varepsilon$", "Bound"]
        for d in dss:
            header += [f"{DS_NAME[d]} single", f"{DS_NAME[d]} all"]
        out = []
        for e in eps_list:
            bound = next(r["bound"] for r in attack_rows if r["eps"] == e)
            line = [("$\\infty$" if math.isinf(e) else f"{e:g}", False), (num(bound), False)]
            for d in dss:
                sel = [r for r in attack_rows if r["dataset"] == d and r["eps"] == e]
                line += [(pm(_ms([r["acc_single_mc"] for r in sel]), "acc"), False),
                         (pm(_ms([r["acc_composed_mc"] for r in sel]), "acc"), False)]
            out.append(line)
        R = {d: sorted({r["R"] for r in attack_rows if r["dataset"] == d}) for d in dss}
        gaps = [abs(r[k] - r[ka]) for r in attack_rows
                for k, ka in (("acc_single_mc", "acc_single_analytic"),
                              ("acc_composed_mc", "acc_composed_analytic"))]
        Rtxt = ", ".join(f"$R={R[d][0]}$ for {DS_NAME[d]}" for d in dss if R[d])
        agree = math.ceil(max(gaps) * 1000) / 1000   # smallest 0.001 multiple covering the gap
        write_table(out_dir, "privacy_attack", header, out,
                    "Differencing attack on the statistics channel: balanced accuracy of the "
                    "likelihood-ratio test over random validation targets, mean$\\pm$sd over "
                    "seeds. Single: one noisy release. All: the $R$ releases of an unchanged "
                    f"vector ({Rtxt}), the worst case the theorem accounts for. Bound: "
                    "$(e^{\\varepsilon}+\\delta)/(1+e^{\\varepsilon})$. Monte Carlo and "
                    f"closed-form values agree to within {agree:.3f}.",
                    size="footnotesize")
    if probe_rows:
        pdss = [d for d in ["german", "bail"] if any(r["dataset"] == d for r in probe_rows)]
        header = ["Method"] + [f"{DS_NAME[d]} AUC" for d in pdss]
        out = []
        names = {"raw-features": "Control: raw features", "untrained-gat": "Control: untrained GAT"}
        for m in ["fedavg-gat", "clamp-clipmean", "clamp", "raw-features", "untrained-gat"]:
            line = [(DISPLAY.get(m, names.get(m, m)), False)]
            for d in pdss:
                sel = [r["auc"] for r in probe_rows if r["method"] == m and r["dataset"] == d]
                line.append((pm(_ms(sel), "auc"), False))
            out.append(line)
        write_table(out_dir, "privacy_probe", header, out,
                    "Attribute probe on the final global model: AUC of predicting the sensitive "
                    "attribute from penultimate test-node embeddings, mean$\\pm$sd over seeds.",
                    size="footnotesize")


def render_privacy_from_csv(out_dir: str) -> None:
    """Re-render the privacy tables from attack.csv and probe.csv (no re-run)."""
    def _read(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            return list(rd)

    attack = []
    for r in _read(os.path.join(out_dir, "attack.csv")):
        row = {"dataset": r["dataset"], "seed": int(r["seed"]), "eps": float(r["eps"]),
               "R": int(r["R"]), "bound": float(r["bound"]),
               "acc_single_mc": float(r["acc_single_mc"]),
               "acc_single_analytic": float(r["acc_single_analytic"]),
               "acc_composed_mc": float(r["acc_composed_mc"]),
               "acc_composed_analytic": float(r["acc_composed_analytic"]),
               "P": int(r["P"]), "sigma": float(r["sigma"]), "n_targets": int(r["n_targets"])}
        attack.append(row)
    probe = []
    p_path = os.path.join(out_dir, "probe.csv")
    if os.path.exists(p_path):
        for r in _read(p_path):
            probe.append({"dataset": r["dataset"], "seed": int(r["seed"]),
                          "method": r["method"], "auc": float(r["auc"])})
    write_privacy_tables(attack, probe, out_dir)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def report_warmup(root: str, outdir: str) -> None:
    """Count and list the runs whose feasible selection the warm-up rule changed."""
    counts: Dict[str, int] = {}
    for c in CHANGED:
        counts[c["group"]] = counts.get(c["group"], 0) + 1
    print("warm-up rule (W = ceil(0.2 T)) changed these runs:")
    for g in ["e1", "e2", "e3", "e4", "e5", "e5b", "e6"]:
        print(f"  {g}: {counts.get(g, 0)} runs")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "warmup_changes.csv"), "w", newline="") as f:
        if not CHANGED:
            f.write("note\nno runs changed\n")
            return
        w = csv.DictWriter(f, fieldnames=list(CHANGED[0].keys()))
        w.writeheader()
        w.writerows(CHANGED)
    for c in CHANGED:
        print(f"  {c['group']} {c['method']} {c['dataset']} {c['attack']} b{c['b']} s{c['seed']}: "
              f"sel {c['old_sel']} -> None; AUC {c['old_auc']:.4f} -> {c['new_auc']:.4f}; "
              f"DPD {c['old_dpd']:.4f} -> {c['new_dpd']:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(_REPO, "results", "clamp"))
    args = p.parse_args(argv)
    outdir = os.path.join(args.root, "tables")
    data = {g: load_group(args.root, g) for g in ["e1", "e2", "e3", "e4", "e5", "e5b", "e6"]}
    report_warmup(args.root, outdir)
    table_e1(data["e1"], outdir)
    write_e1_compact(data["e1"], outdir)
    write_e1_appendix(data["e1"], outdir)
    table_e2(data["e2"], outdir)
    table_e3(data["e3"], outdir)
    write_e3_compact(data["e3"], outdir)
    write_e3_appendix(data["e3"], outdir)
    table_e4(data["e4"], outdir)
    table_e5(data["e5"], outdir, datasets=["german", "bail", "nba", "facebook"],
             name="e5_privacy", default_p="P=10 for German, NBA and Facebook, P=1 for Bail")
    table_e5(data["e5"], outdir, datasets=["credit"], name="e5_credit", default_p="P=1")
    table_e5(data["e5b"], outdir, group="e5b", tau=0.005, datasets=["bail"], default_p="P=1")
    table_e6(data["e6"], outdir)
    privacy = os.path.join(args.root, "privacy")
    if os.path.exists(os.path.join(privacy, "attack.csv")):
        render_privacy_from_csv(privacy)


if __name__ == "__main__":
    main()
