"""Tables and figures for the CLAMP paper, from results/clamp/<group>/runs/*.json.

Writes CSV and LaTeX (booktabs) tables, plus the tau-Pareto figure, into
results/clamp/tables/. A missing group produces a note instead of a failure.

Usage:
    python -m experiments.make_clamp_tables [--root results/clamp]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from experiments.methods import METHODS
from experiments.run_clamp_suite import E1_METHODS, E3_ATTACKS, E3_ROWS, MAIN

# Display names for the E1 table and the ablation tables.
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
TAU_REPORT = 0.03        # budget used for the "DPD <= tau (seeds)" count
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_group(root: str, group: str) -> List[Dict]:
    rows = []
    for path in glob.glob(os.path.join(root, group, "runs", "*.json")):
        with open(path) as f:
            rec = json.load(f)
        cfg, final, hist = rec["config"], rec.get("final", {}), rec.get("history", [])
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
            eod=final.get("eod", math.nan), cert=cert, hist=hist,
            selected=final.get("selected_round"), cfg=cfg,
            byz=rec.get("byzantine_ids", [])))
    return rows


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
# Writers
# --------------------------------------------------------------------------- #
def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(outdir: str, name: str, header: List[str], rows: List[List[tuple]],
                caption: str) -> None:
    """rows: list of rows; each cell is (text, bold)."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.csv"), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join('"' + c[0] + '"' for c in r) + "\n")
    cols = "l" * len(header)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}",
             f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_tex_escape(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        cells = [(f"\\textbf{{{_tex_escape(t)}}}" if b else _tex_escape(t)) for t, b in r]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", f"\\label{{tab:{name}}}", r"\end{table}", ""]
    with open(os.path.join(outdir, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {name}.csv / {name}.tex ({len(rows)} rows)")


def _fmt(ms, digits=3) -> str:
    if ms is None:
        return "--"
    return f"{ms[0]:.{digits}f} ± {ms[1]:.{digits}f}"


def _missing(outdir: str, name: str, group: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.csv"), "w") as f:
        f.write(f"note\nno results for group {group}\n")
    print(f"{name}: no results for group {group}; wrote a placeholder")


# --------------------------------------------------------------------------- #
# E1: main comparison
# --------------------------------------------------------------------------- #
def _e1_stats(rows: List[Dict]) -> Dict:
    """Per dataset: per-method mean/std, seed count with DPD <= TAU_REPORT, and the
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
    """Main comparison. The Wilcoxon/Holm baseline is the feasible-output CLAMP (MAIN)."""
    rows = [r for r in rows if r["method"] in E1_METHODS]
    stats = _e1_stats(rows)
    header = ["dataset", "method", "AUC", "DPD", "EOD", "selected round (mean)",
              "p_AUC (vs CLAMP)", "p_DPD (vs CLAMP)", "certificate",
              f"DPD <= {TAU_REPORT:g} (seeds)"]
    out = []
    for ds, per in stats.items():
        best = {k: (max if k == "auc" else min)(
            (per[m][k][0] for m in per if per[m][k] is not None), default=None)
            for k in ("auc", "dpd", "eod")}
        for m, st in per.items():
            cell = lambda k: (_fmt(st[k]), st[k] is not None and best[k] is not None
                              and abs(st[k][0] - best[k]) < 1e-12)
            if METHODS.get(m, {}).get("r2_output") == "feasible":
                picked = [s for s in st["selected"] if s is not None]
                sel_txt = (f"{np.mean(picked):.1f} ({len(picked)}/{len(st['selected'])})"
                           if picked else "fallback (no feasible round)")
            else:
                sel_txt = "--"
            pa = "--" if m == MAIN else (f"{st['p_auc']:.3g}" if st["p_auc"] == st["p_auc"] else "--")
            pd = "--" if m == MAIN else (f"{st['p_dpd']:.3g}" if st["p_dpd"] == st["p_dpd"] else "--")
            out.append([(DS_NAME[ds], False), (DISPLAY.get(m, m), False), cell("auc"),
                        cell("dpd"), cell("eod"), (sel_txt, False), (pa, False), (pd, False),
                        (_fmt(st["cert"], 4) if st["cert"] else "--", False),
                        (f"{st['tau_ok']}/{st['n']}", False)])
    if not out:
        return _missing(outdir, "e1_main", "e1")
    write_table(outdir, "e1_main", header, out,
                "Main comparison (mean $\\pm$ sd over seeds). Best value per dataset in bold. "
                "Selected round: mean over seeds with a feasible release, (feasible seeds / total). "
                "p-values: Wilcoxon signed-rank vs CLAMP, paired by seed, Holm within dataset.")


def write_e1_compact(rows: List[Dict], outdir: str) -> None:
    """Paper version: two side-by-side panels of four datasets, AUC and DPD per dataset,
    a dagger where the Holm-adjusted Wilcoxon p-value against CLAMP is below 0.05."""
    rows = [r for r in rows if r["method"] in E1_METHODS]
    stats = _e1_stats(rows)
    if not stats:
        return _missing(outdir, "e1_main_compact", "e1")
    os.makedirs(outdir, exist_ok=True)

    def val(ds, m, k):
        st = stats.get(ds, {}).get(m)
        if st is None or st[k] is None:
            return None
        return st[k]

    def cell(ds, m, k, best):
        v = val(ds, m, k)
        if v is None:
            return "--"
        mean, sd, _ = v
        txt = f"{mean:.2f}$\\pm${sd:.2f}"
        if best.get((ds, k)) is not None and abs(mean - best[(ds, k)]) < 1e-12:
            txt = f"\\textbf{{{txt}}}"
        p = stats[ds][m]["p_auc" if k == "auc" else "p_dpd"] if m in stats[ds] else math.nan
        if m != MAIN and p == p and p < 0.05:
            txt += "$^\\dagger$"
        return txt

    best = {}
    for ds in DATASETS:
        for k in ("auc", "dpd"):
            vals = [val(ds, m, k)[0] for m in COMPACT_METHODS if val(ds, m, k) is not None]
            if vals:
                best[(ds, k)] = max(vals) if k == "auc" else min(vals)

    def panel(dss):
        lines = [r"\begin{tabular}{l" + " cc" * len(dss) + "}", r"\toprule",
                 " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in dss)
                 + r" \\", " & " + " & ".join("AUC & DPD" for _ in dss) + r" \\",
                 r"\midrule"]
        for m in COMPACT_METHODS:
            name = DISPLAY.get(m, m)
            if m == MAIN:
                name = f"\\textbf{{{name}}}"
            cells = []
            for d in dss:
                cells += [cell(d, m, "auc", best), cell(d, m, "dpd", best)]
            lines.append(name + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    body = "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Main results: test AUC and DPD (mean $\pm$ sd over seeds). Best per column in "
        r"bold. $^\dagger$: Holm-adjusted Wilcoxon signed-rank $p<0.05$ against CLAMP, paired "
        r"by seed.}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{minipage}[t]{0.49\textwidth}", r"\centering", panel(DATASETS[:4]),
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}", r"\centering", panel(DATASETS[4:]),
        r"\end{minipage}",
        r"\label{tab:e1_main_compact}", r"\end{table*}", ""])
    with open(os.path.join(outdir, "e1_main_compact.tex"), "w") as f:
        f.write(body)
    print("wrote e1_main_compact.tex")


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
        ax.set_title(ds)
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
    header = ["dataset", "tau", "AUC", "DPD", "EOD", "n"]
    out = []
    for ds in ["bail", "credit", "pokec_z"]:
        for tau in sorted({r["tau"] for r in rows if r["dataset"] == ds}):
            sel = [r for r in rows if r["dataset"] == ds and r["tau"] == tau]
            cells = [_ms([r[k] for r in sel]) for k in ("auc", "dpd", "eod")]
            out.append([(ds, False), (f"{tau:g}", False), (_fmt(cells[0]), False),
                        (_fmt(cells[1]), False), (_fmt(cells[2]), False),
                        (str(len(sel)), False)])
    write_table(outdir, "e2_tau", header, out,
                "Fairness budget sweep for CLAMP (mean $\\pm$ sd over seeds).")
    figure_e2(rows, outdir)


# --------------------------------------------------------------------------- #
# E3: update attacks
# --------------------------------------------------------------------------- #
def _label(r: Dict) -> str:
    own = METHODS.get(r["method"], {}).get("aggregator")
    return r["method"] if r["aggregator"] == own else f"{r['method']}[{r['aggregator']}]"


def table_e3(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e3_attacks", "e3")
    header = ["dataset", "attack", "aggregator", "b", "AUC", "DPD"]
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
                    out.append([(ds, False), (attack, False), (lab, False), (str(b), False),
                                (_fmt(_ms([r["auc"] for r in sel])), False),
                                (_fmt(_ms([r["dpd"] for r in sel])), False)])
    write_table(outdir, "e3_attacks", header, out,
                "Update attacks: utility and fairness versus number of Byzantine clients "
                "(K = 10, mean $\\pm$ sd over seeds).")


# --------------------------------------------------------------------------- #
# E4: statistics attacks and the T3 bound
# --------------------------------------------------------------------------- #
_PART_CACHE: Dict = {}


def _honest_counts(cfg: Dict, byz: List[int]) -> np.ndarray:
    """N^H_a = validation counts per group, summed over the honest clients.

    These counts do not change across rounds (they depend only on the partition), so
    they are recomputed here from the same partition the trainer uses instead of
    being read from the round logs, which do not record them. The trainer's per-client
    cap r2_n_cap is derived from the same counts, so it never cuts an honest group.
    """
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
        _PART_CACHE[key] = np.array(
            [[int((d.sensitive_attr[d.val_mask] == a).sum()) for a in (0, 1)] for d in parts])
    counts = _PART_CACHE[key]
    honest = [k for k in range(len(counts)) if k not in set(byz)]
    return counts[honest].sum(0).astype(float)


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


def _noise_term(sigma_s: float, n_h: float, beta: float = 0.05) -> float:
    """Per-group noise contribution 2e / (N^H_a - e), with e = sigma_s sqrt(2 ln(8/beta)),
    so that |S/N - S^H/N^H| from the statistics noise is covered with probability 1-beta."""
    if sigma_s <= 0:
        return 0.0
    e = sigma_s * math.sqrt(2.0 * math.log(8.0 / beta))
    return 2.0 * e / (n_h - e) if n_h > e else math.inf


def table_e4(rows: List[Dict], outdir: str) -> None:
    """Deviation of the server estimate from the honest-only gap, against the full
    Theorem 3 bound: b_0 + b_1 (the logged t3_bound) plus the statistics-noise terms.
    The b = 0 row therefore reports the noise term alone."""
    rows = [r for r in rows if r["method"] == MAIN]
    if not rows:
        return _missing(outdir, "e4_stat_attacks", "e4")
    header = ["dataset", "attack", "b", "runs", "rounds checked", "max_t |dhat-D^H|",
              "max_t bound (b-term + noise)", "rounds violating bound", "AUC", "DPD"]
    out = []
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
                    nh = _honest_counts(r["cfg"], r["byz"])
                    sig = _sigma_s(r["cfg"])
                    noise = sum(_noise_term(sig, nh[a]) for a in (0, 1))
                    d = [abs(h["dhat"] - h["dhat_honest"]) for h in rr]
                    full = [h["t3_bound"] + noise for h in rr]
                    devs.append(max(d)); bounds.append(max(full))
                    viol += sum(1 for x, y in zip(d, full) if x > y + 1e-12)
                    checked += len(rr)
                out.append([(ds, False), (attack, False), (str(b), False),
                            (str(len(sel)), False), (str(checked), False),
                            (f"{np.mean(devs):.3f}" if devs else "--", False),
                            (f"{np.mean(bounds):.3f}" if bounds else "--", False),
                            (str(viol), False),
                            (_fmt(_ms([r["auc"] for r in sel])), False),
                            (_fmt(_ms([r["dpd"] for r in sel])), False)])
    write_table(outdir, "e4_stat_attacks", header, out,
                "Statistics attacks: deviation of the server's estimate from the honest-only "
                "gap, against the Theorem 3 bound plus the statistics-noise term (beta = 0.05). "
                "Violation counts are over all checked rounds and seeds.")


# --------------------------------------------------------------------------- #
# E5: privacy
# --------------------------------------------------------------------------- #
def table_e5(rows: List[Dict], outdir: str, group: str = "e5", tau: Optional[float] = None,
             datasets: Optional[List[str]] = None) -> None:
    """E5 (privacy sweep). ``group`` = e5b reuses it at a binding fairness budget ``tau``,
    and adds the unconstrained clamp-clipmean reference as its own row."""
    name = "e5_privacy" if group == "e5" else f"{group}_privacy"
    if tau is not None:
        rows = [r for r in rows if r["tau"] == tau]
    if not rows:
        return _missing(outdir, name, group)
    header = ["dataset", "epsilon", "P", "stat DP", "AUC", "DPD", "n"]
    out = []
    for ds in (datasets or ["german", "bail"]):
        for eps in sorted({r["eps"] for r in rows if r["dataset"] == ds}):
            for P in sorted({r["P"] for r in rows if r["dataset"] == ds}):
                sel = [r for r in rows if r["dataset"] == ds and r["eps"] == eps
                       and r["P"] == P and r["stat_dp"]]
                sel = [r for r in sel if r["method"] == MAIN]
                if not sel:
                    continue
                out.append([(ds, False), (f"{eps:g}", False), (str(P), False), ("on", False),
                            (_fmt(_ms([r["auc"] for r in sel])), False),
                            (_fmt(_ms([r["dpd"] for r in sel])), False), (str(len(sel)), False)])
        off = [r for r in rows if r["dataset"] == ds and not r["stat_dp"]
               and r["method"] == MAIN]
        if off:
            out.append([(ds, False), ("off", False), (str(off[0]["P"]), False), ("off", False),
                        (_fmt(_ms([r["auc"] for r in off])), False),
                        (_fmt(_ms([r["dpd"] for r in off])), False),
                        (str(len(off)), False)])
        ref = [r for r in rows if r["dataset"] == ds and r["method"] == "clamp-clipmean"]
        if ref and group == "e5b":
            out.append([(ds, False), ("unconstrained", False), (str(ref[0]["P"]), False),
                        ("clamp-clipmean", False),
                        (_fmt(_ms([r["auc"] for r in ref])), False),
                        (_fmt(_ms([r["dpd"] for r in ref])), False), (str(len(ref)), False)])
    write_table(outdir, name, header, out,
                "Privacy: utility and fairness versus statistics budget epsilon and release "
                "period P (mean $\\pm$ sd over seeds)"
                + (f", fairness budget tau={tau:g}." if tau is not None else "."))


# --------------------------------------------------------------------------- #
# E6: ablations
# --------------------------------------------------------------------------- #
def table_e6(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e6_ablation", "e6")
    header = ["dataset", "method", "clip C", "step eta", "update noise", "AUC", "DPD", "EOD", "n"]
    out = []
    for ds in ["bail", "credit"]:
        keys = sorted({(r["method"], r["clip"], r["step"], r["noise"]) for r in rows
                       if r["dataset"] == ds})
        for m, c, s, n in keys:
            sel = [r for r in rows if r["dataset"] == ds and r["method"] == m
                   and (r["clip"], r["step"], r["noise"]) == (c, s, n)]
            out.append([(ds, False), (DISPLAY.get(m, m), False), (f"{c:g}", False),
                        (f"{s:g}", False), (f"{n:g}", False),
                        (_fmt(_ms([r["auc"] for r in sel])), False),
                        (_fmt(_ms([r["dpd"] for r in sel])), False),
                        (_fmt(_ms([r["eod"] for r in sel])), False), (str(len(sel)), False)])
    write_table(outdir, "e6_ablation", header, out,
                "CLAMP ablations, one factor at a time around clip 1, step 0.5, noise 0 "
                "(mean $\\pm$ sd over seeds). The last two rows at the default setting are the "
                "output-rule (last iterate) and direction-rule (proportional) ablations.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join("results", "clamp"))
    args = p.parse_args(argv)
    outdir = os.path.join(args.root, "tables")
    data = {g: load_group(args.root, g) for g in ["e1", "e2", "e3", "e4", "e5", "e5b", "e6"]}
    table_e1(data["e1"], outdir)
    write_e1_compact(data["e1"], outdir)
    table_e2(data["e2"], outdir)
    table_e3(data["e3"], outdir)
    table_e4(data["e4"], outdir)
    table_e5(data["e5"], outdir)
    table_e5(data["e5b"], outdir, group="e5b", tau=0.005, datasets=["bail"])
    table_e6(data["e6"], outdir)


if __name__ == "__main__":
    main()
