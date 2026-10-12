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
from experiments.run_clamp_suite import E1_METHODS, E3_ATTACKS, E3_ROWS

DATASETS = ["german", "bail", "credit", "pokec_z"]


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
            eod=final.get("eod", math.nan), cert=cert, hist=hist))
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
def table_e1(rows: List[Dict], outdir: str) -> None:
    rows = [r for r in rows if r["method"] in E1_METHODS]
    header = ["dataset", "method", "AUC", "DPD", "EOD", "p_AUC (vs clamp)",
              "p_DPD (vs clamp)", "certificate"]
    out = []
    for ds in DATASETS:
        dsr = [r for r in rows if r["dataset"] == ds]
        if not dsr:
            continue
        methods = [m for m in E1_METHODS if any(r["method"] == m for r in dsr)]
        others = [m for m in methods if m != "clamp"]
        stats = {}
        for m in methods:
            stats[m] = {k: _ms([r[k] for r in dsr if r["method"] == m]) for k in ("auc", "dpd", "eod")}
        p_auc = {m: _wilcoxon(_series(dsr, "auc", method=m), _series(dsr, "auc", method="clamp"))
                 for m in others}
        p_dpd = {m: _wilcoxon(_series(dsr, "dpd", method=m), _series(dsr, "dpd", method="clamp"))
                 for m in others}
        h_auc = dict(zip(others, holm([p_auc[m] for m in others])))
        h_dpd = dict(zip(others, holm([p_dpd[m] for m in others])))
        best = {k: (max if k == "auc" else min)(
            (stats[m][k][0] for m in methods if stats[m][k] is not None), default=None)
            for k in ("auc", "dpd", "eod")}
        for m in methods:
            cert = [r["cert"] for r in dsr if r["method"] == m]
            cert_ms = _ms(cert)
            cell = lambda k: (_fmt(stats[m][k]),
                              stats[m][k] is not None and best[k] is not None
                              and abs(stats[m][k][0] - best[k]) < 1e-12)
            if m == "clamp":
                pa, pd = "--", "--"
            else:
                pa = f"{h_auc[m]:.3g}" if h_auc[m] == h_auc[m] else "--"
                pd = f"{h_dpd[m]:.3g}" if h_dpd[m] == h_dpd[m] else "--"
            out.append([(ds, False), (m, False), cell("auc"), cell("dpd"), cell("eod"),
                        (pa, False), (pd, False),
                        (_fmt(cert_ms, 4) if cert_ms else "--", False)])
    if not out:
        return _missing(outdir, "e1_main", "e1")
    write_table(outdir, "e1_main", header, out,
                "Main comparison (mean $\\pm$ sd over seeds). Best value per dataset in bold. "
                "p-values: Wilcoxon signed-rank vs CLAMP, paired by seed, Holm within dataset.")


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
def table_e4(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e4_stat_attacks", "e4")
    header = ["dataset", "attack", "b", "max_t |dhat-D^H|", "max_t T3 bound",
              "rounds violating bound", "AUC", "DPD"]
    out = []
    for ds in ["bail", "pokec_z", "credit"]:
        for attack in ["none", "stat_falsify", "stat_falsify_mid"]:
            for b in ([0] if attack == "none" else [1, 2, 3, 4]):
                sel = [r for r in rows if r["dataset"] == ds and r["attack"] == attack
                       and r["b"] == b]
                if not sel:
                    continue
                devs, bounds, viol = [], [], 0
                for r in sel:
                    rr = [h for h in r["hist"] if h.get("dhat") is not None
                          and h.get("dhat_honest") is not None]
                    if not rr:
                        continue
                    d = [abs(h["dhat"] - h["dhat_honest"]) for h in rr]
                    tb = [h["t3_bound"] for h in rr]
                    devs.append(max(d)); bounds.append(max(tb))
                    viol += sum(1 for x, y in zip(d, tb) if x > y + 1e-12)
                out.append([(ds, False), (attack, False), (str(b), False),
                            (f"{np.mean(devs):.3f}" if devs else "--", False),
                            (f"{np.mean(bounds):.3f}" if bounds else "--", False),
                            (str(viol), False),
                            (_fmt(_ms([r["auc"] for r in sel])), False),
                            (_fmt(_ms([r["dpd"] for r in sel])), False)])
    write_table(outdir, "e4_stat_attacks", header, out,
                "Statistics attacks. Deviation of the server's estimate from the honest-only "
                "gap, against the T3 bound, per run and round. Note: the server estimate "
                "includes the statistics noise, while the bound is noise-free.")


# --------------------------------------------------------------------------- #
# E5: privacy
# --------------------------------------------------------------------------- #
def table_e5(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e5_privacy", "e5")
    header = ["dataset", "epsilon", "P", "stat DP", "AUC", "DPD", "n"]
    out = []
    for ds in ["german", "bail"]:
        for eps in sorted({r["eps"] for r in rows if r["dataset"] == ds}):
            for P in sorted({r["P"] for r in rows if r["dataset"] == ds}):
                sel = [r for r in rows if r["dataset"] == ds and r["eps"] == eps
                       and r["P"] == P and r["stat_dp"]]
                if not sel:
                    continue
                out.append([(ds, False), (f"{eps:g}", False), (str(P), False), ("on", False),
                            (_fmt(_ms([r["auc"] for r in sel])), False),
                            (_fmt(_ms([r["dpd"] for r in sel])), False), (str(len(sel)), False)])
        off = [r for r in rows if r["dataset"] == ds and not r["stat_dp"]]
        if off:
            out.append([(ds, False), ("off", False), (str(off[0]["P"]), False), ("off", False),
                        (_fmt(_ms([r["auc"] for r in off])), False),
                        (_fmt(_ms([r["dpd"] for r in off])), False),
                        (str(len(off)), False)])
    write_table(outdir, "e5_privacy", header, out,
                "Privacy: utility and fairness versus statistics budget epsilon and release "
                "period P (mean $\\pm$ sd over seeds).")


# --------------------------------------------------------------------------- #
# E6: ablations
# --------------------------------------------------------------------------- #
def table_e6(rows: List[Dict], outdir: str) -> None:
    if not rows:
        return _missing(outdir, "e6_ablation", "e6")
    header = ["dataset", "clip C", "step eta", "update noise", "AUC", "DPD", "EOD", "n"]
    out = []
    for ds in ["bail", "credit"]:
        keys = sorted({(r["clip"], r["step"], r["noise"]) for r in rows if r["dataset"] == ds})
        for c, s, n in keys:
            sel = [r for r in rows if r["dataset"] == ds and (r["clip"], r["step"], r["noise"]) == (c, s, n)]
            out.append([(ds, False), (f"{c:g}", False), (f"{s:g}", False), (f"{n:g}", False),
                        (_fmt(_ms([r["auc"] for r in sel])), False),
                        (_fmt(_ms([r["dpd"] for r in sel])), False),
                        (_fmt(_ms([r["eod"] for r in sel])), False), (str(len(sel)), False)])
    write_table(outdir, "e6_ablation", header, out,
                "CLAMP ablations, one factor at a time around clip 1, step 0.5, noise 0 "
                "(mean $\\pm$ sd over seeds).")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join("results", "clamp"))
    args = p.parse_args(argv)
    outdir = os.path.join(args.root, "tables")
    data = {g: load_group(args.root, g) for g in ["e1", "e2", "e3", "e4", "e5", "e6"]}
    table_e1(data["e1"], outdir)
    table_e2(data["e2"], outdir)
    table_e3(data["e3"], outdir)
    table_e4(data["e4"], outdir)
    table_e5(data["e5"], outdir)
    table_e6(data["e6"], outdir)


if __name__ == "__main__":
    main()
