"""Build every table of the CLAMP-RT manuscript from the result files.

Sources (paths relative to the repository root):
  results/clamp_rt/summary/db_e1.txt   honest runs (CLAMP-RT, its uncorrected variant, old CLAMP)
  results/clamp_rt/summary/db_e4.txt   gap attacks on CLAMP-RT
  results/clamp_rt/summary/e4_nodb.txt old CLAMP under stat_falsify, uncorrected CLAMP-RT
  results/clamp_rt/summary/db_e3.txt   update attacks on CLAMP-RT
  results/clamp_rt/summary/db_e5.txt   privacy budget x release period
  results/clamp_rt/summary/e6.txt      ablations
  results/clamp_rt/mpc_cost.json       two-party computation cost
  results/clamp/e1/runs/*.json         baselines and old CLAMP (per seed)
  results/clamp/e3/runs/*.json         update-attack baselines (per seed)
  results/clamp_rt/e1db/runs/*.json    per-seed CLAMP-RT values and channel diagnostics
  results/clamp_rt/e4db/runs/*.json    per-round order-statistic diagnostics (Theorem 3 bracket)
  src/trust/privacy.py                 noise calibration (calibrate_noise_multiplier)

Usage (from the repository root):  python3 manuscript_clamp_rt/make_tables.py
Writes manuscript_clamp_rt/tables/*.tex and tables/macros.tex.
"""
from __future__ import annotations

import collections
import glob
import importlib.util
import json
import math
import os
import re
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
SUM = os.path.join(ROOT, "results", "clamp_rt", "summary")
os.makedirs(OUT, exist_ok=True)

DS_ORDER = ["german", "bail", "credit", "pokec_z", "pokec_n", "aminer_s", "nba", "facebook"]
DS_NAME = {"german": "German", "bail": "Bail", "credit": "Credit", "pokec_z": "Pokec-z",
           "pokec_n": "Pokec-n", "aminer_s": "AMiner-S", "nba": "NBA", "facebook": "Facebook"}
ATT = ["gap_hide", "gap_inflate", "gap_inlier", "gap_zero"]
ATT_NAME = {"gap_hide": "hide", "gap_inflate": "inflate", "gap_inlier": "inlier", "gap_zero": "zero"}
TAU = 0.03
BETA = 0.05
WARMUP = 20          # ceil(0.2 T), T = 100

MACROS = {}


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
LINE = re.compile(r"^\((.*)\)\s+n\s+(\d+)\s+dpd_soft\s+([\d.]+)±([\d.]+)\s+auc\s+([\d.]+)±([\d.]+)"
                  r"(?:\s+sigma_d\s+([\d.]+)±([\d.]+))?")


def parse_summary(name):
    """(key tuple) -> dict(n, dpd=(m, s), auc=(m, s), sigma=(m, s) or None)."""
    out = {}
    with open(os.path.join(SUM, name), encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            key = eval("(" + m.group(1) + ",)")      # trusted local file of python literals
            rec = dict(n=int(m.group(2)), dpd=(float(m.group(3)), float(m.group(4))),
                       auc=(float(m.group(5)), float(m.group(6))), sigma=None)
            if m.group(7):
                rec["sigma"] = (float(m.group(7)), float(m.group(8)))
            out[key] = rec
    return out


def parse_e4_nodb():
    """(dataset, K, method, attack, f) -> dict(n, dpd, auc) (means only)."""
    out, ds = {}, None
    with open(os.path.join(SUM, "e4_nodb.txt"), encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "==":
                ds = t[1]
                continue
            K, meth, att, _, fb, _, n, _, dpd, _, auc = t
            out[(ds, int(K), meth, att, int(fb))] = dict(n=int(n), dpd=float(dpd), auc=float(auc))
    return out


def warmup_final(rec):
    """Final metrics with the warm-up rule of the output rule (selected round >= W).
    Old CLAMP runs were made before the warm-up existed; a selection below W is
    replaced by the last model (history[-1]), as in the earlier manuscript."""
    fin, cfg = rec["final"], rec["config"]
    sel = fin.get("selected_round")
    dpd = fin.get("dpd_soft", fin["dpd"])
    auc = fin["auc"]
    if cfg.get("r2_output") == "feasible" and sel is not None and sel < WARMUP:
        last = rec["history"][-1]
        dpd, auc = last.get("g_dpd_soft", last["g_dpd"]), last["g_auc"]
    return dpd, auc


def load_runs(pattern, key):
    g = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(ROOT, pattern)):
        with open(path) as f:
            rec = json.load(f)
        k = key(rec)
        if k is None:
            continue
        g[k][rec["config"]["seed"]] = rec
    return g


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def ms(v, d=3):
    return f"{v[0]:.{d}f}$\\pm${v[1]:.{d}f}"


def msl(xs, d=3):
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


def write(name, body):
    if "resizebox" not in body:
        body = body.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}")
        body = body.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)


def macro(name, value):
    MACROS[name] = value


# --------------------------------------------------------------------------- #
# Noise calibration
# --------------------------------------------------------------------------- #
def privacy_module():
    spec = importlib.util.spec_from_file_location("clamp_rt_privacy",
                                                  os.path.join(ROOT, "src", "trust", "privacy.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clamp_rt_privacy"] = mod
    spec.loader.exec_module(mod)
    return mod


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# E1
# --------------------------------------------------------------------------- #
BASELINES = [("fedavg-gat", "FedAvg-GAT"), ("fairfed", "FairFed$^{c}$"), ("f2gnn", "F$^2$GNN$^{c}$"),
             ("fairgfl", "FairGFL"), ("fedgraphfair", "FedGraph-Fair"), ("fedfact", "FedFACT$^{c}$"),
             ("puffle", "PUFFLE"), ("fedfdp", "FedFDP"), ("fairgnn", "FairGNN"), ("fairsin", "FairSIN"),
             ("clamp-clipmean", "ClipMean (no controller)"),
             ("clamp-f", "CLAMP (sums channel)"), ("clamp-gcn-f", "CLAMP-GCN (sums channel)")]
RT_ROWS = [("clamp-rt-f", "CLAMP-RT, no bias correction"),
           ("clamp-rt-db-gcn-f", "CLAMP-RT-GCN"),
           ("clamp-rt-db-f", "\\textbf{CLAMP-RT}")]


def table_e1():
    summ = parse_summary("db_e1.txt")
    old = load_runs("results/clamp/e1/runs/*.json",
                    lambda r: (r["config"]["exp_name"], r["config"]["dataset"]))
    rt = load_runs("results/clamp_rt/e1db/runs/*.json",
                   lambda r: (r["run_id"].split("__")[0], r["config"]["dataset"]))
    cells = {}
    for meth, _ in BASELINES:
        for ds in DS_ORDER:
            runs = old.get((meth, ds), {})
            if not runs:
                continue
            vals = [warmup_final(runs[s]) for s in sorted(runs)]
            cells[(meth, ds)] = (msl([v[1] for v in vals], 2), msl([v[0] for v in vals]), len(vals))
    for meth, _ in RT_ROWS:
        for ds in DS_ORDER:
            s = summ[(ds, meth)]
            cells[(meth, ds)] = (ms(s["auc"], 2), ms(s["dpd"]), s["n"])
    # consistency check: per-seed CLAMP-RT values reproduce the summary file
    for ds in DS_ORDER:
        xs = [rt[("clamp-rt-db-f", ds)][s]["final"]["dpd_soft"] for s in rt[("clamp-rt-db-f", ds)]]
        assert abs(st.mean(xs) - summ[(ds, "clamp-rt-db-f")]["dpd"][0]) < 6e-4, ds

    def panel(dss):
        head = " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in dss) + " \\\\\n"
        head += " & " + " & ".join("AUC & DPD" for _ in dss) + " \\\\\n\\midrule\n"
        rows = ""
        for meth, lab in BASELINES + RT_ROWS:
            if meth == "clamp-f":
                rows += "\\midrule\n"
            if meth == "clamp-rt-f":
                rows += "\\midrule\n"
            rows += lab + " & " + " & ".join(
                f"{cells[(meth, d)][0]} & {cells[(meth, d)][1]}" for d in dss) + " \\\\\n"
        return ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "cc" * len(dss) + "}\n\\toprule\n"
                + head + rows + "\\bottomrule\n\\end{tabular}}\n")

    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n"
            "\\caption{Honest runs ($K=5$): test AUC and soft DPD (lower is fairer), mean$\\pm$sd over 10 seeds. "
            "Baselines and CLAMP (sums channel) are the runs of the earlier study, with the output-rule warm-up "
            "applied; CLAMP used $P=1$ on Bail, Credit, Pokec-z, Pokec-n and AMiner-S and $P=10$ on German, NBA and "
            "Facebook, whereas CLAMP-RT uses $P=10$ everywhere. "
            "$^{c}$: reads per-client fairness quantities or group rates in the clear. "
            "PUFFLE and FedFDP barely learn at their privacy level (AUC near 0.5).}\n"
            "\\label{tab:e1}\n" + panel(DS_ORDER[:4]) + "\\vspace{4pt}\n" + panel(DS_ORDER[4:])
            + "\\end{table}\n")
    write("e1_main.tex", body)
    return summ, old, rt


def table_channel(summ, old, rt):
    """Per-dataset channel constants, diagnostics and paired tests CLAMP-RT vs CLAMP."""
    from scipy.stats import wilcoxon
    rows = []
    R = 10
    for ds in DS_ORDER:
        runs = rt[("clamp-rt-db-f", ds)]
        n0 = [runs[s]["final"]["n0_min"] for s in runs]
        sig = [runs[s]["final"]["sigma_d"] for s in runs]
        rel = [x for s in runs for x in runs[s]["history"] if x.get("dhat") is not None]
        frac = sum(x["dhat"] > 0 for x in rel) / len(rel)
        lamT = st.mean(runs[s]["history"][-1]["lambda"] for s in runs)
        sigm = st.median(sig)
        r_R = 2.0 * sigm * math.sqrt(math.log(2 * R / BETA))
        tau_eff = math.sqrt(TAU ** 2 + 2 * sigm ** 2)
        oldr = old[("clamp-f", ds)]
        seeds = sorted(set(oldr) & set(runs))
        a_rt = [runs[s]["final"]["dpd_soft"] for s in seeds]
        a_old = [warmup_final(oldr[s])[0] for s in seeds]
        u_rt = [runs[s]["final"]["auc"] for s in seeds]
        u_old = [warmup_final(oldr[s])[1] for s in seeds]
        p_d = wilcoxon(a_rt, a_old).pvalue
        p_a = wilcoxon(u_rt, u_old).pvalue
        rows.append((ds, min(n0), max(n0), min(sig), max(sig), tau_eff + r_R, frac, lamT,
                     st.mean(a_old), st.mean(a_rt), p_d, st.mean(u_old), st.mean(u_rt), p_a))
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Statistics channel of CLAMP-RT in the honest runs ($K=5$, $b=1$, $\\varepsilon=8$, "
            "$R=10$ releases, 10 seeds). $n_{0,\\min}$ and $\\sigma_d$: range over seeds. "
            "Radius: $\\sqrt{\\tau^2+2\\sigma_d^2}+r_R$ of Theorem~\\ref{thm:output} at the median $\\sigma_d$ "
            "and $\\beta=0.05$ (vacuous when $\\ge1$). "
            "$\\Dhat>0$: share of releases with a non-zero bias-corrected magnitude. $\\lambda^T$: mean final "
            "multiplier. Right block: mean test DPD and AUC of CLAMP (sums channel) and CLAMP-RT, with the "
            "two-sided Wilcoxon signed-rank $p$-value paired by seed (unadjusted).}\n\\label{tab:channel}\n"
            "\\begin{tabular}{lccccc|ccc|ccc}\n\\toprule\n"
            "Dataset & $n_{0,\\min}$ & $\\sigma_d$ & Radius & $\\Dhat>0$ & $\\lambda^T$ & "
            "\\multicolumn{3}{c|}{DPD: CLAMP / RT / $p$} & \\multicolumn{3}{c}{AUC: CLAMP / RT / $p$} \\\\\n\\midrule\n")
    for (ds, n0a, n0b, sa, sb, rad, frac, lamT, do, dr, pd, uo, ur, pa) in rows:
        body += (f"{DS_NAME[ds]} & {n0a}--{n0b} & {sa:.3f}--{sb:.3f} & {rad:.2f} & {frac:.2f} & {lamT:.2f} & "
                 f"{do:.3f} & {dr:.3f} & {pd:.3f} & {uo:.3f} & {ur:.3f} & {pa:.3f} \\\\\n")
        macro("rad" + DS_NAME[ds].replace("-", "").replace("Pokec", "Pokec"), f"{rad:.2f}")
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("channel.tex", body)
    return rows


# --------------------------------------------------------------------------- #
# E4: falsified statistics
# --------------------------------------------------------------------------- #
def table_e4():
    s = parse_summary("db_e4.txt")
    nodb = parse_e4_nodb()
    dss = ["bail", "credit", "pokec_z", "pokec_n"]
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Falsified gap reports against CLAMP-RT: test DPD, mean$\\pm$sd over 3 seeds, for $f$ "
            "lying clients out of $K$. Trimming level $b=1$ ($K=5$) and $b=2$ ($K=10$); rows with $f>b$ "
            "(marked $\\ast$) are beyond the breakdown point. ``Honest'' is the same configuration with no "
            "liars. AUC is in the supplementary material.}\n\\label{tab:e4}\n"
            "\\begin{tabular}{llc" + "c" * 5 + "}\n\\toprule\n"
            "Dataset & $K$ & $f$ & Honest & hide & inflate & inlier & zero \\\\\n\\midrule\n")
    for ds in dss:
        for K in (5, 10):
            b = 1 if K == 5 else 2
            fs = (1, 2) if K == 5 else (1, 2, 3)
            for i, f in enumerate(fs):
                lab = DS_NAME[ds] if (K == 5 and i == 0) else ""
                kl = str(K) if i == 0 else ""
                hon = ms(s[(ds, K, "none", 0)]["dpd"]) if i == 0 else ""
                star = "$^\\ast$" if f > b else ""
                body += (f"{lab} & {kl} & {f}{star} & {hon} & "
                         + " & ".join(ms(s[(ds, K, a, f)]["dpd"]) for a in ATT) + " \\\\\n")
        if ds != dss[-1]:
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("e4_gap.tex", body)

    # supplementary: AUC
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Falsified gap reports against CLAMP-RT: test AUC, mean$\\pm$sd over 3 seeds "
            "(companion of the DPD table in the main text). $\\ast$: $f>b$.}\n\\label{tab:s_e4auc}\n"
            "\\begin{tabular}{llc" + "c" * 5 + "}\n\\toprule\n"
            "Dataset & $K$ & $f$ & Honest & hide & inflate & inlier & zero \\\\\n\\midrule\n")
    for ds in dss:
        for K in (5, 10):
            b = 1 if K == 5 else 2
            fs = (1, 2) if K == 5 else (1, 2, 3)
            for i, f in enumerate(fs):
                lab = DS_NAME[ds] if (K == 5 and i == 0) else ""
                kl = str(K) if i == 0 else ""
                hon = ms(s[(ds, K, "none", 0)]["auc"]) if i == 0 else ""
                star = "$^\\ast$" if f > b else ""
                body += (f"{lab} & {kl} & {f}{star} & {hon} & "
                         + " & ".join(ms(s[(ds, K, a, f)]["auc"]) for a in ATT) + " \\\\\n")
        if ds != dss[-1]:
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("s_e4_auc.tex", body)

    # old vs new: worst case over attacks
    body = ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{Old sums channel versus trimmed channel under falsified statistics: test DPD, mean over "
            "3 seeds. CLAMP: swap falsification of the four group sums with the public cap ($P=1$). "
            "CLAMP-RT: largest mean DPD over the four gap attacks ($P=10$). $f=2$ at $K=5$ exceeds "
            "the breakdown point $b=1$ of CLAMP-RT.}\n\\label{tab:oldnew}\n"
            "\\begin{tabular}{llccc|ccc}\n\\toprule\n"
            " & & \\multicolumn{3}{c|}{CLAMP (sums)} & \\multicolumn{3}{c}{CLAMP-RT (trimmed)} \\\\\n"
            "Dataset & $K$ & $f=0$ & $f=1$ & $f=2$ & $f=0$ & $f=1$ & $f=2$ \\\\\n\\midrule\n")
    for ds in dss:
        for K in (5, 10):
            o = [nodb[(ds, K, "clamp-f", "none", 0)]["dpd"]] + \
                [nodb[(ds, K, "clamp-f", "stat_falsify", f)]["dpd"] for f in (1, 2)]
            n = [s[(ds, K, "none", 0)]["dpd"][0]] + \
                [max(s[(ds, K, a, f)]["dpd"][0] for a in ATT) for f in (1, 2)]
            body += (f"{DS_NAME[ds] if K == 5 else ''} & {K} & " + " & ".join(f"{x:.3f}" for x in o)
                     + " & " + " & ".join(f"{x:.3f}" for x in n) + " \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("e4_oldnew.tex", body)

    # macros for the text
    for ds in ("credit", "pokec_z", "pokec_n"):
        nm = DS_NAME[ds].replace("-", "")
        macro(f"old{nm}KtenHon", f"{nodb[(ds, 10, 'clamp-f', 'none', 0)]['dpd']:.3f}")
        macro(f"old{nm}KtenOne", f"{nodb[(ds, 10, 'clamp-f', 'stat_falsify', 1)]['dpd']:.3f}")

    # Theorem 3 bracket in the logged release rounds
    cnt = collections.Counter()
    spread = collections.defaultdict(list)
    dev = []
    for path in glob.glob(os.path.join(ROOT, "results/clamp_rt/e4db/runs/*.json")):
        with open(path) as f:
            r = json.load(f)
        c = r["config"]
        for x in r["history"]:
            if x.get("honest_lo") is None or x.get("tm_clean") is None:
                continue
            ok = x["honest_lo"] - 1e-12 <= x["tm_clean"] <= x["honest_hi"] + 1e-12
            if c["num_byzantine"] > 0:
                cnt[ok] += 1
                spread[(c["dataset"], c["num_clients"])].append(x["honest_spread"])
    macro("nTthreeRounds", f"{cnt[True]:,}".replace(",", "{,}"))
    macro("nTthreeFail", str(cnt[False]))
    allsp = [v for k in spread for v in spread[k]]
    macro("spreadMin", f"{min(st.mean(v) for v in spread.values()):.3f}")
    macro("spreadMax", f"{max(st.mean(v) for v in spread.values()):.3f}")
    macro("spreadMaxAll", f"{max(allsp):.3f}")
    body = ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n"
            "\\caption{Width $h_{(K-b)}-h_{(b-f+1)}$ of the interval of Theorem 5 of the main text in the "
            "attacked release rounds with $f\\le b$ (pre-noise honest gaps): mean and maximum.}\n"
            "\\label{tab:s_spread}\n\\begin{tabular}{lccc}\n\\toprule\nDataset & $K$ & mean & max \\\\\n\\midrule\n")
    for k in sorted(spread, key=lambda k: (dss.index(k[0]), k[1])):
        v = spread[k]
        body += f"{DS_NAME[k[0]]} & {k[1]} & {st.mean(v):.3f} & {max(v):.3f} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("s_spread.tex", body)
    return s, nodb


# --------------------------------------------------------------------------- #
# E3: update attacks
# --------------------------------------------------------------------------- #
def table_e3():
    s = parse_summary("db_e3.txt")
    base = load_runs("results/clamp/e3/runs/*.json",
                     lambda r: (r["config"]["exp_name"], r["config"]["aggregator"], r["config"]["dataset"],
                                r["config"]["attack"], r["config"]["num_byzantine"]))
    cols = [(("fedavg-gat", "fedavg"), "FedAvg"), (("fedavg-gat", "median"), "Median$^\\dagger$"),
            (("fedavg-gat", "trimmed_mean"), "TrMean$^\\dagger$"), (("fedavg-gat", "krum"), "Krum$^\\dagger$"),
            (("clamp-clipmean", "secagg_lagrangian"), "ClipMean"), (("clamp-f", "secagg_lagrangian"), "CLAMP")]
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Fairness poisoning through updates ($K=10$): test DPD (AUC in parentheses), mean over 3 "
            "seeds. Baselines and CLAMP (sums channel, $P=1$) are from the earlier study; $^\\dagger$ needs "
            "individual updates and cannot run under secure aggregation. CLAMP-RT uses $P=10$.}\n"
            "\\label{tab:e3}\n\\begin{tabular}{lc" + "c" * (len(cols) + 1) + "}\n\\toprule\n"
            "Dataset & $f$ & " + " & ".join(c[1] for c in cols) + " & CLAMP-RT \\\\\n\\midrule\n")
    for ds in ("pokec_z", "bail"):
        for f in (0, 1, 2, 3, 4):
            att = "none" if f == 0 else "fairness_poison"
            cells = []
            for (meth, agg), _ in cols:
                runs = base.get((meth, agg, ds, att, f), {})
                vals = [warmup_final(runs[k]) for k in runs]
                cells.append(f"{st.mean(v[0] for v in vals):.3f} ({st.mean(v[1] for v in vals):.2f})"
                             if vals else "--")
            rt = s[(ds, att, f)]
            cells.append(f"{rt['dpd'][0]:.3f} ({rt['auc'][0]:.2f})")
            body += f"{DS_NAME[ds] if f == 0 else ''} & {f} & " + " & ".join(cells) + " \\\\\n"
        if ds == "pokec_z":
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("e3_poison.tex", body)

    # full CLAMP-RT table (supplement)
    atts = ["alie", "fairness_poison", "ipm", "scaling", "sign_flip"]
    an = {"alie": "ALIE", "fairness_poison": "Fair.\\ poison", "ipm": "IPM", "scaling": "Scaling",
          "sign_flip": "Sign flip"}
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{CLAMP-RT under update attacks ($K=10$, 3 seeds): test DPD / AUC, mean$\\pm$sd.}\n"
            "\\label{tab:s_e3}\n\\begin{tabular}{llcccc}\n\\toprule\nDataset & Attack & $f=1$ & $f=2$ & $f=3$ & $f=4$ \\\\\n\\midrule\n")
    for ds in ("bail", "pokec_z"):
        h = s[(ds, "none", 0)]
        body += f"{DS_NAME[ds]} & none ($f=0$) & \\multicolumn{{4}}{{l}}{{DPD {ms(h['dpd'])}, AUC {ms(h['auc'])}}} \\\\\n"
        for a in atts:
            body += (f" & {an[a]} & " + " & ".join(
                f"{ms(s[(ds, a, f)]['dpd'])} / {s[(ds, a, f)]['auc'][0]:.2f}" for f in (1, 2, 3, 4)) + " \\\\\n")
        if ds == "bail":
            body += "\\midrule\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("s_e3_full.tex", body)


# --------------------------------------------------------------------------- #
# E5, E6
# --------------------------------------------------------------------------- #
def table_e5():
    s = parse_summary("db_e5.txt")
    dss = ["credit", "pokec_z", "pokec_n"]
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Privacy budget $\\varepsilon$ of the gap channel and release period $P$ ($K=5$, 3 seeds): "
            "test DPD, mean$\\pm$sd. All other experiments use $\\varepsilon=8$ and $P=10$.}\n\\label{tab:e5}\n"
            "\\begin{tabular}{c" + "ccc" * len(dss) + "}\n\\toprule\n"
            " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS_NAME[d]}}}" for d in dss) + " \\\\\n"
            "$\\varepsilon$ & " + " & ".join("$P=1$ & $P=5$ & $P=10$" for _ in dss) + " \\\\\n\\midrule\n")
    for eps in (1.0, 2.0, 4.0, 8.0):
        body += f"{int(eps)} & " + " & ".join(ms(s[(d, eps, P)]["dpd"]) for d in dss for P in (1, 5, 10)) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("e5_privacy.tex", body)
    aucs = [v["auc"][0] for v in s.values()]
    macro("eFiveAucMin", f"{min(aucs):.3f}")
    macro("eFiveAucMax", f"{max(aucs):.3f}")
    # supplement AUC
    body = ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Privacy sweep: test AUC, mean$\\pm$sd over 3 seeds.}\n\\label{tab:s_e5auc}\n"
            "\\begin{tabular}{c" + "ccc" * len(dss) + "}\n\\toprule\n"
            " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS_NAME[d]}}}" for d in dss) + " \\\\\n"
            "$\\varepsilon$ & " + " & ".join("$P=1$ & $P=5$ & $P=10$" for _ in dss) + " \\\\\n\\midrule\n")
    for eps in (1.0, 2.0, 4.0, 8.0):
        body += f"{int(eps)} & " + " & ".join(ms(s[(d, eps, P)]["auc"]) for d in dss for P in (1, 5, 10)) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("s_e5_auc.tex", body)


def table_e6():
    s = parse_summary("e6.txt")
    rows = [(("clamp-rt-f", -1, 0.05), "Default ($b=1$, $c=0.05$)"),
            (("clamp-rt", -1, 0.05), "Last iterate (no output rule)"),
            (("clamp-rt-b0-f", 0, 0.05), "$b=0$ (plain mean, no robustness)"),
            (("clamp-rt-f", 2, 0.05), "$b=2$ (median of $K=5$)"),
            (("clamp-rt-f", -1, 0.2), "$c=0.2$")]
    body = ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\caption{Ablations of the trimmed channel ($K=5$, $\\varepsilon=8$, $P=10$, 3 seeds, honest). "
            "These runs use the variant without bias correction. $\\sigma_d$ is the per-server noise "
            "standard deviation (mean$\\pm$sd over seeds).}\n\\label{tab:e6}\n"
            "\\begin{tabular}{l ccc ccc}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Credit} & \\multicolumn{3}{c}{Pokec-z} \\\\\n"
            "Variant & AUC & DPD & $\\sigma_d$ & AUC & DPD & $\\sigma_d$ \\\\\n\\midrule\n")
    for (m, b, c), lab in rows:
        cells = []
        for ds in ("credit", "pokec_z"):
            r = s[(ds, m, b, c)]
            cells.append(f"{ms(r['auc'], 3)} & {ms(r['dpd'])} & {ms(r['sigma'])}")
        body += lab + " & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("e6_ablation.tex", body)


# --------------------------------------------------------------------------- #
# MPC cost and differencing attack
# --------------------------------------------------------------------------- #
def table_mpc():
    with open(os.path.join(ROOT, "results/clamp_rt/mpc_cost.json")) as f:
        rows = json.load(f)
    body = ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\caption{Measured cost of one release round of the server-side protocol in semi-honest two-party "
            "computation (CrypTen, two processes on one machine, local network; median of 3 repetitions). "
            "Bytes are those sent by one server as reported by CrypTen; rounds are communication rounds. "
            "Update channel: norm checks and masked sum of $K$ updates of dimension $d$. Gap channel: range "
            "checks, sort, trimmed mean and noise addition. Err: largest absolute fixed-point error of the "
            "trimmed mean (noise switched off). Client-to-server input sharing is not included.}\n"
            "\\label{tab:mpc}\n\\begin{tabular}{rrr rrr rrrr}\n\\toprule\n"
            " & & & \\multicolumn{3}{c}{Update channel} & \\multicolumn{4}{c}{Gap channel} \\\\\n"
            "$K$ & $b$ & $d$ & MB & rounds & s & KB & rounds & s & Err \\\\\n\\midrule\n")
    for r in rows:
        body += (f"{r['K']} & {r['b']} & $10^{{{int(round(math.log10(r['d'])))}}}$ & {r['bytes_update'] / 1e6:.2f} & "
                 f"{r['rounds_update']} & {r['t_update_s']:.2f} & {r['bytes_gap'] / 1e3:.1f} & {r['rounds_gap']} & "
                 f"{r['t_gap_s']:.2f} & {r['max_abs_err_tm']:.1e} \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    body = body.replace("e-06", "$\\times10^{-6}$").replace("e-05", "$\\times10^{-5}$")
    write("mpc_cost.tex", body)
    macro("mpcErrMax", f"{max(r['max_abs_err_tm'] for r in rows) * 1e5:.1f}")


def table_privacy_attack():
    p = privacy_module()
    R = 10
    body = ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n"
            "\\caption{Balanced accuracy of the optimal test that distinguishes two attribute-neighbouring "
            "datasets from the released gap (closed form, worst case $|\\Delta|=\\Delta_{\\mathrm{TM}}$; $R=10$ "
            "releases, $\\delta=10^{-5}$). $z=\\sigma_d/\\Delta_{\\mathrm{TM}}$ from the RDP calibration. "
            "Both servers honest: one release with variance $2\\sigma_d^2$. One server colluding: it removes its "
            "own noise. All releases: the colluding server combines all $R$ releases of a target whose "
            "contribution is unchanged. Bound: $(e^{\\varepsilon}+\\delta)/(1+e^{\\varepsilon})$.}\n"
            "\\label{tab:attack}\n\\begin{tabular}{ccccccc}\n\\toprule\n"
            "$\\varepsilon$ & $z$ & Both honest, one release & One server, one release & One server, all $R$ & Bound \\\\\n\\midrule\n")
    for eps in (1, 2, 4, 8):
        z = p.calibrate_noise_multiplier(eps, steps=R, delta=1e-5)
        a2 = Phi(1.0 / (2.0 * math.sqrt(2.0) * z))
        a1 = Phi(1.0 / (2.0 * z))
        aR = Phi(math.sqrt(R) / (2.0 * z))
        bd = (math.exp(eps) + 1e-5) / (1.0 + math.exp(eps))
        body += f"{eps} & {z:.2f} & {a2:.3f} & {a1:.3f} & {aR:.3f} & {bd:.4f} \\\\\n"
        if eps == 8:
            macro("zEight", f"{z:.2f}")
            macro("accBothEight", f"{a2:.3f}")
            macro("accOneEight", f"{a1:.3f}")
            macro("accAllEight", f"{aR:.3f}")
            macro("boundEight", f"{bd:.4f}")
    body = body.replace("& Bound", "& Bound").replace("\\begin{tabular}{ccccccc}", "\\begin{tabular}{cccccc}")
    body += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    write("privacy_attack.tex", body)


def main():
    summ, old, rt = table_e1()
    table_channel(summ, old, rt)
    table_e4()
    table_e3()
    table_e5()
    table_e6()
    table_mpc()
    table_privacy_attack()
    with open(os.path.join(OUT, "macros.tex"), "w") as f:
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
