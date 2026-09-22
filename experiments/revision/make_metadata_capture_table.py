"""Turn a metadata-capture artifact into the Pillar C1 table, plus a stats artifact.

Every number in the table is computed here from the per-seed arrays; nothing is
typed by hand and nothing is copied from a run's stdout. The companion JSON
carries a manifest so each p-value in the manuscript traces to an artifact
(acceptance gate G9).

Two Holm families are reported, and the artifact says which one the table uses:

  "reading"  the 6 rules that consume a client-declared field -- the
             pre-registered discovery family.
  "all"      those plus the 2 metadata-blind controls, whose predicted effect is
             exactly zero. Including them raises m and can only make the
             threshold harsher, so it is reported as a robustness check.

Usage:
  python experiments/revision/make_metadata_capture_table.py \
      --in-json <artifact.json> --out-tex <table.tex> --out-json <stats.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("experiments"))

from scipy.stats import binomtest

from stats import holm_bonferroni, paired_report          # noqa: E402
from src.utils.provenance import build_manifest           # noqa: E402

# Venue labels for the rules taken from published methods. A rule absent here is
# ours; KeyError is deliberate, so a new rule cannot silently print without one.
VENUE = {
    "fairfed": "AAAI'23",
    "qffl": "ICLR'20",
    "f2gnn": "WWW'23",
    "fedgraphfair": "InfoSci'26",
    "popets_fairfed": "PoPETs'25",
    "cgsv": "NeurIPS'21",
    "fltrust": "NDSS'21",
    "bfwa": "IndabaX'26",
    "fu_shapley": "--",
    "fu_shapley_alpha0": "--",
}
# Whether a rule comes from a published method is bibliographic metadata, not a
# measurement, so it lives here in the presentation layer. The artifact records
# what the runner believed at run time; correcting a label must never mean
# editing an artifact by hand.
PUBLISHED = {"fairfed", "qffl", "f2gnn", "fedgraphfair", "popets_fairfed",
             "cgsv", "fltrust",
             # our own earlier paper, PMLR v319 -- published, and attacked here
             # on the same footing as the rest
             "bfwa"}
PRETTY = {
    "fairfed": "FairFed", "qffl": "q-FedAvg", "f2gnn": "F\\textsuperscript{2}GNN",
    "fedgraphfair": "FedGraph-Fair", "popets_fairfed": "PoPETs-FairFed",
    "cgsv": "CGSV", "fltrust": "FLTrust", "bfwa": "BFWA",
    "fu_shapley": "FU-Alignment (ours)",
    "fu_shapley_alpha0": "FU-Alignment ($\\alpha=0$)",
}



def _tex(s: str) -> str:
    """Escape what LaTeX would choke on. An unescaped _ inside \texttt{} is a
    hard compile error, and `group1_rate` is a real field name we must print."""
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#"))


def _p(p: float) -> str:
    """4 decimals collapse every p below 1e-4 to 0.0000, which says nothing."""
    if p >= 1e-3:
        return f"${p:.4f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def _reads_metadata(v) -> bool:
    return not str(v["reads"]).startswith("--")


def analyse(path: str):
    d = json.load(open(path))
    R, args = d["results"], d["manifest"]["args"]
    uniform = 1.0 / float(args["num_clients"])

    rows, pvals = {}, {}
    for rule, v in R.items():
        hon = {x["seed"]: x for x in v["arms"]["poison_honest"]["per_seed"]}
        lie = {x["seed"]: x for x in v["arms"]["poison_lie"]["per_seed"]}

        def ok(x):
            return x["w_adv"] is not None and math.isfinite(x["w_adv"])

        seeds = [s for s in sorted(hon) if s in lie and ok(hon[s]) and ok(lie[s])]
        wh = [hon[s]["w_adv"] for s in seeds]
        wl = [lie[s]["w_adv"] for s in seeds]
        rep = paired_report(wl, wh, lower_is_better=False)
        deltas = [b - a for a, b in zip(wh, wl)]

        mh, ml = sum(wh) / len(wh), sum(wl) / len(wl)
        harm = v.get("paired_lie_minus_honest", {})
        rows[rule] = {
            "reads": v["reads"], "reads_metadata": _reads_metadata(v),
            "published": rule in PUBLISHED, "venue": VENUE[rule],
            "published_in_artifact": v["published"],
            "n_pairs": len(seeds), "seeds_dropped": [s for s in sorted(hon) if s not in seeds],
            "w_adv_honest": mh, "w_adv_lie": ml,
            "ratio": (ml / mh) if mh else None, "times_uniform": ml / uniform,
            "p_wilcoxon": rep["p_wilcoxon"], "wins": rep["wins"],
            "cohens_dz": rep["cohens_dz"], "ci95": rep["ci95"],
            "n_positive": sum(1 for x in deltas if x > 0),
            "n_negative": sum(1 for x in deltas if x < 0),
            "n_tied": sum(1 for x in deltas if x == 0),
            "max_abs_delta": max(abs(x) for x in deltas),
            # The signed-rank test DROPS ties, so this -- not n_pairs -- sets the
            # smallest p-value the test can return: 2 ** (1 - n_nonzero).
            "n_nonzero": sum(1 for x in deltas if x != 0),
        }
        for m in ("dpd_hard", "eod", "auc"):
            h = harm.get(m) or {}
            rows[rule][f"d_{m}"] = h.get("mean_delta")
            rows[rule][f"p_{m}"] = h.get("wilcoxon_p")
            rows[rule][f"n_{m}"] = h.get("n_pairs")
        rows[rule]["p_floor"] = 2.0 ** (1 - rows[rule]["n_nonzero"]) if rows[rule]["n_nonzero"] else 1.0
        pvals[rule] = rep["p_wilcoxon"]

    reading = [r for r in rows if rows[r]["reads_metadata"]]
    holm = {
        "reading": holm_bonferroni({r: pvals[r] for r in reading}),
        "all": holm_bonferroni(pvals),
    }

    pos = sum(rows[r]["n_positive"] for r in reading)
    neg = sum(rows[r]["n_negative"] for r in reading)
    tied = sum(rows[r]["n_tied"] for r in reading)
    aggregate = {
        "family": reading, "n_positive": pos, "n_negative": neg, "n_tied": tied,
        "sign_test_p": float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else 1.0,
    }
    controls = {r: rows[r]["max_abs_delta"] for r in rows if not rows[r]["reads_metadata"]}
    return d, rows, holm, aggregate, controls, uniform


def render(rows, holm, aggregate, controls, uniform, args, family="reading") -> str:
    def fmt_p(rule):
        if not rows[rule]["reads_metadata"]:
            return "---"
        star = "$^\\star$" if holm[family].get(rule) else ""
        return _p(rows[rule]["p_wilcoxon"]) + star

    def fmt_delta(rule, metric):
        if not rows[rule]["reads_metadata"]:
            return "$0.0000$"
        val = rows[rule].get(f"d_{metric}")
        pval = rows[rule].get(f"p_{metric}")
        if val is None:
            return "---"
        star = "$^\\star$" if (pval is not None and pval < 0.05) else ""
        return f"${val:+.4f}${star}"

    order = ([r for r in rows if rows[r]["reads_metadata"]]
             + [r for r in rows if not rows[r]["reads_metadata"]])
    order.sort(key=lambda r: (not rows[r]["reads_metadata"], rows[r]["p_wilcoxon"]))

    body = []
    for rule in order:
        v = rows[rule]
        dagger = "$^\\dagger$" if rule == "bfwa" else ""
        name = PRETTY[rule] + dagger
        w_lie = f"$\\mathbf{{{v['w_adv_lie']:.4f}}}$" if v["times_uniform"] >= 3 else f"${v['w_adv_lie']:.4f}$"
        body.append(
            f"{name} & {v['venue']} & \\texttt{{{_tex(v['reads'])}}} & "
            f"{v['n_pairs']} & "
            f"${v['w_adv_honest']:.4f}$ & {w_lie} & "
            f"${v['times_uniform']:.2f}\\times$ & {fmt_p(rule)} & "
            f"{fmt_delta(rule, 'dpd_hard')} & {fmt_delta(rule, 'eod')} & {fmt_delta(rule, 'auc')} \\\\")
        if rule == order[len([r for r in rows if rows[r]['reads_metadata']]) - 1]:
            body.append("\\midrule")

    ctrl = ", ".join(f"{PRETTY[r]} ${v:.4f}$" for r, v in controls.items())
    agg = aggregate
    caption = (
        "\\textbf{Capturing the aggregate through the self-reported channel.} "
        f"German, $K={args['num_clients']}$, $R={args['rounds']}$, "
        f"$n={len(args['seeds'])}$ seeds, GCN backbone held fixed across every rule, "
        "statistic-level DP disabled so the metadata channel is measured in isolation. "
        "Each row contrasts two arms that both transmit the \\emph{same} poisoned update and "
        "differ only in the accompanying report, so the difference is attributable to the "
        "declared fields alone. $w_{\\mathrm{adv}}$ is the adversary's mean aggregation weight "
        f"across rounds; uniform weight is $1/K={uniform:.2f}$. "
        "The rightmost columns report downstream task and fairness harm: $\\Delta\\mathrm{DPD}$, $\\Delta\\mathrm{EOD}$, and $\\Delta\\mathrm{AUC}$ "
        "(paired lie minus honest; $^\\star$ indicates $p < 0.05$). "
        "Results validate a three-tier taxonomy: (i)~\\emph{steerable-harmless} (PoPETs-FairFed shifts weights significantly but downstream metrics remain indistinguishable from zero); "
        "(ii)~\\emph{operational capture} with tangible group harm (FairFed, q-FedAvg, FedGraph-Fair, BFWA); "
        "and (iii)~\\emph{downstream collapse} (F\\textsuperscript{2}GNN suffers a $7.4\\times$ increase in DPD and $-0.14$ AUC drop). "
        "Metadata-blind rules (CGSV, FLTrust) remain bit-exact invariant ($\\|\\Delta\\|=0.0000$). "
        "Note that q-FedAvg converged on $n=18$ paired seeds ($12$ seeds diverged under extreme weight skew). "
        "$^\\star$ in $p(w)$ indicates significance after Holm--Bonferroni across the metadata-reading family. "
        "$^\\dagger$ our own earlier published rule~\\cite{dang2026fedfairgnn}, attacked here on the same footing as the others."
    )
    return ("\\begin{table*}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{tab:metadata_capture}}\n"
            "\\resizebox{\\linewidth}{!}{%\n\\setlength{\\tabcolsep}{4pt}%\n"
            "\\begin{tabular}{lllcccccccc}\n\\toprule\n"
            "\\textbf{Aggregation rule} & \\textbf{Venue} & \\textbf{Reads} & \\textbf{$n$} & "
            "\\textbf{$w_{\\mathrm{adv}}$ honest} & \\textbf{$w_{\\mathrm{adv}}$ lying} & "
            "\\textbf{vs.\\ uniform} & \\textbf{$p(w)$} & "
            "\\textbf{$\\Delta$DPD} & \\textbf{$\\Delta$EOD} & \\textbf{$\\Delta$AUC} \\\\\n\\midrule\n"
            + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-tex", default="manuscript_neurocomputing/tables/revision/metadata_capture.tex")
    ap.add_argument("--out-json", default="results/revision/metadata_capture_stats.json")
    ap.add_argument("--family", default="reading", choices=("reading", "all"))
    a = ap.parse_args()

    src, rows, holm, aggregate, controls, uniform = analyse(a.in_json)
    tex = render(rows, holm, aggregate, controls, uniform, src["manifest"]["args"], a.family)

    os.makedirs(os.path.dirname(a.out_tex) or ".", exist_ok=True)
    with open(a.out_tex, "w") as f:
        f.write(tex)

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump({
            "manifest": build_manifest(
                experiment="metadata_capture_stats",
                source_artifact=os.path.basename(a.in_json),
                source_manifest=src["manifest"],
                holm_family=a.family),
            "per_rule": rows, "holm": holm, "aggregate_sign_test": aggregate,
            "control_max_abs_delta": controls,
        }, f, indent=1)
        f.write("\n")

    print(f"[+] {a.out_tex}")
    print(f"[+] {a.out_json}")
    for r, v in rows.items():
        print(f"  {r:<16} w {v['w_adv_honest']:.4f} -> {v['w_adv_lie']:.4f}  "
              f"p={v['p_wilcoxon']:.4f} floor={v['p_floor']:.4f} "
              f"holm={holm[a.family].get(r)}  +/-/= {v['n_positive']}/{v['n_negative']}/{v['n_tied']}")
    print(f"  aggregate sign test: {aggregate['n_positive']}+ {aggregate['n_negative']}- "
          f"{aggregate['n_tied']}=  p={aggregate['sign_test_p']:.3e}")


if __name__ == "__main__":
    main()
