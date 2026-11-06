"""Summaries of the CLAMP-RT campaign (E1, E3, E5, E6) against the reused CLAMP runs."""
import collections, glob, json, statistics as st, sys

def load(pat):
    out = []
    for f in glob.glob(pat):
        r = json.load(open(f))
        out.append((r["run_id"].split("__")[0], r["config"], r["final"]))
    return out

def ms(x):
    return f"{st.mean(x):.3f}±{(st.stdev(x) if len(x) > 1 else 0):.3f}"

def table(rows, key, cols=("dpd_soft", "auc")):
    g = collections.defaultdict(list)
    for m, c, fin in rows:
        g[key(m, c)].append(fin)
    for k in sorted(g, key=str):
        v = g[k]
        print(k, "n", len(v), *[f"{c} {ms([x[c] for x in v if x.get(c) is not None])}" for c in cols])

which = sys.argv[1] if len(sys.argv) > 1 else "e1"
if which == "e1":
    rows = load("results/clamp_rt/e1/runs/*.json") + [r for r in load("results/clamp/e1/runs/*.json")
                                                      if r[0] in ("clamp-f", "clamp-gcn-f", "clamp-clipmean", "clamp")]
    table(rows, lambda m, c: (c["dataset"], m))
elif which == "e3":
    rows = load("results/clamp_rt/e3/runs/*.json")
    table(rows, lambda m, c: (c["dataset"], c["attack"], c["num_byzantine"]))
elif which == "e5":
    rows = load("results/clamp_rt/e5/runs/*.json")
    table(rows, lambda m, c: (c["dataset"], c["dp_epsilon"] if c["r2_stat_dp"] else "inf", c["r2_stat_period"]))
elif which == "e6":
    rows = load("results/clamp_rt/e6/runs/*.json")
    table(rows, lambda m, c: (c["dataset"], m, c["r2_trim_b"], c["r2_floor_frac"]), cols=("dpd_soft", "auc", "sigma_d"))
