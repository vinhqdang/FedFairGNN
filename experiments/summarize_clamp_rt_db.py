import collections, glob, json, statistics as st, sys
def load(pat):
    o=[]
    for f in glob.glob(pat):
        r=json.load(open(f)); o.append((r["run_id"].split("__")[0], r["config"], r["final"]))
    return o
def ms(x): return f"{st.mean(x):.3f}±{(st.stdev(x) if len(x)>1 else 0):.3f}" if x else "na"
def table(rows,key,cols=("dpd_soft","auc")):
    g=collections.defaultdict(list)
    for m,c,f in rows: g[key(m,c)].append(f)
    for k in sorted(g,key=str):
        v=g[k]; print(k,"n",len(v),*[f"{c} {ms([x[c] for x in v if x.get(c) is not None])}" for c in cols])
w=sys.argv[1]
if w=="e1":
    rows=load("results/clamp_rt/e1db/runs/*.json")+load("results/clamp_rt/e1/runs/*.json")+[r for r in load("results/clamp/e1/runs/*.json") if r[0] in("clamp-f","clamp-gcn-f")]
    table(rows,lambda m,c:(c["dataset"],m))
elif w=="e3":
    rows=load("results/clamp_rt/e3db/runs/*.json"); table(rows,lambda m,c:(c["dataset"],c["attack"],c["num_byzantine"]))
elif w=="e5":
    rows=load("results/clamp_rt/e5db/runs/*.json"); table(rows,lambda m,c:(c["dataset"],c["dp_epsilon"],c["r2_stat_period"]))
elif w=="e4":
    rows=load("results/clamp_rt/e4db/runs/*.json"); table(rows,lambda m,c:(c["dataset"],c["num_clients"],c["attack"],c["num_byzantine"]))
