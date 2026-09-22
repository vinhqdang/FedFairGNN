import json

with open("results/revision/metadata_capture_bail.json") as f:
    d = json.load(f)

res = d["results"]
print("="*90)
print(f"{'Rule':<18} {'Reads':<18} {'Delta w_adv':<14} {'p(w_adv)':<12} {'Delta DPD':<14} {'p(DPD)':<12} {'Over 0.05?'}")
print("="*90)

count_over_threshold = 0
meta_rules = ["fairfed", "qffl", "f2gnn", "fedgraphfair", "popets_fairfed", "bfwa"]

for r in ["fairfed", "qffl", "f2gnn", "fedgraphfair", "popets_fairfed", "bfwa", "cgsv", "fltrust", "fu_shapley", "fu_shapley_alpha0"]:
    reads = res[r]["reads"]
    p_pair = res[r].get("paired_lie_minus_honest", {})
    w_delta = p_pair.get("w_adv", {}).get("mean_delta", 0.0)
    w_p = p_pair.get("w_adv", {}).get("wilcoxon_p", 1.0)
    dpd_delta = p_pair.get("dpd_hard", {}).get("mean_delta", 0.0)
    dpd_p = p_pair.get("dpd_hard", {}).get("wilcoxon_p", 1.0)
    
    over = "YES (>=0.05)" if dpd_delta >= 0.0500 else "no"
    if r in meta_rules and dpd_delta >= 0.0500:
        count_over_threshold += 1
        
    print(f"{r:<18} {reads:<18} {w_delta:+.4f}         {w_p:.4e}   {dpd_delta:+.4f}         {dpd_p:.4e}   {over}")

print("="*90)
print(f"Meta-rules with Delta DPD >= 0.0500: {count_over_threshold}/6")
