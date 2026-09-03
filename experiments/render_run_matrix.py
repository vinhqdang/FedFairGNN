"""Script to automatically render Markdown tables from results JSON artifacts into docs/04.

Eliminates manual transcription errors by populating tables directly from results/*.json.
"""
import json
import os
import sys

def format_cell(mean, std, digits=4):
    if mean is None or np.isnan(mean):
        return "—"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"

def render_markdown_tables(json_path="results/stage4_remediation_results.json"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path) as f:
        data = json.load(f)
        
    print(f"[*] Loaded results from {json_path}")
    manifest = data.get("_manifest", {})
    print(f"[*] Manifest: device={manifest.get('device')}, commit={manifest.get('git_commit')}, dirty={manifest.get('git_dirty')}")
    
    # 1. Stage 4.2 Matrix
    print("\n--- STAGE 4.2 RUN MATRIX ---")
    headers = ["Run ID", "Dataset", "Method", "Config", "AUC-ROC", "DPD (soft)", "DPD (hard@0.5)", "EOD", "Omega_w"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join([":---:" for _ in headers]) + "|")
    
    for rid in ["RUN-4.2-01", "RUN-4.2-02", "RUN-4.2-04", "RUN-4.2-05"]:
        res = data.get(rid, {})
        auc = f"${res.get('auc_mean', 0):.4f} \\pm {res.get('auc_std', 0):.4f}$"
        dpd_s = f"${res.get('dpd_soft_mean', res.get('dpd_mean', 0)):.4f} \\pm {res.get('dpd_soft_std', res.get('dpd_std', 0)):.4f}$"
        dpd_h = f"${res.get('dpd_hard_mean', 0):.4f} \\pm {res.get('dpd_hard_std', 0):.4f}$"
        eod = f"${res.get('eod_mean', 0):.4f} \\pm {res.get('eod_std', 0):.4f}$"
        omg = f"${res.get('omega_w_mean', 0):.4f} \\pm {res.get('omega_w_std', 0):.4f}$"
        print(f"| `{rid}` | {res.get('dataset', '')} | {res.get('method', rid)} | canonical | {auc} | {dpd_s} | {dpd_h} | {eod} | {omg} |")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/stage4_remediation_results.json"
    render_markdown_tables(path)
