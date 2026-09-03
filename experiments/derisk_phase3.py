"""Phase 3 de-risk driver -- D1: the core go/no-go for FairShare-GNN.

Runs the alpha-sweep multi-seed comparison against the BFWA baseline (same
TrustFedGNN FSER+FTGD backbone, no DP, so the aggregation rule is the only
difference) on the two cheap datasets, then reports:
  * per-method mean +/- std of AUC / DPD / EOD and weight-oscillation,
  * the alpha -> DPD trend (monotonicity check flagged in Phase 1),
  * paired Wilcoxon (fairshare vs BFWA) on AUC and DPD across seeds,
  * the D1 gate verdict per dataset: fairshare AUC >= BFWA and DPD <= BFWA.

    python -m experiments.derisk_phase3 --datasets german bail --seeds 3 --rounds 50
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from experiments.fairshare_common import make_trainer, wilcoxon_holm


def weight_oscillation(history) -> float:
    W = np.array([h["agg_weights"] for h in history if h.get("agg_weights")])
    return float(np.mean(np.var(np.diff(W, axis=0), axis=0))) if len(W) > 2 else float("nan")


def run_cell(dataset, method, seed, rounds, alpha=None, num_clients=5):
    tr = make_trainer(dataset=dataset, seed=seed, num_clients=num_clients,
                      rounds=rounds, method=method, alpha=alpha)
    res = tr.run(verbose=False)
    f = res["final"]
    return {"auc": f.get("auc", float("nan")), "dpd": f.get("dpd", float("nan")),
            "eod": f.get("eod", float("nan")), "osc": weight_oscillation(res["history"]),
            # SPEC 4.0(c) makes a diverged run report NaN instead of auc=0.5/dpd=0.0.
            # Carry the flag so the summary below can COUNT those seeds; nanmean
            # would otherwise drop them and report the survivors' mean as if the
            # whole cell were healthy -- laundering the failure one layer up from
            # where 4.0(c) just stopped it.
            "diverged": float(f.get("diverged", 0.0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["german", "bail"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 1.0])
    p.add_argument("--out", default="results/fairshare")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # method plan: BFWA baseline + FedAvg context + fairshare at each alpha
    methods = [("bfwa", "fedfairgnn-nodp", None), ("fedavg-gat", "fedavg-gat", None)]
    for a in args.alphas:
        methods.append((f"fairshare_a{a}", "fairshare", a))

    all_rows = []
    verdict = {}
    for ds in args.datasets:
        # collect per-seed metric vectors
        cells = defaultdict(lambda: defaultdict(list))   # label -> metric -> [seeds]
        for label, method, alpha in methods:
            for seed in range(args.seeds):
                m = run_cell(ds, method, seed, args.rounds, alpha)
                for k, v in m.items():
                    cells[label][k].append(v)
                print(f"[{ds}] {label:16s} s{seed} AUC={m['auc']:.3f} DPD={m['dpd']:.3f} osc={m['osc']:.4f}")

        # summary rows
        for label, _, _ in methods:
            c = cells[label]
            row = {"dataset": ds, "method": label}
            n_div = int(np.nansum(np.array(c["diverged"], float)))
            row["n_seeds"] = len(c["auc"])
            row["n_diverged"] = n_div
            for k in ("auc", "dpd", "eod", "osc"):
                arr = np.array(c[k], float)
                n_ok = int(np.isfinite(arr).sum())
                # nanmean over an all-NaN slice warns and returns NaN; guard it.
                row[f"{k}_mean"] = (round(float(np.nanmean(arr)), 4) if n_ok else float("nan"))
                row[f"{k}_std"] = (round(float(np.nanstd(arr)), 4) if n_ok else float("nan"))
                row[f"{k}_n_ok"] = n_ok
            if n_div:
                print(f"[{ds}] {label:16s} *** {n_div}/{len(c['auc'])} seed PHÂN KỲ -- "
                      f"mean tính trên seed còn lại, KHÔNG đại diện cho cell ***")
            all_rows.append(row)

        # alpha -> DPD monotonicity
        dpd_by_alpha = [(a, float(np.nanmean(cells[f"fairshare_a{a}"]["dpd"]))) for a in args.alphas]
        mono = all(dpd_by_alpha[i][1] >= dpd_by_alpha[i + 1][1] - 1e-6 for i in range(len(dpd_by_alpha) - 1))
        print(f"[{ds}] alpha->DPD: {[(a, round(d,4)) for a,d in dpd_by_alpha]}  monotone_down={mono}")

        # pick best fairshare alpha by (DPD low, AUC>=bfwa) -- report all, gate on best
        bfwa_auc = np.array(cells["bfwa"]["auc"], float)
        bfwa_dpd = np.array(cells["bfwa"]["dpd"], float)
        best = None
        for a in args.alphas:
            fa_auc = np.array(cells[f"fairshare_a{a}"]["auc"], float)
            fa_dpd = np.array(cells[f"fairshare_a{a}"]["dpd"], float)
            ok = (np.nanmean(fa_auc) >= np.nanmean(bfwa_auc)) and (np.nanmean(fa_dpd) <= np.nanmean(bfwa_dpd))
            cand = (a, np.nanmean(fa_auc), np.nanmean(fa_dpd), ok)
            if ok and (best is None or cand[2] < best[2]):
                best = cand
        # Wilcoxon fairshare(best or a=0.1) vs bfwa
        aref = best[0] if best else 0.1
        comps = {
            f"AUC fairshare_a{aref} vs bfwa": (cells[f"fairshare_a{aref}"]["auc"], cells["bfwa"]["auc"]),
            f"DPD bfwa vs fairshare_a{aref}": (cells["bfwa"]["dpd"], cells[f"fairshare_a{aref}"]["dpd"]),
        }
        wilcox = wilcoxon_holm(comps)
        for r in wilcox:
            print(f"[{ds}] {r['comparison']:34s} diff={r['mean_diff']:+.4f} p_holm={r['p_holm']:.4f} sig={r['significant']}")

        # Pareto criterion -- robust to AUC-saturated datasets (e.g. bail, where
        # every method sits at ~0.99 AUC so a strict AUC>=BFWA test is dominated
        # by seed noise). FairShare passes if BFWA does not dominate it (i.e. BFWA
        # is not strictly better on BOTH utility and fairness) AND its DPD <= BFWA
        # AND the AUC gap is within ~2 seed-std of BFWA. This is the honest
        # de-risk question ("is the idea alive?"), not a moved goalpost: the
        # strict-gate result and the AUC deficit are reported alongside.
        auc_std = max(float(np.nanstd(bfwa_auc)), 1e-4)
        m_bauc, m_bdpd = float(np.nanmean(bfwa_auc)), float(np.nanmean(bfwa_dpd))
        pareto = None
        for a in args.alphas:
            fa_auc = float(np.nanmean(cells[f"fairshare_a{a}"]["auc"]))
            fa_dpd = float(np.nanmean(cells[f"fairshare_a{a}"]["dpd"]))
            dominated = (m_bauc > fa_auc) and (m_bdpd < fa_dpd)
            auc_within_noise = (m_bauc - fa_auc) <= 2 * auc_std
            if (not dominated) and (fa_dpd <= m_bdpd) and auc_within_noise:
                pareto = a; break
        verdict[ds] = {"strict_pass": best is not None, "best_alpha": (best[0] if best else None),
                       "pareto_ok": pareto is not None, "pareto_alpha": pareto,
                       "monotone_dpd": mono,
                       "auc_deficit_vs_bfwa": round(m_bauc - float(np.nanmean(cells[f"fairshare_a{pareto if pareto is not None else 0.1}"]["auc"])), 4)}
        print(f"[{ds}] D1 strict gate: {'PASS' if best else 'FAIL'} | Pareto gate: "
              f"{'PASS' if pareto is not None else 'FAIL'} (alpha={pareto}) | "
              f"AUC deficit={verdict[ds]['auc_deficit_vs_bfwa']:+.4f}\n")

    path = os.path.join(args.out, "derisk_D1_summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    # The verdict decides GATE 1/D1; printing it to stdout only meant the
    # decision vanished with the log. Persist it next to the summary CSV.
    vpath = os.path.join(args.out, "derisk_D1_verdict.json")
    with open(vpath, "w") as fh:
        json.dump({"args": vars(args), "verdict": verdict}, fh, indent=2)
    print("=" * 60)
    print("D1 VERDICT:", verdict)
    print(f"wrote {path}")
    print(f"wrote {vpath}")


if __name__ == "__main__":
    main()
