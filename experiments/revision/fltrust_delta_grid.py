"""P5-X1 -- FLTrust Delta 2x2 Grid on GAT backbone (CPU, OMP_NUM_THREADS=1).

Pre-registration locked in docs/04_1_novelty_advantages.md §4.6.
CC-4 Resolution (Option A): runs all three attack conditions:
  - clean: benign operation (num_byzantine=0)
  - poison_honest: fairness poisoned gradient with honest metadata report
  - fairness_poison: fairness poisoned gradient with best-response lie

4 arms on German Credit and Bail, 30 seeds {42..71}:
  A1: FLTrust verbatim       -- aggregator=fltrust, fu_alpha=0.0
  A2: FLTrust + alpha*g_fair  -- aggregator=fltrust, fu_alpha=0.1
  A3: FU-Alignment alpha=0   -- aggregator=fu_shapley, fu_alpha=0.0
  A4: FU-Alignment alpha=0.1 -- aggregator=fu_shapley, fu_alpha=0.1

MANDATORY CPU: expected Delta_w_adv ~0.0019, 54x smaller than CUDA jitter.
Per L7: each worker process is strictly pinned to torch_num_threads=1.
Gate sequence: S0 (1 run) -> S1 (repeat S0) -> S3 (3 seeds x 4 arms x 3 scenarios) -> S4 (30 seeds).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import statistics as st

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.abspath("."))

import torch
torch.set_num_threads(1)

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

# ---- ARM DEFINITIONS per §4.6 pre-registration ----
ARMS = {
    "A1_fltrust": dict(
        aggregator="fltrust",
        fu_alpha=0.0,
        dp_enabled=False,
    ),
    "A2_fltrust_fair": dict(
        aggregator="fltrust",
        fu_alpha=0.1,
        dp_enabled=False,
    ),
    "A3_fu_alpha0": dict(
        aggregator="fu_shapley",
        fu_alpha=0.0,
        dp_enabled=False,
    ),
    "A4_fu_shapley": dict(
        aggregator="fu_shapley",
        fu_alpha=0.1,
        dp_enabled=False,
    ),
}

# ---- ATTACK SCENARIOS per CC-4 Option A ----
SCENARIOS = {
    "clean": dict(
        attack="none",
        num_byzantine=0,
    ),
    "poison_honest": dict(
        attack="fairness_poison_honest_report",
        num_byzantine=1,
        attack_intensity=10.0,
    ),
    "fairness_poison": dict(
        attack="fairness_poison",
        num_byzantine=1,
        attack_intensity=10.0,
    ),
}

DATASETS = ["german", "bail"]
NUM_CLIENTS = 5


def run_one(
    dataset: str,
    arm_name: str,
    arm_kwargs: dict,
    scenario_name: str,
    attack_kwargs: dict,
    seed: int,
    device: str = "cpu",
) -> dict:
    """Run a single experiment and return metrics dict."""
    torch.set_num_threads(1)
    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=NUM_CLIENTS,
        device=device,
        **arm_kwargs,
        **attack_kwargs,
    )
    t0 = time.perf_counter()
    trainer = FederatedTrainer(cfg)
    res = trainer.run(verbose=False)
    wall = time.perf_counter() - t0
    f = res.get("final", res)

    n_byz = attack_kwargs.get("num_byzantine", 0)
    adv_weights = []
    for rec in getattr(trainer, "history", []):
        w = rec.get("agg_weights")
        if w and n_byz > 0:
            adv_weights.append(float(sum(w[:n_byz])))
    w_adv = float(sum(adv_weights) / len(adv_weights)) if adv_weights else 0.0

    return {
        "seed": seed,
        "auc": float(f["auc"]),
        "dpd_hard": float(f["dpd_hard"]),
        "w_adv": w_adv,
        "wall_s": float(wall),
        "threads": torch.get_num_threads(),
    }


def _worker_entry(task: tuple) -> dict:
    dataset, scenario_name, arm_name, seed, device = task
    row = run_one(
        dataset=dataset,
        arm_name=arm_name,
        arm_kwargs=ARMS[arm_name],
        scenario_name=scenario_name,
        attack_kwargs=SCENARIOS[scenario_name],
        seed=seed,
        device=device,
    )
    return {
        "dataset": dataset,
        "scenario": scenario_name,
        "arm": arm_name,
        "seed": seed,
        "row": row,
    }


def run_gate(
    out_json: str,
    seeds: list[int],
    datasets: list[str] | None = None,
    scenarios: list[str] | None = None,
    arms: list[str] | None = None,
    device: str = "cpu",
    workers: int = 1,
) -> dict:
    target_datasets = datasets or DATASETS
    target_scenarios = scenarios or list(SCENARIOS.keys())
    target_arms = arms or list(ARMS.keys())

    tasks = []
    for dataset in target_datasets:
        for scenario in target_scenarios:
            for arm in target_arms:
                for seed in seeds:
                    tasks.append((dataset, scenario, arm, seed, device))

    total_runs = len(tasks)
    done = 0
    all_rows: list[dict] = []
    t_start = time.perf_counter()

    print(f"[*] Launching {total_runs} runs with {workers} worker process(es) (threads=1 per process)...", flush=True)


    if workers > 1 and total_runs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_worker_entry, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                res = future.result()
                all_rows.append(res)
                done += 1
                r = res["row"]
                print(
                    f"[{done:3d}/{total_runs:3d}] {res['dataset']:6s} {res['scenario']:15s} {res['arm']:16s} "
                    f"seed={res['seed']:2d} auc={r['auc']:.4f} dpd={r['dpd_hard']:.4f} w_adv={r['w_adv']:.4f} "
                    f"({r['wall_s']:.1f}s)",
                    flush=True,
                )
    else:
        for t in tasks:
            res = _worker_entry(t)
            all_rows.append(res)
            done += 1
            r = res["row"]
            print(
                f"[{done:3d}/{total_runs:3d}] {res['dataset']:6s} {res['scenario']:15s} {res['arm']:16s} "
                f"seed={res['seed']:2d} auc={r['auc']:.4f} dpd={r['dpd_hard']:.4f} w_adv={r['w_adv']:.4f} "
                f"({r['wall_s']:.1f}s)",
                flush=True,
            )

    total_wall_s = time.perf_counter() - t_start

    # Assemble structured results
    results: dict = {}
    for dataset in target_datasets:
        results[dataset] = {}
        for scenario in target_scenarios:
            results[dataset][scenario] = {}
            for arm in target_arms:
                results[dataset][scenario][arm] = {"per_seed": []}

    for item in all_rows:
        d = item["dataset"]
        s = item["scenario"]
        a = item["arm"]
        results[d][s][a]["per_seed"].append(item["row"])

    # Sort per_seed by seed
    for d in results:
        for s in results[d]:
            for a in results[d][s]:
                results[d][s][a]["per_seed"].sort(key=lambda x: x["seed"])
                per_seed = results[d][s][a]["per_seed"]
                if per_seed:
                    aucs = [r["auc"] for r in per_seed]
                    dpds = [r["dpd_hard"] for r in per_seed]
                    w_advs = [r["w_adv"] for r in per_seed]
                    walls = [r["wall_s"] for r in per_seed]
                    results[d][s][a]["auc"] = {
                        "mean": float(st.mean(aucs)),
                        "std": float(st.stdev(aucs)) if len(aucs) > 1 else 0.0,
                    }
                    results[d][s][a]["dpd_hard"] = {
                        "mean": float(st.mean(dpds)),
                        "std": float(st.stdev(dpds)) if len(dpds) > 1 else 0.0,
                    }
                    results[d][s][a]["w_adv"] = {
                        "mean": float(st.mean(w_advs)),
                        "std": float(st.stdev(w_advs)) if len(w_advs) > 1 else 0.0,
                    }
                    results[d][s][a]["mean_wall_s"] = float(st.mean(walls))

    all_walls = [item["row"]["wall_s"] for item in all_rows]
    mean_wall_per_run = float(st.mean(all_walls)) if all_walls else 0.0

    manifest = build_manifest(
        device=device,
        experiment="p5_x1_fltrust_delta_grid",
        wall_clock_s=round(total_wall_s, 2),
        mean_wall_s_per_run=round(mean_wall_per_run, 2),
        workers=workers,
        total_runs=total_runs,
        args={
            "datasets": target_datasets,
            "scenarios": target_scenarios,
            "arms": target_arms,
            "seeds": seeds,
            "num_clients": NUM_CLIENTS,
            "device": device,
        },
    )

    # Compute 2x2 effects if full grid is present
    deltas = {}
    for d in target_datasets:
        deltas[d] = {}
        for s in target_scenarios:
            arms_data = results[d][s]
            if all(k in arms_data and "auc" in arms_data[k] for k in ("A1_fltrust", "A2_fltrust_fair", "A3_fu_alpha0", "A4_fu_shapley")):
                a1_w = arms_data["A1_fltrust"]["w_adv"]["mean"]
                a2_w = arms_data["A2_fltrust_fair"]["w_adv"]["mean"]
                a3_w = arms_data["A3_fu_alpha0"]["w_adv"]["mean"]
                a4_w = arms_data["A4_fu_shapley"]["w_adv"]["mean"]

                a1_dpd = arms_data["A1_fltrust"]["dpd_hard"]["mean"]
                a2_dpd = arms_data["A2_fltrust_fair"]["dpd_hard"]["mean"]
                a3_dpd = arms_data["A3_fu_alpha0"]["dpd_hard"]["mean"]
                a4_dpd = arms_data["A4_fu_shapley"]["dpd_hard"]["mean"]

                deltas[d][s] = {
                    "delta_alpha_fltrust_w_adv": round(a2_w - a1_w, 6),
                    "delta_alpha_fu_w_adv": round(a4_w - a3_w, 6),
                    "delta_gating_alpha0_w_adv": round(a3_w - a1_w, 6),
                    "delta_gating_alpha01_w_adv": round(a4_w - a2_w, 6),
                    "delta_alpha_fltrust_dpd": round(a2_dpd - a1_dpd, 6),
                    "delta_alpha_fu_dpd": round(a4_dpd - a3_dpd, 6),
                }


    payload = {
        "manifest": manifest,
        "results": results,
        "deltas_2x2": deltas,
    }

    os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[+] Saved to {out_json} (total wall: {total_wall_s:.1f}s, mean/run: {mean_wall_per_run:.2f}s)", flush=True)

    # Print summary table
    print("\n" + "=" * 95, flush=True)
    print("P5-X1 SUMMARY TABLE (2x2 Grid x Scenarios x Datasets)", flush=True)
    print("=" * 95, flush=True)
    for d in target_datasets:
        print(f"\n--- DATASET: {d.upper()} ---", flush=True)
        for s in target_scenarios:
            print(f"  Scenario [{s}]:")
            for a in target_arms:
                m = results[d][s][a]
                if "auc" in m:
                    print(
                        f"    {a:16s} | AUC={m['auc']['mean']:.4f}±{m['auc']['std']:.4f} | "
                        f"DPD={m['dpd_hard']['mean']:.4f}±{m['dpd_hard']['std']:.4f} | "
                        f"w_adv={m['w_adv']['mean']:.4f}±{m['w_adv']['std']:.4f} | "
                        f"wall={m['mean_wall_s']:.2f}s"
                    )
    print("=" * 95, flush=True)


    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=["s0", "s1", "s3", "s4"], default="s0")
    parser.add_argument("--out-json", default="results/revision/fltrust_delta_grid.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args, _ = parser.parse_known_args()

    datasets = None
    scenarios = None
    arms = None

    if args.seeds is not None:
        seeds = args.seeds
    elif args.gate in ("s0", "s1"):
        seeds = [42]
        datasets = ["german"]
        scenarios = ["clean"]
        arms = ["A1_fltrust"]
        print(f"=== P5-X1 {args.gate.upper()}: Single smoke run (German, clean, A1, seed 42) ===", flush=True)
    elif args.gate == "s3":
        seeds = [42, 43, 44]
        print("=== P5-X1 S3: Probe (3 seeds x 4 arms x 3 scenarios x 2 datasets = 72 runs) ===", flush=True)
    else:  # s4
        seeds = list(range(42, 72))
        print(f"=== P5-X1 S4: Full run ({len(seeds)} seeds x 4 arms x 3 scenarios x 2 datasets = {len(seeds)*24} runs) ===", flush=True)

    t0 = time.perf_counter()
    run_gate(
        args.out_json,
        seeds=seeds,
        datasets=datasets,
        scenarios=scenarios,
        arms=arms,
        device=args.device,
        workers=args.workers,
    )
    total = time.perf_counter() - t0
    print(f"\n=== P5-X1 {args.gate.upper()} COMPLETE in {total:.1f}s ===", flush=True)

