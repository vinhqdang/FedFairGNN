"""Empirical probe for the cross-round theta_t sensitivity gap in Proposition 1.

Proposition 1 (docs/02) bounds the L2 sensitivity of the released group-mean
statistic mu_k by conditioning on the current global model theta_t as fixed,
public state. That is correct *within one round* -- FTGD's released statistic
is computed on a separate, sensitive-blind forward pass (see
Client._release_pred / cfg.dp_statistic_s_blind), so the released prediction
for a fixed theta_t provably does not read s directly (verified by
tests/test_dp_mode_resolution.py::test_released_statistic_is_computed_s_blind_when_dp_is_live).

What the proof does NOT establish is that theta_t itself is independent of a
single node's sensitive attribute across the federated training that produced
it: FSER reads s directly in the (non-released) task pathway, and the
fairness gradient depends on group statistics, so training on neighbouring
datasets D, D' (differing in one node's s) can in principle yield theta_t(D)
!= theta_t(D'). If the resulting shift in the s-blind release prediction is
non-negligible relative to the DP noise sigma already added, the "condition
on theta_t" step in the composition argument (Corollary 1) is on shakier
ground than the per-round algebra alone suggests.

This script does NOT attempt a new proof. It measures the effect directly:
for a small set of (seed, client, node) probes, train the federated protocol
twice with everything identical except one node's sensitive attribute, then
compare the s-blind release prediction the two runs would have produced on
the SAME (unflipped) grouping -- isolating the weight-drift channel from the
"the flipped node itself moved bucket" effect the proof already accounts for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import load_flat_state
from src.utils.provenance import build_manifest


@torch.no_grad()
def s_blind_predictions(trainer: FederatedTrainer, client_id: int, flat: torch.Tensor) -> torch.Tensor:
    """Predictions on client_id's full local node set from an s=None forward
    pass at the given flat weight vector -- the same release pathway
    Client._release_pred uses, applied post-hoc to a fixed snapshot."""
    load_flat_state(trainer.ref_model, flat)
    trainer.ref_model.eval()
    d = trainer.clients_data[client_id]
    return trainer.ref_model(d.x, d.edge_index, None).cpu()


def run_variant(seed: int, flip_client: int, flip_local_idx: int | None) -> dict:
    cfg = ExperimentConfig.canonical(seed=seed, device="cpu")
    trainer = FederatedTrainer(cfg)
    if flip_local_idx is not None:
        d = trainer.clients_data[flip_client]
        old = int(d.sensitive_attr[flip_local_idx])
        d.sensitive_attr[flip_local_idx] = 1 - d.sensitive_attr[flip_local_idx]
        # Client.data is the same tensor object post .to("cpu") (no-op copy),
        # but patch explicitly so this does not depend on that implementation
        # detail holding across torch versions.
        trainer.clients[flip_client].data.sensitive_attr[flip_local_idx] = d.sensitive_attr[flip_local_idx]
        flipped = {"client": flip_client, "local_idx": flip_local_idx,
                   "s_before": old, "s_after": int(d.sensitive_attr[flip_local_idx])}
    else:
        flipped = None
    trainer.run()
    return {"trainer": trainer, "flat": trainer.global_flat.clone(), "flipped": flipped}


def probe_one(seed: int, flip_client: int, flip_local_idx: int, dp_sigma_ref: float) -> dict:
    base = run_variant(seed, flip_client, None)
    flipped = run_variant(seed, flip_client, flip_local_idx)

    theta_l2 = float(torch.norm(base["flat"] - flipped["flat"], p=2))

    # Release prediction on the UNFLIPPED grouping (base["trainer"]'s own
    # clients_data, s never touched) at both weight snapshots -- isolates the
    # weight-drift channel from the "node moved bucket" effect.
    pred_base_theta = s_blind_predictions(base["trainer"], flip_client, base["flat"])
    pred_flip_theta = s_blind_predictions(base["trainer"], flip_client, flipped["flat"])

    d = base["trainer"].clients_data[flip_client]
    s = d.sensitive_attr
    n0, n1 = int((s == 0).sum()), int((s == 1).sum())
    mu0_base = float(pred_base_theta[s == 0].mean()) if n0 else float("nan")
    mu1_base = float(pred_base_theta[s == 1].mean()) if n1 else float("nan")
    mu0_flip = float(pred_flip_theta[s == 0].mean()) if n0 else float("nan")
    mu1_flip = float(pred_flip_theta[s == 1].mean()) if n1 else float("nan")
    delta_mu_weight_only = ((mu0_flip - mu0_base) ** 2 + (mu1_flip - mu1_base) ** 2) ** 0.5

    return {
        "seed": seed, "flip_client": flip_client, "flip_local_idx": flip_local_idx,
        "flipped_node": flipped["flipped"],
        "theta_l2_diff": theta_l2,
        "delta_mu_weight_only": delta_mu_weight_only,
        "delta_mu_weight_only_over_sigma_dp": delta_mu_weight_only / dp_sigma_ref if dp_sigma_ref > 0 else None,
        "mu_base": [mu0_base, mu1_base], "mu_flip": [mu0_flip, mu1_flip],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--flip-client", type=int, default=0)
    ap.add_argument("--flip-local-idxs", type=int, nargs="+", default=[0, 5, 10, 15])
    ap.add_argument("--out-json", default="results/revision/theta_sensitivity_probe.json")
    args = ap.parse_args()

    # sigma_DP reference at the canonical operating point (eps=8.0, German),
    # read from one throwaway client rather than recomputed by hand.
    ref_cfg = ExperimentConfig.canonical(seed=args.seeds[0], device="cpu")
    ref_trainer = FederatedTrainer(ref_cfg)
    dp_sigma_ref = ref_trainer.clients[0].dp_sigma

    results = []
    total = len(args.seeds) * len(args.flip_local_idxs)
    i = 0
    for seed in args.seeds:
        for idx in args.flip_local_idxs:
            i += 1
            print(f"[{i}/{total}] seed={seed} flip_local_idx={idx} ...", flush=True)
            try:
                r = probe_one(seed, args.flip_client, idx, dp_sigma_ref)
                print(f"    -> theta_l2={r['theta_l2_diff']:.6f}  "
                      f"delta_mu_weight_only={r['delta_mu_weight_only']:.6f}  "
                      f"(/sigma_dp={r['delta_mu_weight_only_over_sigma_dp']:.4f})", flush=True)
                results.append(r)
            except IndexError:
                print(f"    -> SKIPPED (client {args.flip_client} has < {idx+1} local nodes)")

    ratios = [r["delta_mu_weight_only_over_sigma_dp"] for r in results if r["delta_mu_weight_only_over_sigma_dp"] is not None]
    summary = {
        "dp_sigma_ref": dp_sigma_ref,
        "n_probes": len(results),
        "delta_mu_weight_only_over_sigma_dp": {
            "mean": sum(ratios) / len(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "min": min(ratios) if ratios else None,
        },
    }
    out = {"manifest": build_manifest(experiment="theta_sensitivity_probe"),
           "summary": summary, "probes": results}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[+] Saved theta sensitivity probe to {args.out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
