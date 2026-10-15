"""Privacy experiments for the CLAMP paper.

(a) Differencing attack on the statistics channel (empirical check of Theorem 1).
    For a target validation node v of client k, the adversary knows every other
    validation node's attribute and s-blind prediction, and sees the client's
    released 4-vector (S0, S1, N0, N1) with N(0, sigma^2) noise per coordinate.
    Flipping s_v moves p_v between the S-sums and adds 1 to one N-count, so the
    two hypotheses differ by d = (-p, p, -1, 1). The likelihood-ratio test for
    equal-covariance Gaussians is a threshold on (x - c) . d, with c the midpoint.
    Reported: balanced accuracy averaged over targets, for one release (sigma) and
    for R releases with the same vector (worst case; each noise has std
    sigma / sqrt(R)), against the bound (e^eps + delta) / (1 + e^eps).

    The sweep varies only the release noise (sigma from eps and R = ceil(T/P)).
    The final model is the one trained at the campaign default (dp_epsilon = 8).

(b) Attribute probe on the final global model. Logistic regression on the
    penultimate embeddings of test nodes, fit on 50% and scored (AUC) on the rest.

Usage:
    python -m experiments.clamp_privacy --datasets german,bail --seeds 0-2 --workers 4
    python -m experiments.clamp_privacy --render-only --out results/clamp/privacy
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
import traceback
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

from src.trust.privacy import calibrate_noise_multiplier

OUT_DIR = os.path.join("results", "clamp", "privacy")
EPS_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, float("inf")]
PROBE_METHODS = ["fedavg-gat", "clamp-clipmean", "clamp"]


# --------------------------------------------------------------------------- #
# (a) Differencing attack
# --------------------------------------------------------------------------- #
def release_inputs(trainer) -> List[tuple]:
    """Per client: (s-blind validation predictions, true attributes), with the
    final global model. Mirrors Client.group_stats (s-blind pass)."""
    out = []
    gp = trainer.global_flat
    for c in trainer.clients:
        c.set_flat(gp)
        d = c.data
        c.model.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pred = c.model(d.x, d.edge_index, None) if c._model_uses_s \
                else c.model(d.x, d.edge_index, d.sensitive_attr)
        m = d.val_mask
        out.append((pred[m].double().numpy(), d.sensitive_attr[m].numpy()))
    return out


def sigma_for_eps(eps: float, R: int, delta: float) -> float:
    """sigma_s = 2 z, with z calibrated for R releases (Theorem 1). eps = inf -> 0."""
    if math.isinf(eps):
        return 0.0
    return 2.0 * calibrate_noise_multiplier(eps, steps=R, delta=delta)


def dp_bound(eps: float, delta: float) -> float:
    if math.isinf(eps):
        return 1.0
    return (math.exp(eps) + delta) / (1.0 + math.exp(eps))


def attack_balanced_accuracy(inputs: List[tuple], sigma: float, releases: int,
                             n_targets: int, trials: int, seed: int) -> Dict[str, float]:
    """Balanced accuracy of the LR test, averaged over random targets.

    ``releases`` = number of independent releases of the same vector; the noise
    on their mean has std sigma / sqrt(releases). Returns Monte Carlo and
    analytic values (analytic: Phi(||d|| / (2 sigma_eff)) per target).
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    pool = [(k, i) for k, (p, _) in enumerate(inputs) for i in range(len(p))]
    if not pool:
        return {"mc": math.nan, "analytic": math.nan, "n_targets": 0}
    if len(pool) > n_targets:
        pick = rng.choice(len(pool), n_targets, replace=False)
        pool = [pool[j] for j in pick]
    ks = np.array([k for k, _ in pool])
    iv = np.array([i for _, i in pool])
    tot = np.array([[p[s == 0].sum(), p[s == 1].sum(), float((s == 0).sum()),
                     float((s == 1).sum())] for p, s in inputs])
    pv = np.array([inputs[k][0][i] for k, i in pool])
    sv = np.array([inputs[k][1][i] for k, i in pool])
    z = np.zeros_like(pv)
    c0 = np.stack([pv, z, z + 1.0, z], 1)        # contribution if s_v = 0
    c1 = np.stack([z, pv, z, z + 1.0], 1)        # contribution if s_v = 1
    base = tot[ks] - np.where(sv[:, None] == 0, c0, c1)
    m0, m1 = base + c0, base + c1
    d = m1 - m0
    mid = (m0 + m1) / 2.0
    T, M = len(pool), trials
    s_true = np.tile(np.arange(M) % 2, (T, 1))            # balanced hypotheses
    mean = np.where(s_true[..., None] == 1, m1[:, None, :], m0[:, None, :])
    sig_eff = sigma / math.sqrt(releases)
    x = mean + (rng.normal(size=mean.shape) * sig_eff if sig_eff > 0 else 0.0)
    score = ((x - mid[:, None, :]) * d[:, None, :]).sum(-1)
    pred = (score > 0).astype(int)
    correct = pred == s_true
    per_target = 0.5 * (correct[:, s_true[0] == 0].mean(1) + correct[:, s_true[0] == 1].mean(1))
    if sig_eff > 0:
        analytic = norm.cdf(np.sqrt((d ** 2).sum(1)) / (2.0 * sig_eff)).mean()
    else:
        analytic = 1.0
    return {"mc": float(per_target.mean()), "analytic": float(analytic), "n_targets": T}


# --------------------------------------------------------------------------- #
# (b) Attribute probe
# --------------------------------------------------------------------------- #
def _probe_fit(X: np.ndarray, S: np.ndarray, seed: int) -> float:
    """Logistic regression for s on a random 50% of the rows, AUC on the other 50%."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(S))
    half = len(S) // 2
    tr, te = perm[:half], perm[half:]
    if len(np.unique(S[tr])) < 2 or len(np.unique(S[te])) < 2:
        return math.nan
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    clf = LogisticRegression(max_iter=1000)
    clf.fit((X[tr] - mu) / sd, S[tr])
    return float(roc_auc_score(S[te], clf.predict_proba((X[te] - mu) / sd)[:, 1]))


def _test_pool(trainer, features_fn):
    """Stack per-client test-node features (features_fn(d) -> [n, k]) and attributes."""
    X, S = [], []
    for d in trainer.clients_data:
        if int(d.test_mask.sum()) == 0:
            continue
        X.append(features_fn(d.to(trainer.device))[d.test_mask.to(trainer.device)].cpu().numpy())
        S.append(d.sensitive_attr[d.test_mask].cpu().numpy())
    return np.concatenate(X), np.concatenate(S)


@torch.no_grad()
def probe_auc(trainer, seed: int) -> float:
    """AUC of a logistic regression that predicts s from penultimate test-node
    embeddings of the final global model. Fit on a random 50%, scored on the rest."""
    from src.federated.client import load_flat_state

    model = trainer.ref_model
    load_flat_state(model, trainer.global_flat.to(trainer.device))
    model.eval()
    feats = []
    hook = model.classifier.register_forward_pre_hook(lambda m, inp: feats.append(inp[0]))

    def emb(d):
        feats.clear()
        model(d.x, d.edge_index, d.sensitive_attr)
        return feats[-1]
    try:
        X, S = _test_pool(trainer, emb)
    finally:
        hook.remove()
    return _probe_fit(X, S, seed)


@torch.no_grad()
def control_probes(trainer, seed: int) -> Dict[str, float]:
    """Two controls with no training. raw-features: the probe on the node features x.
    untrained-gat: the probe on the penultimate embeddings of a freshly initialised GAT
    (same architecture as the trained model, seeded)."""
    from src.models import build_model
    from src.federated.client import load_flat_state
    raw_X, raw_S = _test_pool(trainer, lambda d: d.x)
    out = {"raw-features": _probe_fit(raw_X, raw_S, seed)}

    torch.manual_seed(seed)
    model = build_model("gat", trainer.in_channels, trainer.cfg).to(trainer.device)
    model.eval()
    feats = []
    hook = model.classifier.register_forward_pre_hook(lambda m, inp: feats.append(inp[0]))

    def emb(d):
        feats.clear()
        model(d.x, d.edge_index, d.sensitive_attr)
        return feats[-1]
    try:
        X, S = _test_pool(trainer, emb)
    finally:
        hook.remove()
    out["untrained-gat"] = _probe_fit(X, S, seed)
    return out


def write_control_probes(datasets, seeds, out_dir: str) -> None:
    """Add the two control rows (per dataset and seed) to probe.csv, replacing old ones."""
    import csv
    from experiments.run_clamp_suite import _job
    from experiments.run_experiment import build_config
    from src.federated.trainer import FederatedTrainer
    path = os.path.join(out_dir, "probe.csv")
    kept = []
    if os.path.exists(path):
        with open(path) as f:
            kept = [r for r in csv.DictReader(f) if r["method"] not in ("raw-features", "untrained-gat")]
    new = []
    for ds in datasets:
        for seed in seeds:
            job = _job("privacy", "clamp", ds, seed, {})
            cfg = build_config("clamp", ds, seed, **job["overrides"])
            trainer = FederatedTrainer(cfg)   # partition only; nothing is trained here
            for method, auc in control_probes(trainer, seed).items():
                new.append({"dataset": ds, "seed": seed, "method": method, "auc": auc})
                print(f"control {method} {ds} s{seed}: AUC {auc:.4f}", flush=True)
    rows = [{"dataset": r["dataset"], "seed": int(r["seed"]), "method": r["method"],
             "auc": float(r["auc"])} for r in kept] + new
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "method", "auc"])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _train(method: str, dataset: str, seed: int, rounds: Optional[int]):
    from experiments.run_clamp_suite import _job
    from experiments.run_experiment import build_config
    from src.federated.trainer import FederatedTrainer
    job = _job("privacy", method, dataset, seed, {"rounds": rounds} if rounds else {})
    cfg = build_config(method, dataset, seed, **job["overrides"])
    trainer = FederatedTrainer(cfg)
    trainer.run()
    return cfg, trainer


def _run_job(args: Dict) -> Dict:
    """Worker: returns {"attack": [...], "probe": [...]} rows, or an error record."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    method, dataset, seed = args["method"], args["dataset"], args["seed"]
    try:
        cfg, trainer = _train(method, dataset, seed, args["rounds"])
        out = {"attack": [], "probe": []}
        out["probe"].append({"dataset": dataset, "seed": seed, "method": method,
                             "auc": probe_auc(trainer, seed)})
        if method == "clamp":
            R = math.ceil(cfg.rounds / cfg.r2_stat_period)
            inputs = release_inputs(trainer)
            for eps in EPS_GRID:
                sig = sigma_for_eps(eps, R, cfg.dp_delta)
                single = attack_balanced_accuracy(inputs, sig, 1, args["targets"],
                                                  args["trials"], seed)
                comp = attack_balanced_accuracy(inputs, sig, R, args["targets"],
                                                args["trials"], seed)
                out["attack"].append({
                    "dataset": dataset, "seed": seed, "eps": eps, "P": cfg.r2_stat_period,
                    "R": R, "sigma": sig, "bound": dp_bound(eps, cfg.dp_delta),
                    "acc_single_mc": single["mc"], "acc_single_analytic": single["analytic"],
                    "acc_composed_mc": comp["mc"], "acc_composed_analytic": comp["analytic"],
                    "n_targets": single["n_targets"]})
        return out
    except Exception:
        return {"error": traceback.format_exc(), "method": method,
                "dataset": dataset, "seed": seed}


def _parse_seeds(text: str) -> List[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _write_outputs(rows_attack: List[Dict], rows_probe: List[Dict], out_dir: str) -> None:
    """Per-seed CSVs (attack.csv, probe.csv), then the LaTeX tables from them."""
    import csv
    from experiments.make_clamp_tables import write_privacy_tables
    os.makedirs(out_dir, exist_ok=True)
    if rows_attack:
        with open(os.path.join(out_dir, "attack.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_attack[0].keys()))
            w.writeheader()
            w.writerows(rows_attack)
    if rows_probe:
        with open(os.path.join(out_dir, "probe.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_probe[0].keys()))
            w.writeheader()
            w.writerows(rows_probe)
    write_privacy_tables(rows_attack, rows_probe, out_dir)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="german,bail")
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--rounds", type=int, default=None, help="default: the campaign's 100")
    p.add_argument("--targets", type=int, default=2000)
    p.add_argument("--trials", type=int, default=200, help="even number of noise draws per target")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--controls-only", action="store_true",
                   help="add the raw-feature and untrained-GAT probe controls (no training)")
    p.add_argument("--render-only", action="store_true",
                   help="re-render the privacy tables from the existing attack.csv / probe.csv")
    args = p.parse_args(argv)
    if args.controls_only:
        write_control_probes(args.datasets.split(","), _parse_seeds(args.seeds), args.out)
        from experiments.make_clamp_tables import render_privacy_from_csv
        render_privacy_from_csv(args.out)
        return
    if args.render_only:
        from experiments.make_clamp_tables import render_privacy_from_csv
        render_privacy_from_csv(args.out)
        return

    jobs = []
    for ds in args.datasets.split(","):
        for seed in _parse_seeds(args.seeds):
            for m in PROBE_METHODS:
                jobs.append({"method": m, "dataset": ds, "seed": seed, "rounds": args.rounds,
                             "targets": args.targets, "trials": args.trials})
    print(f"{len(jobs)} training jobs with {args.workers} workers", flush=True)

    attack_rows, probe_rows, errors = [], [], []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=1), 1):
            if "error" in res:
                errors.append(res)
                print(f"[{i}/{len(jobs)}] ERROR {res['method']} {res['dataset']} s{res['seed']}")
            else:
                attack_rows += res["attack"]
                probe_rows += res["probe"]
                print(f"[{i}/{len(jobs)}] ok {res['probe'][0]['method']} "
                      f"{res['probe'][0]['dataset']} s{res['probe'][0]['seed']} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    _write_outputs(attack_rows, probe_rows, args.out)
    if errors:
        with open(os.path.join(args.out, "errors.txt"), "w") as f:
            for e in errors:
                f.write(f"{e['method']} {e['dataset']} s{e['seed']}\n{e['error']}\n")
        print(f"{len(errors)} jobs failed; see {args.out}/errors.txt")
    print(f"wrote outputs to {args.out}")


if __name__ == "__main__":
    main()
