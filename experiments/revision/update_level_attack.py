"""Update-Level Attribute Inference Attack (Issue I1, Stanford Q10, DeepSeek 3.1).

Quantifies sensitive-attribute leakage through the transmitted model updates

    g_k = theta_global - theta_local_k

versus the released fairness statistics (mu0, mu1), and prices each mitigation
in units of task utility.

Regimes compared (all five rows are MEASURED in this script -- see below):
  1. Exact statistic release (unnoised mu0, mu1)
  2. FTGD statistic-level DP at the configured epsilon
  3. Model-update channel (TrustFedGNN, no parameter-level DP)
  4. Model-update channel + client-level DP-SGD noise at the same epsilon
  5. Random-guess baseline (0.500 by definition)

What changed, and why it mattered
---------------------------------
* Rows 1 and 2 used to be the literals ``1.000`` and ``0.512``, transcribed
  from ``experiments/privacy_attack.py``. They are now computed here by calling
  that script's ``attack()`` adversary on predictions from the model this
  script actually trains. (The MAP adversary emits a hard 0/1 guess on
  class-balanced targets, so its ROC-AUC equals its balanced accuracy -- the
  two names describe the same number for this probe.)
* Row 4's noise used an arbitrary ``sigma = 1.5 * clip_c`` while the imported
  ``calibrate_noise_multiplier`` was never called. The noise multiplier is now
  solved for the target epsilon over the run's real release count
  (``rounds * local_epochs``, matching the trainer's accountant), and updates
  are clipped to C before noising, as DP-SGD requires.
* The probe used ``StratifiedKFold(shuffle=True)``. The label
  (a client's majority sensitive attribute) is CONSTANT per client across all
  rounds and seeds, so shuffled folds put the same client's updates in train
  and test and the probe could win by recognising the client rather than by
  extracting the attribute. Folds are now grouped by client identity
  ``(seed, client_index)``, which is the confound. The reported AUC is
  correspondingly lower -- and honest.
* The "utility cost" column was the strings "Severe"/"Negligible". It is now
  the measured global test-AUC delta of each mitigation against the no-DP run,
  with the qualitative word chosen by a documented threshold (see
  ``_utility_bucket``).

Outputs:
  - results/revision/update_level_attack.json
  - manuscript/tables/revision/update_attack.tex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import flatten_state, load_flat_state
from src.trust.privacy import calibrate_noise_multiplier
from experiments.privacy_attack import attack as statistic_attack

# Utility-cost buckets, in absolute global test-AUC lost against the no-DP run.
# Chosen so that a drop inside typical seed-to-seed noise reads as negligible
# and a drop that would change a leaderboard position reads as severe.
UTILITY_BUCKETS = ((0.01, "Negligible"), (0.05, "Moderate"), (float("inf"), "Severe"))


def _utility_bucket(delta_auc: float) -> str:
    """Qualitative label for a measured utility cost (AUC lost vs. no DP).

    < 0.01 AUC -> "Negligible"  (inside run-to-run noise)
    < 0.05 AUC -> "Moderate"
    otherwise  -> "Severe"
    A cost <= 0 (the mitigation did not hurt, or helped) reports as "None".
    """
    if delta_auc <= 0:
        return "None"
    for hi, name in UTILITY_BUCKETS:
        if delta_auc < hi:
            return name
    return "Severe"                                       # pragma: no cover


def _utility_cell(delta_auc: float) -> str:
    """Human-readable utility cost: the bucket plus the number behind it."""
    if delta_auc <= 0:
        return f"None ({delta_auc:+.3f} AUC)"
    return f"{_utility_bucket(delta_auc)} ({-delta_auc:+.3f} AUC)"


def _base_cfg(dataset, seed, rounds, num_clients, **over) -> ExperimentConfig:
    kw = dict(dataset=dataset, seed=seed, rounds=rounds, num_clients=num_clients,
              model="trustfedgnn", aggregator="fedavg", local_fairness=True,
              dp_enabled=False)
    kw.update(over)
    return ExperimentConfig.canonical(**kw)


def collect_update_dataset(dataset="bail", seeds=(42, 43, 44), rounds=15, num_clients=10):
    """Run federated rounds and collect (update_vector, majority_s, client_id).

    ``groups`` identifies the client that produced each row as
    ``(seed, client_index)``. It is the grouping variable the probe must respect:
    a client's label never changes across rounds, so any fold split that is not
    grouped leaks the answer.

    Also returns, per seed, the material for the statistic-channel attack
    (predictions + sensitive attributes on a real client's training nodes) and
    the run's global test AUC, which is the no-DP utility baseline.
    """
    X_updates, y_sens, groups = [], [], []
    stat_material, baseline_aucs = [], []

    for s in seeds:
        cfg = _base_cfg(dataset, s, rounds, num_clients)
        trainer = FederatedTrainer(cfg)

        client_labels = []
        for c in trainer.clients:
            s_tensor = c.data.sensitive_attr[c.data.train_mask]
            client_labels.append(int((s_tensor.float().mean() >= 0.5).item()))

        for t in range(rounds):
            g_old = trainer.global_flat.clone()
            trainer._round(t)
            for idx, c in enumerate(trainer.clients):
                w_local = flatten_state(c.model.state_dict())
                X_updates.append((g_old - w_local).detach().cpu().numpy())
                y_sens.append(client_labels[idx])
                groups.append(f"s{s}_c{idx}")

        # Utility baseline and the statistic channel's raw material, taken from
        # the same trained model (nothing here is copied from another script).
        baseline_aucs.append(float(trainer.evaluate_global()["auc"]))
        load_flat_state(trainer.ref_model, trainer.global_flat.to(trainer.device))
        trainer.ref_model.eval()
        d = trainer.clients_data[0].to(trainer.device)
        m = d.train_mask
        with torch.no_grad():
            yhat = trainer.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu().numpy()
        stat_material.append((yhat.astype(float),
                              d.sensitive_attr[m].cpu().numpy().astype(int)))

    return (np.array(X_updates, dtype=np.float32), np.array(y_sens, dtype=int),
            np.array(groups), stat_material, float(np.mean(baseline_aucs)))


def evaluate_probe_auc(X, y, groups, n_splits=5):
    """Grouped 5-fold AUC of an L2-regularised logistic-regression probe.

    Folds are split by ``groups`` (client identity), never by row. With row-wise
    stratified folds the same client's updates appear in both train and test and
    the probe scores client re-identification, not attribute inference.
    ``StratifiedGroupKFold`` keeps the grouping while balancing the label across
    folds so the per-fold AUC stays defined; plain ``GroupKFold`` is the
    fallback on older scikit-learn.
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    except ImportError:                                   # pragma: no cover
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_splits)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X_norm = X / norms

    aucs = []
    for train_idx, test_idx in splitter.split(X_norm, y, groups):
        X_train, X_test = X_norm[train_idx], X_norm[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue
        # sanity: the grouping must actually hold
        assert not (set(groups[train_idx]) & set(groups[test_idx]))
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf.fit(X_train, y_train)
        try:
            aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        except ValueError:
            pass
    if not aucs:
        # Not a result of 0.50 -- an absence of one. Say so instead of writing a
        # chance-level number into the table as if it had been measured.
        warnings.warn(
            "update-level probe: no cross-validation fold had both classes "
            "present in train and test, so the attack AUC is NOT MEASURABLE on "
            "this configuration. This happens when every client shares the same "
            "majority sensitive attribute (german and credit partition this way; "
            "bail does not). Reporting NaN.", RuntimeWarning, stacklevel=2)
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def evaluate_probe_auc_ungrouped(X, y, n_splits=5):
    """The OLD (leaky) row-wise stratified probe, kept only as a diagnostic.

    Reported next to the grouped figure so the size of the client-identity
    confound is visible rather than asserted: this splitter puts the same
    client's updates on both sides of the fold, and a client's label never
    changes, so it measures re-identification.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X_norm = X / norms
    aucs = []
    for tr, te in skf.split(X_norm, y):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        clf.fit(X_norm[tr], y[tr])
        try:
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X_norm[te])[:, 1]))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def measure_utility(dataset, seeds, rounds, num_clients, dp_epsilon):
    """Global test AUC under no DP, statistic-level DP (FTGD) and DP-SGD."""
    out = {}
    for name, over in (("ftgd", dict(dp_enabled=True, dp_mode="ftgd",
                                     dp_epsilon=dp_epsilon, dp_delta=1e-5)),
                       ("dpsgd", dict(dp_enabled=True, dp_mode="gradient",
                                      dp_epsilon=dp_epsilon, dp_delta=1e-5))):
        aucs = []
        for s in seeds:
            tr = FederatedTrainer(_base_cfg(dataset, s, rounds, num_clients, **over))
            aucs.append(float(tr.run(verbose=False)["final"]["auc"]))
        out[name] = float(np.mean(aucs))
        print(f"[*] Utility with {name}: global AUC = {out[name]:.4f}", flush=True)
    return out


def run_update_attack_experiment(out_json="results/revision/update_level_attack.json",
                                 out_tex="manuscript/tables/revision/update_attack.tex",
                                 dataset="bail", seeds=(42, 43), rounds=12,
                                 num_clients=10, dp_epsilon=8.0, dp_delta=1e-5,
                                 clip_c=1.0):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    t0 = time.perf_counter()

    print(f"[*] Collecting transmitted model updates on {dataset}...", flush=True)
    X_updates, y_sens, groups, stat_material, auc_nodp_utility = \
        collect_update_dataset(dataset=dataset, seeds=seeds, rounds=rounds,
                               num_clients=num_clients)
    print(f"[*] Collected {len(X_updates)} updates of dimension {X_updates.shape[1]} "
          f"from {len(set(groups))} distinct clients. "
          f"Class balance: {np.mean(y_sens):.2f}", flush=True)

    # The number of privatised releases the deployed run actually makes: the
    # trainer's accountant steps local_epochs times per round.
    steps = int(rounds * _base_cfg(dataset, seeds[0], rounds, num_clients).local_epochs)
    z_stat = calibrate_noise_multiplier(dp_epsilon, steps, delta=dp_delta)
    print(f"[*] Noise multiplier for eps={dp_epsilon} over {steps} releases: "
          f"z = {z_stat:.3f}", flush=True)

    # --- rows 1 & 2: the fairness-statistic channel, computed here ------------
    rng = np.random.default_rng(0)
    exact, ftgd = [], []
    for yhat, s in stat_material:
        exact.append(statistic_attack(yhat, s, 0.0, rng=rng))
        ftgd.append(statistic_attack(yhat, s, z_stat, rng=rng))
    auc_exact_stat = float(np.mean(exact))
    auc_ftgd_stat = float(np.mean(ftgd))
    print(f"[*] Statistic channel: exact release AUC={auc_exact_stat:.4f}, "
          f"FTGD (eps={dp_epsilon}) AUC={auc_ftgd_stat:.4f}", flush=True)

    # --- row 3: the update channel, no parameter DP ---------------------------
    auc_update_nodp, n_folds = evaluate_probe_auc(X_updates, y_sens, groups)
    auc_update_nodp_leaky = evaluate_probe_auc_ungrouped(X_updates, y_sens)
    print(f"[*] Probe AUC on unnoised model updates: {auc_update_nodp:.4f} "
          f"({n_folds} grouped folds)  [ungrouped/leaky, for reference: "
          f"{auc_update_nodp_leaky:.4f}]", flush=True)

    # --- row 4: the update channel under client-level DP-SGD ------------------
    # DP-SGD releases clip(g, C) + N(0, (z*C)^2 I): clip first, then add noise at
    # the multiplier the accountant says eps buys. (Previously an arbitrary
    # sigma = 1.5 * C, with calibrate_noise_multiplier imported but unused.)
    row_norms = np.linalg.norm(X_updates, axis=1, keepdims=True) + 1e-12
    X_clipped = X_updates * np.minimum(1.0, clip_c / row_norms)
    sigma_param = z_stat * clip_c
    noise = np.random.default_rng(0).normal(
        0, sigma_param, size=X_clipped.shape).astype(np.float32)
    auc_update_dp, _ = evaluate_probe_auc(X_clipped + noise, y_sens, groups)
    print(f"[*] Probe AUC on DP-SGD-noised updates (C={clip_c}, sigma={sigma_param:.3f}): "
          f"{auc_update_dp:.4f}", flush=True)

    # --- the utility column ---------------------------------------------------
    util = measure_utility(dataset, seeds, rounds, num_clients, dp_epsilon)
    cost_ftgd = auc_nodp_utility - util["ftgd"]
    cost_dpsgd = auc_nodp_utility - util["dpsgd"]

    results = {
        "dataset": dataset,
        "seeds": list(seeds),
        "rounds": rounds,
        "num_clients": num_clients,
        "total_updates": int(len(X_updates)),
        "param_dim": int(X_updates.shape[1]),
        "distinct_clients": int(len(set(groups.tolist()))),
        "cv": "StratifiedGroupKFold(n_splits=5) grouped by (seed, client index)",
        "dp_epsilon": dp_epsilon,
        "dp_delta": dp_delta,
        "dp_releases_accounted": steps,
        "noise_multiplier_z": z_stat,
        "dpsgd_clip_c": clip_c,
        "dpsgd_sigma": sigma_param,
        "utility_auc": {"no_dp": auc_nodp_utility, **util},
        "utility_cost_auc": {"ftgd": cost_ftgd, "dpsgd": cost_dpsgd},
        "utility_buckets": {"negligible_below": 0.01, "moderate_below": 0.05},
        "probe_folds_evaluated": int(n_folds),
        "probe_measurable": bool(n_folds > 0),
        # diagnostic only -- the inflation the old row-wise StratifiedKFold produced
        "attack_auc_update_nodp_ungrouped_leaky": auc_update_nodp_leaky,
        "wall_clock_s": time.perf_counter() - t0,
        "channels": [
            {"channel": "Fairness Statistic (Unnoised)", "dp_level": "None",
             "probe": "MAP differencing", "attack_auc": auc_exact_stat,
             "utility_impact": _utility_cell(0.0)},
            {"channel": "Fairness Statistic (FTGD)",
             "dp_level": f"eps={dp_epsilon} (stat)", "probe": "MAP differencing",
             "attack_auc": auc_ftgd_stat, "utility_impact": _utility_cell(cost_ftgd)},
            {"channel": "Model Updates (TrustFedGNN)", "dp_level": "None (stat-only)",
             "probe": "Grouped linear probe", "attack_auc": auc_update_nodp,
             "utility_impact": _utility_cell(0.0)},
            {"channel": "Model Updates + Client DP-SGD",
             "dp_level": f"eps={dp_epsilon} (param)", "probe": "Grouped linear probe",
             "attack_auc": auc_update_dp, "utility_impact": _utility_cell(cost_dpsgd)},
            {"channel": "Theoretical Baseline", "dp_level": "Infinite Noise",
             "probe": "Random Chance", "attack_auc": 0.500,
             "utility_impact": "Total collapse"},
        ],
    }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved update attack JSON to {out_json}")

    eps_s = f"{dp_epsilon:g}"
    closes = ("closes" if auc_ftgd_stat < auc_exact_stat - 0.05 else "does not close")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{Empirical Attribute Inference Attack Success Across Release "
        f"Channels ({dataset.capitalize()}).}}",
        "Sensitive-attribute leakage ($S_k$) through the released fairness statistic "
        "$(\\mu_0, \\mu_1)$ versus the transmitted parameter updates "
        "$g_k = \\theta_{\\text{global}} - \\theta_{\\text{local}, k}$. "
        f"FTGD {closes} the fairness-statistic channel "
        f"(${auc_exact_stat:.3f} \\to {auc_ftgd_stat:.3f}$) at a measured utility cost of "
        f"${-cost_ftgd:+.3f}$ AUC, while parameter updates retain residual correlation "
        f"(AUC $\\approx {auc_update_nodp:.3f}$), so FTGD is \\emph{{complementary}} to "
        "update-level encryption / Secure Aggregation rather than an all-parameter DP "
        "replacement. "
        "Update-channel probes use 5-fold cross-validation \\emph{grouped by client}: a "
        "client's majority attribute is constant across rounds, so ungrouped folds measure "
        "client re-identification instead of attribute inference. "
        "Utility cost is the measured drop in global test AUC against the no-DP run "
        f"({auc_nodp_utility:.3f}); \\emph{{Negligible}} $<0.01$, \\emph{{Moderate}} $<0.05$, "
        "\\emph{Severe} otherwise.}",
        "\\label{tab:update_attack}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Observation Channel} & \\textbf{Privacy Mechanism} & "
        "\\textbf{Attack AUC $\\downarrow$} & \\textbf{Utility Cost} \\\\",
        "\\midrule",
        f"Fairness Statistic (Raw) & None & {auc_exact_stat:.3f} & "
        f"{_utility_cell(0.0)} \\\\",
        f"Fairness Statistic (FTGD) & $(\\epsilon={eps_s},\\delta=10^{{-5}})$ & "
        f"\\textbf{{{auc_ftgd_stat:.3f}}} & {_utility_cell(cost_ftgd)} \\\\",
        f"Model Parameter Updates & None (TrustFedGNN default) & {auc_update_nodp:.3f} & "
        f"{_utility_cell(0.0)} \\\\",
        f"Model Parameter Updates & Client DP-SGD ($\\epsilon={eps_s}$) & "
        f"{auc_update_dp:.3f} & {_utility_cell(cost_dpsgd)} \\\\",
        "Random Guess Baseline & -- & 0.500 & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    # LaTeX-escape the measured utility strings' parentheses are fine; only '%'
    # would need escaping and we emit none.
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX update attack table to {out_tex}")
    return results


def main():
    ap = argparse.ArgumentParser(description="Update-level attribute inference attack.")
    ap.add_argument("--dataset", default="bail")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--dp-epsilon", type=float, default=8.0)
    ap.add_argument("--out-json", default="results/revision/update_level_attack.json")
    ap.add_argument("--out-tex", default="manuscript/tables/revision/update_attack.tex")
    a = ap.parse_args()
    run_update_attack_experiment(out_json=a.out_json, out_tex=a.out_tex,
                                 dataset=a.dataset, seeds=tuple(a.seeds),
                                 rounds=a.rounds, num_clients=a.num_clients,
                                 dp_epsilon=a.dp_epsilon)


if __name__ == "__main__":
    main()
