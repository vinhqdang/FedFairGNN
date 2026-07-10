"""Attribute-inference (reconstruction) attack on the released fairness
statistic, quantifying what FTGD's statistic-level DP actually buys.

Threat model (worst case, favouring the adversary). A client releases the group
means (mu0, mu1) that drive the soft demographic-parity term. A strong,
honest-but-curious server knows every node's model prediction y_hat and the
sensitive attribute of *all but one* target node, and tries to infer the
target's sensitive attribute from the released statistic. This is the canonical
"differencing" reconstruction attack against an aggregate release.

Without noise (epsilon = infinity) the adversary computes, for each hypothesis
s_target in {0,1}, the exact means that hypothesis implies (correctly accounting
for the changing group sizes / denominators), and picks the one matching the
release -- recovering s_target almost perfectly. FTGD instead releases
mu_g + N(0, (z * Delta)^2); the adversary now does a maximum-likelihood guess
under the known Gaussian. We sweep the target epsilon (mapped to the deployed
per-release noise multiplier z by the RDP accountant) and report the attack's
balanced accuracy on *real* Bail predictions.

    python -m experiments.privacy_attack
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ExperimentConfig, set_seed
from src.federated.trainer import FederatedTrainer
from src.federated.client import load_flat_state
from src.trust.privacy import calibrate_noise_multiplier
from experiments.methods import apply_method

FIG = "manuscript/figures"
TAB = "manuscript/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def real_predictions(dataset="bail", seed=0):
    """Train the flagship briefly and return (y_hat, s) on a real client's
    training nodes -- the exact objects the FTGD statistic is computed from."""
    set_seed(seed)
    cfg = ExperimentConfig(dataset=dataset, seed=seed, rounds=15, num_clients=5,
                           local_epochs=2, hidden_channels=64)
    apply_method(cfg, "fedfairgnn-nodp")
    tr = FederatedTrainer(cfg)
    tr.run()
    load_flat_state(tr.ref_model, tr.global_flat.to(tr.device))
    tr.ref_model.eval()
    d = tr.clients_data[0].to(tr.device)
    m = d.train_mask
    with torch.no_grad():
        yhat = tr.ref_model(d.x, d.edge_index, d.sensitive_attr)[m].cpu().numpy()
    s = d.sensitive_attr[m].cpu().numpy().astype(int)
    return yhat.astype(float), s


def attack(yhat, s, z, n_trials=4000, rng=None):
    """Balanced-accuracy of the MAP differencing adversary at per-release noise
    multiplier z (sigma_g = z * Delta on each released mean). z=0 -> no noise."""
    rng = rng or np.random.default_rng(0)
    idx0 = np.where(s == 0)[0]
    idx1 = np.where(s == 1)[0]
    n = len(s)
    S0 = yhat[idx0].sum(); n0 = len(idx0)
    S1 = yhat[idx1].sum(); n1 = len(idx1)
    Delta = math.sqrt(1.0 / n0 ** 2 + 1.0 / n1 ** 2)
    sigma = z * Delta
    correct = tot = 0
    # balance targets across the two true classes
    per = n_trials // 2
    targets = np.concatenate([rng.choice(idx0, per), rng.choice(idx1, per)])
    for t in targets:
        st = int(s[t]); yt = yhat[t]
        # group sums/counts of everyone EXCEPT the target
        s0o = S0 - (yt if st == 0 else 0.0); n0o = n0 - (1 if st == 0 else 0)
        s1o = S1 - (yt if st == 1 else 0.0); n1o = n1 - (1 if st == 1 else 0)
        # means implied by each hypothesis (correct denominators)
        mu0_h0 = (s0o + yt) / (n0o + 1); mu1_h0 = s1o / max(n1o, 1)
        mu0_h1 = s0o / max(n0o, 1);     mu1_h1 = (s1o + yt) / (n1o + 1)
        # the true released statistic (+ noise)
        mu0 = S0 / n0; mu1 = S1 / n1
        if sigma > 0:
            o0 = mu0 + rng.normal(0, sigma); o1 = mu1 + rng.normal(0, sigma)
            # log-likelihood of the observation under each hypothesis
            ll0 = -((o0 - mu0_h0) ** 2 + (o1 - mu1_h0) ** 2)
            ll1 = -((o0 - mu0_h1) ** 2 + (o1 - mu1_h1) ** 2)
            guess = 0 if ll0 >= ll1 else 1
        else:
            # exact release: pick the hypothesis whose means match exactly
            d0 = abs(mu0 - mu0_h0) + abs(mu1 - mu1_h0)
            d1 = abs(mu0 - mu0_h1) + abs(mu1 - mu1_h1)
            guess = 0 if d0 <= d1 else 1
        correct += (guess == st); tot += 1
    return correct / tot


def main():
    yhat, s = real_predictions()
    n0 = int((s == 0).sum()); n1 = int((s == 1).sum())
    base_rate = max(n0, n1) / (n0 + n1)          # majority-class baseline
    # per-release noise is calibrated for the DEPLOYED accounting budget (a
    # 50-round x 2-epoch Bail deployment = 100 statistic releases), independent
    # of the short training used only to obtain realistic predictions above.
    steps = 50 * 2
    rng = np.random.default_rng(0)

    eps_grid = [None, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5]   # None = no DP (eps=inf)
    rows = []
    for eps in eps_grid:
        z = 0.0 if eps is None else calibrate_noise_multiplier(eps, steps)
        acc = attack(yhat, s, z, rng=rng)
        # advantage over guessing the majority class, normalised to [0,1]
        adv = max(0.0, (acc - base_rate) / (1.0 - base_rate))
        rows.append((eps, z, acc, adv))
        tag = "inf" if eps is None else f"{eps:g}"
        print(f"eps={tag:>4}  z={z:6.2f}  attack_acc={acc:.3f}  norm_adv={adv:.3f}")

    # --- figure: attack accuracy vs epsilon (log x) ---
    finite = [(e, a) for e, _, a, _ in rows if e is not None]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xs = [e for e, _ in finite]; ys = [a for _, a in finite]
    ax.plot(xs, ys, "o-", color="#1b7837", label="FTGD (release + DP noise)")
    nodp = [a for e, _, a, _ in rows if e is None][0]
    ax.axhline(nodp, ls="--", color="#762a83",
               label=f"no DP (exact release): {nodp:.2f}")
    ax.axhline(base_rate, ls=":", color="#999999",
               label=f"chance (base rate): {base_rate:.2f}")
    ax.set_xscale("log"); ax.set_xlabel("privacy budget $\\epsilon$")
    ax.set_ylabel("sensitive-attribute inference accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Attribute-inference attack on the released statistic (Bail)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "privacy_attack.pdf"))
    plt.close(fig)
    print("[fig] privacy_attack.pdf")

    # --- table ---
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Release & $\\epsilon$ & Noise mult.\\ $z$ & Inference acc. \\\\",
             "\\midrule"]
    for eps, z, acc, adv in rows:
        if eps is None:
            lines.append(f"Exact ($s$-statistic, no DP) & $\\infty$ & 0 & {acc:.3f} \\\\")
        else:
            lines.append(f"FTGD & {eps:g} & {z:.1f} & {acc:.3f} \\\\")
    lines.append("\\midrule")
    lines.append(f"\\emph{{Chance (majority base rate)}} & --- & --- & {base_rate:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "privacy_attack.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] privacy_attack.tex")


if __name__ == "__main__":
    main()
