"""Compute calibration/uncertainty + sustainability + composite trust for key
methods on Bail, and write manuscript/tables/efficiency.tex. Also emits an
EU AI Act / NIST model card for the flagship FedFairGNN run.

    python -m experiments.trust_eval
"""
from __future__ import annotations

import os

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import load_flat_state
from src.trust import uncertainty as U
from src.trust import sustainability as S
from src.trust.trust_score import trust_score
from experiments.methods import apply_method

TAB = "manuscript/tables"
os.makedirs(TAB, exist_ok=True)

METHODS = ["fedavg-gat", "dp-fedavg", "fedfairgnn-nodp", "fedfairgnn"]
PRETTY = {"fedavg-gat": "FedAvg-GAT", "dp-fedavg": "DP-FedAvg",
          "fedfairgnn-nodp": "FedFairGNN (no DP)", "fedfairgnn": "\\textbf{FedFairGNN}"}


def run(method, rounds=40):
    cfg = ExperimentConfig(dataset="bail", num_clients=5, rounds=rounds,
                           local_epochs=2, hidden_channels=64, seed=0)
    apply_method(cfg, method)
    tr = FederatedTrainer(cfg)
    res = tr.run()
    load_flat_state(tr.ref_model, tr.global_flat.to(tr.device))
    # evaluate uncertainty on the first client's test split (representative)
    data = tr.clients_data[0]
    unc = U.uncertainty_report(tr.ref_model, data, "test", T=20)
    sus = S.sustainability_report(tr.ref_model, data, cfg,
                                  wall_seconds=res["final"].get("wall_s", 0.0))
    return res["final"], unc, sus, cfg


def main():
    rows = []
    for m in METHODS:
        final, unc, sus, cfg = run(m)
        eps = final.get("epsilon")
        t = trust_score(final, p=1.0, epsilon=eps, ece=unc["ece"])
        rows.append((m, final, unc, sus, eps, t))
        print(f"{m}: AUC={final['auc']:.3f} DPD={final['dpd']:.3f} ECE={unc['ece']:.3f} "
              f"comm={sus['per_round_mb']}MB trust={t:.3f}")

    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Method & ECE & Brier & U-gap & Comm/rd (MB) & Energy (Wh) & Trust \\\\",
             "\\midrule"]
    for m, final, unc, sus, eps, t in rows:
        lines.append(f"{PRETTY.get(m,m)} & {unc['ece']:.3f} & {unc['brier']:.3f} & "
                     f"{unc['uncertainty_gap']:.3f} & {sus['per_round_mb']:.3f} & "
                     f"{sus.get('energy_wh',0):.2f} & {t:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, "efficiency.tex"), "w") as f:
        f.write("\n".join(lines))
    print("[table] efficiency.tex written")

    # model card for flagship run
    from src.trust.compliance import model_card
    flagship = [r for r in rows if r[0] == "fedfairgnn"][0]
    card = model_card(flagship[3] and ExperimentConfig(dataset="bail").to_dict() or {},
                      {"final": flagship[1]}, {"sensitive_name": "Race (WHITE)",
                       "positive_meaning": "recidivism within follow-up"})
    with open("manuscript/model_card.md", "w") as f:
        f.write(card)
    print("[doc] model_card.md written")


if __name__ == "__main__":
    main()
