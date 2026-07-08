"""Experiment configuration and reproducibility utilities.

All experiments in this repository are driven by an :class:`ExperimentConfig`
dataclass so that every reported number can be traced back to an exact,
serialisable configuration. This is a deliberate design choice for a research
artifact: the manuscript's tables and figures are regenerated from logged runs,
not hand-authored.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Full specification of a federated experiment.

    Grouped by concern: data / partition / model / federated optimisation /
    fairness / privacy / robustness / trust. Defaults reproduce the primary
    configuration reported in the manuscript.
    """

    # ---- experiment identity ----
    exp_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "results"

    # ---- data ----
    dataset: str = "german"          # german | credit | bail | pokec_z | pokec_n | synthetic
    data_root: str = "data"
    label_number: int = 1000         # max labelled train nodes per NIFTY convention (small datasets ignore)

    # ---- federated partition ----
    num_clients: int = 5
    partition: str = "dirichlet"     # dirichlet | uniform | community
    dirichlet_alpha: float = 0.5     # smaller alpha => more non-IID
    partition_by: str = "label"      # label | sensitive  (what the Dirichlet skews)

    # ---- model ----
    model: str = "fedfairgnn"        # fedfairgnn | gcn | gat | fairgnn | ...
    hidden_channels: int = 64
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.3

    # ---- federated optimisation ----
    rounds: int = 100
    local_epochs: int = 3
    local_lr: float = 0.01
    weight_decay: float = 1e-5

    # ---- aggregation ----
    aggregator: str = "bfwa"         # fedavg | bfwa | krum | multikrum | median | trimmed_mean | robust_bfwa
    trimmed_beta: float = 0.1        # fraction trimmed each side for trimmed_mean
    krum_f: int = 1                  # assumed number of Byzantine clients for Krum/Multi-Krum

    # ---- fairness ----
    fairness_weight: float = 1.0     # lambda: weight of fairness loss
    fairness_budget: float = 0.05    # tau: max allowed global DPD in BFWA
    beta_init: float = 0.5           # FSER coefficient init
    fw_iterations: int = 20
    dual_step_size: float = 0.1

    # ---- privacy (DP) ----
    dp_enabled: bool = True
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_clip: float = 1.0
    dp_mode: str = "auto"            # auto | none | ftgd | gradient
    #   ftgd     : privatise fairness statistics (ours)
    #   gradient : standard full-gradient DP-SGD (DP-FedAvg baseline)

    # ---- generic fairness (for baselines that add a local DP penalty) ----
    local_fairness: bool = False     # add soft-DPD penalty for non-fedfairgnn models
    q_ffl: float = 2.0               # q for q-FedAvg client-fairness aggregation
    fairfed_beta: float = 1.0        # FairFed fairness-gap step

    # ---- robustness (attack simulation) ----
    attack: str = "none"             # none | label_flip | gaussian | scaling | fairness_poison
    num_byzantine: int = 0
    attack_intensity: float = 10.0

    # ---- uncertainty / trust ----
    mc_dropout_samples: int = 20     # forward passes for MC-dropout uncertainty
    eval_uncertainty: bool = False

    # ---- bookkeeping ----
    notes: str = ""

    def run_id(self) -> str:
        return f"{self.exp_name}__{self.model}__{self.dataset}__{self.aggregator}__seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def set_seed(seed: int) -> None:
    """Seed all RNGs for deterministic behaviour.

    Note: differential-privacy noise is *intentionally* stochastic; results are
    reported as mean +/- std over multiple seeds (see experiments/).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)  # some scatter ops lack deterministic kernels
