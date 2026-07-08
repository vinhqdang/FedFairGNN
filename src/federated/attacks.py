"""Byzantine / adversarial client behaviours for the robustness study.

Two families:
  * data-level  -- ``label_flip`` corrupts a malicious client's local labels
    before training (handled in the client via a flag).
  * update-level -- the attacker crafts its transmitted pseudo-gradient (and,
    for ``fairness_poison``, its *reported* fairness/utility metrics) after
    seeing benign updates. Applied server-side over the collected updates,
    which is the standard omniscient-attacker model.

Supported update-level attacks
    gaussian         additive large-variance noise.
    scaling          model-replacement: scale the update up by ``intensity``.
    sign_flip        negate and scale (drives the model the wrong way).
    ipm              inner-product manipulation: -eps * mean(benign).
    alie             "A Little Is Enough": mean(benign) - z * std(benign).
    fairness_poison  send a biased update but *report* DPD~=0 & high utility to
                     capture BFWA's fairness-based weighting (attack on ours).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def poison_updates(attack: str, updates: List[torch.Tensor], metas: List[dict],
                   byzantine_ids: List[int], intensity: float = 10.0
                   ) -> Tuple[List[torch.Tensor], List[dict]]:
    if attack in ("none", None) or not byzantine_ids:
        return updates, metas

    updates = [u.clone() for u in updates]
    metas = [dict(m) for m in metas]
    benign = [i for i in range(len(updates)) if i not in byzantine_ids]
    if benign:
        bstack = torch.stack([updates[i].flatten() for i in benign])
        bmean = bstack.mean(0)
        bstd = bstack.std(0) + 1e-8
    else:
        bmean = bstd = None

    for i in byzantine_ids:
        g = updates[i].flatten()
        if attack == "gaussian":
            g = g + torch.randn_like(g) * intensity
        elif attack == "scaling":
            g = g * intensity
        elif attack == "sign_flip":
            g = -intensity * g
        elif attack == "ipm" and bmean is not None:
            g = -intensity * bmean
        elif attack == "alie" and bmean is not None:
            z = 1.5
            g = bmean - z * bstd
        elif attack == "fairness_poison":
            # The bias is already baked into the (honestly-shaped) update by the
            # attacker's local training that maximised the fairness gap (see
            # Client.train). Here the attacker only *lies* about its reported
            # fairness/utility so a fairness-aware server up-weights it.
            metas[i]["dpd"] = 0.0
            metas[i]["eod"] = 0.0
            metas[i]["perf"] = 0.99
        updates[i] = g.view_as(updates[i])
    return updates, metas


def flip_labels(y: torch.Tensor) -> torch.Tensor:
    """Binary label flip for data-poisoning (label_flip attack)."""
    return 1.0 - y
