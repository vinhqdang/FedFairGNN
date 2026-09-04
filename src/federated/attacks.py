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

Calibration of the two *stealth* attacks
----------------------------------------
``ipm`` and ``alie`` are not magnitude attacks: their whole point is that the
malicious update stays inside the benign population's spread so that a
distance/median screen cannot separate it. Driving them with the shared
``attack_intensity`` (10.0, sized for ``scaling``/``sign_flip``) destroys that
property and silently turns both into plain scaling attacks. They therefore
take their own parameters, ``ipm_epsilon`` and ``alie_z``; see the docstrings
of :func:`alie_z_from_counts` and the notes in :func:`poison_updates`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


def alie_z_from_counts(n: int, f: int) -> float:
    """ALIE deviation coefficient z, calibrated from the client counts.

    Baruch, Baruch & Goldberg, "A Little Is Enough: Circumventing Defenses For
    Distributed Learning" (NeurIPS 2019), Section 3. With ``n`` workers of which
    ``f`` are Byzantine, the attacker needs at least ``s`` benign workers to end
    up on its side of the coordinate-wise ordering for a median-style rule to be
    captured::

        s     = floor(n / 2 + 1) - f
        z_max = Phi^-1( (n - f - s) / (n - f) )

    where ``Phi^-1`` is the standard-normal quantile function. The poisoned
    update is then ``mean(benign) - z_max * std(benign)``: a perturbation
    measured in units of the benign population's own standard deviation, which
    is exactly what makes it survive range/median screening. A hardcoded z (the
    previous ``z = 1.5``) ignores both ``n`` and ``f`` and typically lands far
    outside the benign range, i.e. it is trivially detectable and is no longer
    the ALIE attack.

    Direction of the dependence: the numerator ``n - f - s`` simplifies to
    ``n - floor(n/2 + 1)``, which does not depend on ``f``, while the
    denominator ``n - f`` shrinks as ``f`` grows. The quantile argument, and
    hence ``z_max``, is therefore *non-decreasing* in ``f`` for fixed ``n`` --
    the more of the population the attacker controls, the further it may push
    the shared mean while still hiding inside the surviving benign spread. (For
    small ``f/n`` the quantile falls below 0.5 and ``z_max`` is negative, i.e.
    the safe perturbation points the other way; this is the paper's formula
    behaving as intended, not a sign bug.)

    Practical consequence worth knowing before reading a robustness table: the
    calibrated z is small, and sometimes zero or negative, for the small client
    populations used in cross-silo FL. At ``n = 10, f = 2`` (the repo's 20%
    Byzantine default) the closed form gives exactly ``z = 0``, i.e. the honest
    ALIE attacker may not deviate from the benign mean at all without becoming
    detectable -- calibrated ALIE is genuinely powerless at that operating
    point, which the old hardcoded ``z = 1.5`` concealed by launching a
    different, easily-screened attack instead. Set ``cfg.alie_z`` explicitly to
    study a deliberately over-aggressive attacker.

    Returns 0.0 for degenerate inputs (``n <= 0`` or ``f >= n``).
    """
    n = int(n)
    f = int(f)
    if n <= 0 or f < 0 or f >= n:
        return 0.0
    s = math.floor(n / 2 + 1) - f
    denom = n - f
    q = (n - f - s) / denom
    # Guard the open interval (0, 1): Phi^-1 is +-inf at the endpoints.
    q = min(max(q, 1e-6), 1.0 - 1e-6)
    try:
        from scipy.stats import norm
        return float(norm.ppf(q))
    except Exception:                                    # pragma: no cover
        # Acklam-style rational fallback via the inverse error function.
        return float(math.sqrt(2.0) * _erfinv(2.0 * q - 1.0))


def _erfinv(x: float) -> float:                          # pragma: no cover
    """Minimal inverse error function (only used if scipy is unavailable)."""
    a = 0.147
    ln = math.log(max(1.0 - x * x, 1e-300))
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(max(math.sqrt(t * t - ln / a) - t, 0.0)), x)


def poison_updates(attack: str, updates: List[torch.Tensor], metas: List[dict],
                   byzantine_ids: List[int], intensity: float = 10.0,
                   ipm_epsilon: float = 0.5, alie_z: Optional[float] = None,
                   ) -> Tuple[List[torch.Tensor], List[dict]]:
    """Craft the Byzantine clients' transmitted updates (omniscient attacker).

    ``intensity`` drives the *magnitude* attacks (gaussian / scaling /
    sign_flip / fairness_poison). The two stealth attacks are parameterised
    separately, because a large intensity is precisely what breaks them:

    ``ipm_epsilon``
        Xie, Koyejo & Gupta, "Fall of Empires: Breaking Byzantine-tolerant SGD
        by Inner Product Manipulation" (UAI 2019). The malicious update is
        ``-eps * mean(benign)``, which makes the aggregate point against the
        descent direction (negative inner product with the true gradient) while
        the attack REQUIRES ``eps < 1`` so the crafted vector stays inside the
        benign radius and survives screening. At the old ``eps = intensity =
        10`` the attack degenerated into ``scaling`` with a flipped sign and
        lost its defining stealth property. Default 0.5 mirrors
        ``ExperimentConfig.ipm_epsilon``.

    ``alie_z``
        ``None`` (default) calibrates z from ``(n, f)`` via
        :func:`alie_z_from_counts`, which is the actual attack. Pass a float to
        pin z for an ablation that sweeps it manually.
    """
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

    # ALIE's deviation is a function of the client population, both of which are
    # already known here: n = number of participating clients this round,
    # f = number of them the adversary controls.
    z_alie = alie_z if alie_z is not None else \
        alie_z_from_counts(len(updates), len(byzantine_ids))

    for i in byzantine_ids:
        g = updates[i].flatten()
        if attack == "gaussian":
            g = g + torch.randn_like(g) * intensity
        elif attack == "scaling":
            g = g * intensity
        elif attack == "sign_flip":
            g = -intensity * g
        elif attack == "ipm" and bmean is not None:
            # eps < 1 keeps ||g_byz|| = eps * ||mean(benign)|| inside the
            # benign radius -- the stealth condition of the IPM attack.
            g = -ipm_epsilon * bmean
        elif attack == "alie" and bmean is not None:
            g = bmean - z_alie * bstd
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
