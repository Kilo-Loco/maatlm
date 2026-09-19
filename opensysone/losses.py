"""Proper scoring rules — the objective that makes probabilities calibrated.

A scoring rule S(p, y) is *proper* if the expected score is minimised by
reporting the true probability. Training on one (rather than on a preference
reward) is what pushes the model towards "epistemically honest" probabilities:
being over- or under-confident is penalised in expectation, not rewarded.

  log   : cross-entropy / negative log-likelihood (strictly proper)
  brier : squared error between the distribution and the target (strictly proper)

For a one-step decision, "RL with a calibration reward" collapses to exactly
this: the policy is the distribution itself, so the expected reward under a
proper scoring rule is a differentiable function of the logits and can be
optimised directly. That is what `train.py` does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decision_loss(logits: torch.Tensor, target: torch.Tensor, qtype: str, rule: str = "log") -> torch.Tensor:
    """logits: [n] (choice/score) or [] (noul). target: matching shape of probabilities."""
    if qtype == "noul":
        p = torch.sigmoid(logits)
        if rule == "log":
            return F.binary_cross_entropy_with_logits(logits, target)
        return (p - target) ** 2 + ((1 - p) - (1 - target)) ** 2
    if rule == "log":
        return -(target * F.log_softmax(logits, dim=-1)).sum()
    p = F.softmax(logits, dim=-1)
    return ((p - target) ** 2).sum()


def batch_loss(raw, targets, layouts, rule: str = "log", type_weights=None):
    """raw: model output (per layout, per question). targets: same nesting. layouts: for qtypes."""
    total, n = 0.0, 0
    per_type = {"choice": [], "score": [], "noul": []}
    for per_q, per_t, lay in zip(raw, targets, layouts):
        for logits, t, qh in zip(per_q, per_t, lay.heads):
            qt = qh.qtype
            l = decision_loss(logits, t.to(logits.device), qt, rule)
            w = 1.0 if type_weights is None else type_weights.get(qt, 1.0)
            total = total + w * l
            n += 1
            per_type[qt].append(l.detach())
    return total / max(n, 1), {k: (torch.stack(v).mean().item() if v else None) for k, v in per_type.items()}
