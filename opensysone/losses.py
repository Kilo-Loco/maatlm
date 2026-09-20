"""Proper scoring rules — the objective that makes probabilities calibrated.

A scoring rule S(p, y) is *proper* if the expected score is minimised by
reporting the true probability. Training on one (rather than on a preference
reward) is what pushes the model towards "epistemically honest" probabilities:
being over- or under-confident is penalised in expectation, not rewarded.

  log   : cross-entropy / negative log-likelihood (strictly proper)
  brier : squared error between the distribution and the target (strictly proper)
  rps   : ranked probability score — squared error between the CDFs. Strictly
          proper for ORDINAL targets; used as an extra term for `score`
          questions so that being one level off is cheaper than being three
          levels off (log/Brier are blind to level order).

For a one-step decision, "RL with a calibration reward" collapses to exactly
this: the policy is the distribution itself, so the expected reward under a
proper scoring rule is a differentiable function of the logits and can be
optimised directly. That is what `train.py` does.

Optional consistency term: for a (state, paraphrase) pair with identical
questions, the Jensen–Shannon divergence between the two predicted
distributions is penalised. This is TypeSafe's "similar answers for similar
inputs" property; the pairs come from `datasets/generator.py` (or any data
with a `paraphrases` field).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def ranked_probability_score(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RPS = sum_k (P(k) - T(k))^2 over cumulative distributions; p, target: [n]."""
    return ((p.cumsum(-1) - target.cumsum(-1)) ** 2)[:-1].sum()


def decision_loss(
    logits: torch.Tensor, target: torch.Tensor, qtype: str, rule: str = "log", rps_weight: float = 0.0
) -> torch.Tensor:
    """logits: [n] (choice/score) or [] (noul). target: matching shape of probabilities."""
    if qtype == "noul":
        p = torch.sigmoid(logits)
        if rule == "log":
            return F.binary_cross_entropy_with_logits(logits, target)
        return (p - target) ** 2 + ((1 - p) - (1 - target)) ** 2
    if rule == "log":
        loss = -(target * F.log_softmax(logits, dim=-1)).sum()
    else:
        loss = ((F.softmax(logits, dim=-1) - target) ** 2).sum()
    if qtype == "score" and rps_weight > 0:
        loss = loss + rps_weight * ranked_probability_score(F.softmax(logits, dim=-1), target)
    return loss


def _dist(logits: torch.Tensor, qtype: str) -> torch.Tensor:
    if qtype == "noul":
        p = torch.sigmoid(logits)
        return torch.stack([p, 1 - p])
    return F.softmax(logits, dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * ((p * ((p + eps) / (m + eps)).log()).sum() + (q * ((q + eps) / (m + eps)).log()).sum())


def batch_loss(
    raw,
    targets,
    layouts,
    rule: str = "log",
    type_weights: Optional[Dict[str, float]] = None,
    rps_weight: float = 0.0,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    consistency_weight: float = 0.0,
):
    """raw: model output (per layout, per question). targets: same nesting. layouts: for qtypes.

    pairs: indices (i, j) into the batch whose layouts carry the same questions
    (state vs paraphrase); their distributions are pulled together with JS."""
    total, n = 0.0, 0
    per_type = {"choice": [], "score": [], "noul": [], "consistency": []}
    for per_q, per_t, lay in zip(raw, targets, layouts):
        for logits, t, qh in zip(per_q, per_t, lay.heads):
            qt = qh.qtype
            l = decision_loss(logits, t.to(logits.device), qt, rule, rps_weight)
            w = 1.0 if type_weights is None else type_weights.get(qt, 1.0)
            total = total + w * l
            n += 1
            per_type[qt].append(l.detach())
    if pairs and consistency_weight > 0:
        for i, j in pairs:
            for li, lj, qh in zip(raw[i], raw[j], layouts[i].heads):
                d = js_divergence(_dist(li, qh.qtype), _dist(lj, qh.qtype))
                total = total + consistency_weight * d
                per_type["consistency"].append(d.detach())
    return total / max(n, 1), {k: (torch.stack(v).mean().item() if v else None) for k, v in per_type.items()}
