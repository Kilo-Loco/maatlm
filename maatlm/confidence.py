"""Confidence: a scalar summary of how peaked a probability distribution is.

The default, `top1`, is the formula TypeSafe ships in their confidence docs:
(n * p_max - 1) / (n - 1), clamped to [0, 1] — 0 at uniform, 1 at a delta
(their quickstart's {0.85, 0.15, 0} -> 0.78 and {0.9, 0.06, 0.04} -> 0.85 match).
`margin` and `entropy` are alternatives; pick by name in `SystemOneModel(confidence="...")`.
"""

from __future__ import annotations

import math
from typing import Sequence


def margin(p: Sequence[float], gamma: float = 0.7) -> float:
    if len(p) == 1:
        return 1.0
    s = sorted(p, reverse=True)
    return float(max(0.0, s[0] - s[1]) ** gamma)


def entropy(p: Sequence[float]) -> float:
    """1 - H(p) / log(n): 1 for a delta, 0 for uniform."""
    n = len(p)
    if n == 1:
        return 1.0
    h = -sum(x * math.log(x) for x in p if x > 0)
    return float(max(0.0, 1.0 - h / math.log(n)))


def top1(p: Sequence[float]) -> float:
    """Rescaled max probability: 0 at uniform, 1 at a delta."""
    n = len(p)
    if n == 1:
        return 1.0
    return float(min(1.0, max(0.0, (max(p) - 1.0 / n) / (1.0 - 1.0 / n))))


CONFIDENCE_FNS = {"margin": margin, "entropy": entropy, "top1": top1}
