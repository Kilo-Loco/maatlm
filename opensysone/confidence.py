"""Confidence: a scalar summary of how peaked a probability distribution is.

TypeSafe does not publish its formula and says you may substitute your own.
The default here, `margin`, is (p_top1 - p_top2) ** 0.7, which lands close to
the worked examples in their docs (e.g. {0.7, 0.3} -> 0.54, {0.92, 0.08} -> 0.88,
{0.37, 0.29, 0.24, 0.10} -> 0.16). `entropy` and `top1` are provided as
alternatives; pick by name in `SystemOneModel(confidence="...")`.
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
    return float((max(p) - 1.0 / n) / (1.0 - 1.0 / n))


CONFIDENCE_FNS = {"margin": margin, "entropy": entropy, "top1": top1}
