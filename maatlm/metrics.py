"""Calibration & accuracy metrics for typed decisions."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """Expected calibration error on (top-1 prob, correctness) pairs, plus reliability bins.

    `correct` may be fractional: with soft targets it is the target probability of
    the predicted class (expected correctness), so a model that outputs the true
    0.85/0.15 split on an ambiguous item is scored as calibrated, not as
    under-confident."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if mask.sum() == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0})
            continue
        acc, conf = y[mask].mean(), c[mask].mean()
        total += mask.mean() * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()), "acc": float(acc), "conf": float(conf)})
    return float(total), bins


def brier_multi(p: np.ndarray, t: np.ndarray) -> float:
    return float(((p - t) ** 2).sum(-1).mean())


def nll_multi(p: np.ndarray, t: np.ndarray, eps: float = 1e-9) -> float:
    return float(-(t * np.log(p + eps)).sum(-1).mean())


def summarize(records: List[Dict]) -> Dict:
    """records: dicts with keys qtype, probs (list), target (list or float)."""
    out: Dict = {}
    for qt in ("choice", "score", "noul"):
        rs = [r for r in records if r["qtype"] == qt]
        if not rs:
            continue
        if qt == "noul":
            p = np.array([r["probs"] for r in rs], dtype=float)
            t = np.array([r["target"] for r in rs], dtype=float)
            pred = p >= 0.5
            conf = np.where(pred, p, 1 - p)
            # expected correctness under the (possibly soft) target: 0/1 for hard labels
            correct = np.where(pred, t, 1 - t)
            e, bins = ece(conf, correct)
            out[qt] = {
                "n": len(rs),
                "accuracy": float(correct.mean()),
                "brier": float(((p - t) ** 2).mean()),
                "nll": float(-(t * np.log(p + 1e-9) + (1 - t) * np.log(1 - p + 1e-9)).mean()),
                "ece": e,
                "bins": bins,
            }
            continue
        # variable n options -> per-record
        top_p, correct, brier, nll, mae = [], [], [], [], []
        for r in rs:
            p = np.asarray(r["probs"], dtype=float)
            t = np.asarray(r["target"], dtype=float)
            top_p.append(p.max())
            correct.append(float(t[p.argmax()]))  # expected correctness; 0/1 for hard labels
            brier.append(((p - t) ** 2).sum())
            nll.append(-(t * np.log(p + 1e-9)).sum())
            if qt == "score":
                idx = np.arange(len(p))
                mae.append(abs((idx * p).sum() - (idx * t).sum()))
        e, bins = ece(top_p, correct)
        out[qt] = {
            "n": len(rs),
            "accuracy": float(np.mean(correct)),
            "brier": float(np.mean(brier)),
            "nll": float(np.mean(nll)),
            "ece": e,
            "bins": bins,
        }
        if qt == "score":
            out[qt]["score_mae"] = float(np.mean(mae))
    return out
