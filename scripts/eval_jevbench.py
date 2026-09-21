"""Score a maatlm checkpoint on JevBench public sets (TypeSafe wire format).

    python scripts/eval_jevbench.py --model runs/maatlm-4b/final --data data/jevbench --bf16

JevBench rows are one decision each:
    {"id", "family", "state", "question": {type, instructions, criteria}, "labels": [...],
     "expected": <gold>, "provenance": {...}}

Reports argmax accuracy, Brier and ECE per split and per family, plus latency and
input tokens per decision (so $/1,000 decisions can be estimated at any tariff).

IMPORTANT: pass --max-state-tokens so every model under comparison gets the same state
budget. 33% of the hard tier is over 1024 tokens, so a checkpoint carrying a small
budget silently truncates those states and scores far below its real ability.

Use --serial for JevBench-comparable latency (they measure one decision at a time).

Datasets: https://github.com/fstandhartinger/jevbench (MIT).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch

from maatlm.metrics import ece as ece_fn
from maatlm.model import SystemOneModel


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def record(ans, row):
    """-> (prob_vector, target_vector, predicted_label, top_prob, expected_correctness)."""
    q, exp = row["question"], row["expected"]
    if q["type"] == "noul":
        p = ans.noul
        yes = str(exp).strip().lower() in ("yes", "true", "1")
        pred = "yes" if p >= 0.5 else "no"
        top = p if p >= 0.5 else 1 - p
        ok = float((p >= 0.5) == yes)
        return np.array([p, 1 - p]), np.array([float(yes), float(not yes)]), pred, top, ok
    if q["type"] == "choice":
        keys = list(ans.probabilities)
        p = np.array([ans.probabilities[k] for k in keys])
        t = np.array([1.0 if k == exp else 0.0 for k in keys])
        if t.sum() == 0:  # gold not among the declared options
            return None
        return p, t, ans.choice, float(p.max()), float(ans.choice == exp)
    keys = list(ans.probabilities)
    p = np.array([ans.probabilities[k] for k in keys])
    gold = str(exp)
    if gold not in keys:  # gold given as a description rather than a level index
        return None
    t = np.array([1.0 if k == gold else 0.0 for k in keys])
    pred = max(ans.probabilities, key=ans.probabilities.get)
    return p, t, pred, float(p.max()), float(pred == gold)


def summarize(recs):
    if not recs:
        return {}
    acc = float(np.mean([r["ok"] for r in recs]))
    brier = float(np.mean([((r["p"] - r["t"]) ** 2).sum() for r in recs]))
    e, _ = ece_fn([r["top"] for r in recs], [r["ok"] for r in recs])
    return {"n": len(recs), "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(e, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/jevbench")
    ap.add_argument("--splits", nargs="+", default=["easy", "original", "hard"])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--serial", action="store_true", help="batch=1; JevBench measures latency serially")
    ap.add_argument("--max-state-tokens", type=int, default=None,
                    help="pin the state budget for a symmetric comparison (recommended: 32768)")
    ap.add_argument("--price-per-mtok", type=float, default=None,
                    help="USD per million input tokens, to estimate $ per 1,000 decisions")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.serial:
        a.batch = 1

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SystemOneModel.from_pretrained(a.model, torch_dtype=torch.bfloat16 if a.bf16 else None, device=device)
    if a.max_state_tokens is not None:
        m.max_state_tokens = a.max_state_tokens
    m.eval()

    report, latencies, tokens = {"config": {"model": a.model, "state_budget": m.max_state_tokens,
                                            "batch": a.batch, "serial": bool(a.serial)}}, [], []
    print(json.dumps(report["config"]))
    for split in a.splits:
        path = os.path.join(a.data, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        rows = load(path)
        recs, skipped = [], 0
        for i in range(0, len(rows), a.batch):
            chunk = rows[i : i + a.batch]
            t0 = time.time()
            try:
                resp = m.predict_batch([(r["state"], {"q": r["question"]}) for r in chunk])
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {split} batch {i}: {type(exc).__name__}: {exc}")
                skipped += len(chunk)
                continue
            latencies.append((time.time() - t0) / len(chunk))
            tokens.extend(x.usage.input_tokens for x in resp)
            for r, resp_i in zip(chunk, resp):
                got = record(resp_i.answers["q"], r)
                if got is None:
                    skipped += 1
                    continue
                p, t, pred, top, ok = got
                recs.append({"p": p, "t": t, "top": top, "ok": ok, "family": r.get("family", "?"), "type": r["question"]["type"]})
        report[split] = summarize(recs)
        report[split]["skipped"] = skipped
        by_fam = collections.defaultdict(list)
        by_type = collections.defaultdict(list)
        for r in recs:
            by_fam[r["family"]].append(r)
            by_type[r["type"]].append(r)
        report[split]["by_family"] = {k: summarize(v) for k, v in sorted(by_fam.items())}
        report[split]["by_type"] = {k: summarize(v) for k, v in sorted(by_type.items())}
        print(f"{split:9s} " + json.dumps({k: v for k, v in report[split].items() if not isinstance(v, dict)}))

    if latencies:
        lat = sorted(latencies)
        p50, p95 = lat[len(lat) // 2], lat[min(int(len(lat) * 0.95), len(lat) - 1)]
        report["latency_ms"] = {
            "mean": round(1000 * sum(lat) / len(lat), 1),
            "p50": round(1000 * p50, 1),
            "p95": round(1000 * p95, 1),
            "note": f"per decision, batch {a.batch}, {device}" + ("" if a.serial else " (NOT serial; JevBench measures serially)"),
        }
        # JevBench speed axis: mean of score(p50), score(p95); score(s) = 100 - 20*log10(s/0.1)
        sc = lambda s_: max(0.0, min(100.0, 100 - 20 * math.log10(max(s_, 1e-6) / 0.1)))
        report["speed_axis_estimate"] = round((sc(p50) + sc(p95)) / 2, 1)
    if tokens:
        mean_tok = sum(tokens) / len(tokens)
        report["input_tokens"] = {"mean": round(mean_tok, 1), "total": sum(tokens), "n": len(tokens)}
        if a.price_per_mtok:
            usd_per_1k = mean_tok * 1000 * a.price_per_mtok / 1e6
            report["cost"] = {
                "usd_per_1k_decisions": round(usd_per_1k, 5),
                "price_per_mtok": a.price_per_mtok,
                # JevBench cost axis: 100 - 30*log10(usd / 0.001)
                "cost_axis_estimate": round(max(0.0, min(100.0, 100 - 30 * math.log10(max(usd_per_1k, 1e-9) / 0.001))), 1),
            }
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
