"""Score a maatlm checkpoint on JevBench public sets (TypeSafe wire format).

    python scripts/eval_jevbench.py --model runs/maatlm-4b/final --data data/jevbench --bf16

JevBench rows are one decision each:
    {"id", "family", "state", "question": {type, instructions, criteria}, "labels": [...],
     "expected": <gold>, "provenance": {...}}

Reports argmax accuracy, Brier and ECE per split and per family, plus latency.
Datasets: https://github.com/fstandhartinger/jevbench (MIT).
"""

from __future__ import annotations

import argparse
import collections
import json
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
    ap.add_argument("--out")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SystemOneModel.from_pretrained(a.model, torch_dtype=torch.bfloat16 if a.bf16 else None, device=device)
    m.eval()

    report, latencies = {}, []
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
        report["latency_ms"] = {
            "mean": round(1000 * sum(lat) / len(lat), 1),
            "p50": round(1000 * lat[len(lat) // 2], 1),
            "p95": round(1000 * lat[int(len(lat) * 0.95)], 1),
            "note": f"per decision, batch {a.batch}, {device}",
        }
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
