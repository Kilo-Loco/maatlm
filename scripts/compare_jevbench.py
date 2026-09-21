"""Rank our checkpoints against every published JevBench system on the IDENTICAL public items.

    python scripts/compare_jevbench.py runs/maatlm-4b/report_jevbench.json:maatlm-4b \
                                       runs/baseline_4b_jevbench.json:"Qwen3-4B untrained"

JevBench publishes per-item outcomes for the 231 redistributable public items
(results/v1.2/jevbench-v1.2-per-task.json). Their headline `by_tier` accuracies include
held-out items we cannot run, so they are NOT comparable to ours. Recomputing every system
on the public subset gives a like-for-like table.

Our scoring is validated: scripts/validate_harness.py reproduces Laya's official public-item
numbers to four decimals.

Caveat: this is the Intelligence axis only, on the public subset. It is NOT the JevBench
Score, which also weights a judge tier (no public items), calibration fidelity, speed and
cost measured under their protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

PER_TASK_URL = (
    "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/"
    "results/v1.2/jevbench-v1.2-per-task.json"
)
# our split filenames -> JevBench tier names
SPLIT_TIER = {"easy": "easy", "original": "standard", "hard": "hard"}


def fetch_per_task(cache: str) -> dict:
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        print(f"downloading {PER_TASK_URL}")
        urllib.request.urlretrieve(PER_TASK_URL, cache)
    with open(cache) as f:
        return json.load(f)


def published_rows(d: dict):
    tasks = {t["id"]: t for t in d["tasks"]}
    rows = []
    for _, s in d["systems"].items():
        per = {"easy": [], "standard": [], "hard": []}
        for tid, val in (s.get("public_tasks") or {}).items():
            code = val[0] if isinstance(val, (list, tuple)) else val
            t = tasks.get(tid)
            if t and t["tier"] in per:
                per[t["tier"]].append(1.0 if code == "c" else 0.0)
        if not any(per.values()):
            continue
        n = sum(len(v) for v in per.values())
        rows.append({
            "name": s["display"],
            "acc": {k: (sum(v) / len(v) if v else None) for k, v in per.items()},
            "n": {k: len(v) for k, v in per.items()},
            "all": sum(sum(v) for v in per.values()) / n,
            "ours": False,
        })
    return rows


def our_row(path: str, name: str):
    with open(path) as f:
        r = json.load(f)
    acc, n = {}, {}
    for split, tier in SPLIT_TIER.items():
        if split in r and r[split].get("accuracy") is not None:
            acc[tier] = r[split]["accuracy"]
            n[tier] = r[split]["n"]
    total = sum(n.values())
    return {
        "name": name,
        "acc": {t: acc.get(t) for t in ("easy", "standard", "hard")},
        "n": n,
        "all": sum(acc[t] * n[t] for t in acc) / total if total else 0.0,
        "ours": True,
        "budget": (r.get("config") or {}).get("state_budget"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*", help="path/to/report.json[:display name]")
    ap.add_argument("--cache", default="data/jevbench/per-task.json")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = published_rows(fetch_per_task(a.cache))
    for spec in a.reports:
        path, _, name = spec.partition(":")
        rows.append(our_row(path, name or os.path.basename(path)))

    ref = next((r for r in rows if r["n"]), None)
    print(f"\nJevBench v1.2 — {sum(ref['n'].values())} identical PUBLIC items "
          f"(easy {ref['n'].get('easy')} / standard {ref['n'].get('standard')} / hard {ref['n'].get('hard')})")
    print("Intelligence axis on the public subset only — not the JevBench Score.\n")
    print(f"{'#':>3}  {'system':<44}{'easy':>7}{'std':>7}{'hard':>7}{'all':>8}")
    print("-" * 78)
    fmt = lambda x: f"{x:.3f}" if x is not None else "  -  "
    for i, r in enumerate(sorted(rows, key=lambda r: -r["all"]), 1):
        star = "  <<<" if r["ours"] else ""
        print(f"{i:>3}  {r['name'][:43]:<44}{fmt(r['acc'].get('easy')):>7}"
              f"{fmt(r['acc'].get('standard')):>7}{fmt(r['acc'].get('hard')):>7}{r['all']:>8.3f}{star}")
    for r in rows:
        if r["ours"] and r.get("budget") and r["budget"] < 4096:
            print(f"\n!! {r['name']}: state budget {r['budget']} tokens — 33% of the hard tier is "
                  f"longer than that, so those states were truncated. Re-run with --max-state-tokens 32768.")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(sorted(rows, key=lambda r: -r["all"]), f, indent=2)


if __name__ == "__main__":
    main()
