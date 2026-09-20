"""Build a stratified training subset so a run fits a time budget.

    python scripts/subset_mix.py --public-cap 2000 --gen 8000 --real-repeat 3 --out data/mix/train.jsonl

Public rows are grouped by their (single) question id, i.e. by source dataset,
and capped per group; generator rows are sampled; the teacher-labelled real
rows are repeated. Full sets stay in data/public and data/gen untouched.
"""

from __future__ import annotations

import argparse
import collections
import json
import random


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-cap", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=8000)
    ap.add_argument("--real-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/mix/train.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pub, gen, real = load("data/public/train.jsonl"), load("data/gen/train.jsonl"), load("data/real/train.labeled.jsonl")
    by = collections.defaultdict(list)
    for r in pub:
        by[next(iter(r["questions"]))].append(r)
    sub = []
    for k, rows in sorted(by.items()):
        rng.shuffle(rows)
        sub += rows[: a.public_cap]
        print(f"{k:12s} {min(len(rows), a.public_cap)}")
    rng.shuffle(gen)
    sub += gen[: a.gen]
    print(f"generator    {min(len(gen), a.gen)}")
    sub += real * a.real_repeat
    print(f"real x{a.real_repeat}      {len(real) * a.real_repeat}")
    rng.shuffle(sub)
    with open(a.out, "w") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train rows:", len(sub))


if __name__ == "__main__":
    main()
