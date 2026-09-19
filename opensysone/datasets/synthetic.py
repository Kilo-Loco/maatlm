"""Synthetic support-ticket generator with KNOWN ground-truth probabilities.

Used for the offline smoke test and as a calibration sanity check: because the
generator is stochastic given the visible text (e.g. a ticket that mentions a
double charge AND a wrong size is routed to billing 40% of the time), the ideal
model must output soft probabilities, and its ECE against the generator's true
probabilities is measurable.

    python -m opensysone.datasets.synthetic --out data/synth --n 4000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

from ..data import Example, write_jsonl

ISSUES = {
    "wrong_size": ["arrived in the wrong size", "don't fit at all, way too small", "came a full size too big"],
    "late": ["arrived two weeks late", "still hasn't shown up after 10 days", "tracking has said 'in transit' for a week"],
    "double_charge": ["I see two charges on my card", "was billed twice for one order", "my statement shows a duplicate charge"],
    "damaged": ["the box was crushed and one shoe is scuffed", "arrived with a torn sole", "came damaged"],
}
ISSUE_DEPT = {"wrong_size": "returns", "late": "shipping", "double_charge": "billing", "damaged": "returns"}
TONES = {
    "calm": ["Could you help me sort this out? Thanks.", "Just letting you know.", "No rush, but please advise."],
    "frustrated": ["This is the second time I've written in.", "Honestly, this is getting annoying.", "I expected better."],
    "angry": ["What are you going to do about this?!", "This is unacceptable, fix it NOW.", "I'm done with this store."],
}
WANTS = {
    "refund": ["I want my money back.", "Please refund me."],
    "exchange": ["Can I swap them for a size 10?", "I'd like an exchange."],
    "information": ["When will this be resolved?", "Can you tell me what happened?"],
    "none": [""],
}
PRODUCTS = ["running shoes", "boots", "sandals", "trainers", "sneakers"]

DEPT_Q = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    },
}
TONE_Q = {
    "type": "score",
    "instructions": "How frustrated is the customer?",
    "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threatening to leave"],
}
REFUND_Q = {"type": "noul", "instructions": "The customer explicitly asks for a refund."}
WANT_Q = {
    "type": "choice",
    "instructions": "What does the customer want to happen?",
    "criteria": {
        "refund": "Money back",
        "exchange": "Swap the item for a different one",
        "information": "Just an answer, no action needed",
        "none": "The customer does not say what they want",
    },
}


def gen_one(rng: random.Random) -> Example:
    n_issues = 1 if rng.random() < 0.65 else 2
    issues = rng.sample(list(ISSUES), n_issues)
    tone = rng.choices(list(TONES), weights=[0.5, 0.3, 0.2])[0]
    want = rng.choices(list(WANTS), weights=[0.3, 0.25, 0.15, 0.3])[0]
    product = rng.choice(PRODUCTS)

    sent = [f"My {product} " + rng.choice(ISSUES[issues[0]]) + "."]
    if n_issues == 2:
        sent.append("Also, " + rng.choice(ISSUES[issues[1]]) + ".")
    w = rng.choice(WANTS[want])
    if w:
        sent.append(w)
    sent.append(rng.choice(TONES[tone]))
    state = " ".join(sent)

    # --- ground-truth probabilities (what a perfectly calibrated model would say)
    depts = list(DEPT_Q["criteria"])
    if n_issues == 1:
        p_dept = [1.0 if d == ISSUE_DEPT[issues[0]] else 0.0 for d in depts]
    else:
        d0, d1 = ISSUE_DEPT[issues[0]], ISSUE_DEPT[issues[1]]
        if d0 == d1:
            p_dept = [1.0 if d == d0 else 0.0 for d in depts]
        else:  # first-mentioned issue wins 60/40 — deliberately ambiguous
            p_dept = [0.6 if d == d0 else 0.4 if d == d1 else 0.0 for d in depts]
    # tone: labellers confuse adjacent levels 15% of the time
    ti = list(TONES).index(tone)
    p_tone = [0.0, 0.0, 0.0]
    p_tone[ti] = 0.85
    if ti > 0:
        p_tone[ti - 1] += 0.15 if ti == 2 else 0.075
    if ti < 2:
        p_tone[ti + 1] += 0.15 if ti == 0 else 0.075
    p_refund = 0.97 if want == "refund" else 0.03
    wants = list(WANT_Q["criteria"])
    p_want = [0.9 if x == want else 0.1 / 3 for x in wants]

    return Example(
        state=state,
        questions={"department": DEPT_Q, "frustration": TONE_Q, "asks_refund": REFUND_Q, "wants": WANT_Q},
        targets={
            "department": {"probabilities": p_dept},
            "frustration": {"probabilities": p_tone},
            "asks_refund": {"p": p_refund},
            "wants": {"probabilities": p_want},
        },
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    exs = [gen_one(rng) for _ in range(args.n)]
    n_val = max(1, args.n // 10)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "train.jsonl"), exs[: -2 * n_val])
    write_jsonl(os.path.join(args.out, "calib.jsonl"), exs[-2 * n_val : -n_val])
    write_jsonl(os.path.join(args.out, "val.jsonl"), exs[-n_val:])
    print(f"wrote {len(exs)} examples to {args.out}/(train|calib|val).jsonl")


if __name__ == "__main__":
    main()
