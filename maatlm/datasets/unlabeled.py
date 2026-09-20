"""Build UNLABELED (state, questions) requests from real support text, ready for `distill.py`.

Why: the generator gives exact targets on templated worlds; teacher labels are
worth paying for only on messy, real inputs. And every teacher call labels ALL
questions on a state, so each state carries a wide fan-out pack (10 questions
of all three types) — that is what makes a small credit budget go far.

Sources (public, no script datasets):
  Tobi-Bueck/customer-support-tickets   real multi-field tickets (subject, body, queue, priority, type); English rows only
  bitext/Bitext-customer-support-...     short customer utterances with an intent label

    python -m maatlm.datasets.unlabeled --out data/real --n 1500 --eval 300
    # -> data/real/eval.jsonl (fixed, seed-chosen; label with BOTH frontier teachers, never train on it)
    # -> data/real/train.jsonl (the rest)

Nothing here is labelled; the `queue`/`priority`/`type` columns are deliberately
NOT copied into targets (teacher probabilities are the target, the columns can be
used later as a sanity check).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, Iterable, List

QUEUES = {
    "billing": "Charges, invoices, payments, refunds of money, subscriptions",
    "technical": "Bugs, errors, crashes, integrations, login or setup problems",
    "shipping": "Delivery, tracking, lost or delayed packages",
    "returns": "Returns, exchanges, wrong or damaged items",
    "account": "Profile, credentials, closing or changing an account",
    "security": "Suspicious activity, breaches, 2FA, compromised credentials",
    "sales": "Pricing, quotes, upgrades, pre-sales questions",
    "product": "Feature requests, feedback, how-to questions about features",
    "other": "None of the above",
}

PACK: Dict[str, Any] = {
    "queue": {"type": "choice", "instructions": "Which team should handle this ticket? Pick the team for the main problem.", "criteria": QUEUES},
    "priority": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Low: no time pressure, informational", "Medium: needs attention this week", "High: blocks the customer, needs attention today", "Critical: outage, security incident or money at risk right now"],
    },
    "frustration": {"type": "score", "instructions": "How frustrated does the writer sound?", "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language or threats to leave"]},
    "is_incident": {"type": "choice", "instructions": "What kind of message is this?", "criteria": {"incident": "Something is broken or went wrong", "request": "Asks for something to be done or changed", "question": "Asks for information only", "feedback": "Opinion, praise or complaint without a request"}},
    "asks_refund": {"type": "noul", "instructions": "The writer explicitly asks for money back (refund, chargeback, credit)."},
    "wants_human": {"type": "noul", "instructions": "The writer asks to speak with a person or to escalate."},
    "security_risk": {"type": "noul", "instructions": "The message describes a possible security problem (compromised account, leaked data, suspicious access)."},
    "has_identifier": {"type": "noul", "instructions": "The message includes a concrete identifier such as an order number, account id, invoice number or ticket id."},
    "actionable_now": {"type": "noul", "instructions": "An agent could act on this immediately without asking the writer for more information."},
    "mentions_deadline": {"type": "noul", "instructions": "The writer mentions a deadline, a date by which something must happen, or says it is time-critical."},
}

EN_RE = re.compile(r"^[\x00-\x7F’“”–—…€£]+$")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def tickets(n: int) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train", streaming=True)
    seen = set()
    for r in ds:
        if (r.get("language") or "").lower() not in ("en", "english"):
            continue
        body = _clean(r.get("body"))
        if not body or len(body) < 60 or len(body) > 2500 or not EN_RE.match(body[:200]):
            continue
        key = body[:120]
        if key in seen:
            continue
        seen.add(key)
        yield {"state": {"subject": _clean(r.get("subject")), "message": body}, "meta": {"source": "Tobi-Bueck/customer-support-tickets", "hint_queue": r.get("queue"), "hint_priority": r.get("priority"), "hint_type": r.get("type")}}
        if len(seen) >= n:
            return


def bitext(n: int) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train", streaming=True)
    seen = set()
    for r in ds:
        text = _clean(r.get("instruction")).replace("{{Order Number}}", f"#{random.randint(10000, 99999)}").replace("{{", "").replace("}}", "")
        if len(text) < 25 or text in seen:
            continue
        seen.add(text)
        yield {"state": text, "meta": {"source": "bitext/customer-support", "hint_intent": r.get("intent"), "hint_category": r.get("category")}}
        if len(seen) >= n:
            return


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500, help="total states (2/3 tickets, 1/3 short utterances)")
    ap.add_argument("--eval", type=int, default=300, help="size of the fixed eval split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    random.seed(args.seed)
    rows: List[Dict[str, Any]] = []
    rows += list(tickets(args.n * 2 // 3))
    rows += list(bitext(args.n - len(rows)))
    for r in rows:
        r["questions"] = PACK
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    os.makedirs(args.out, exist_ok=True)
    ev, tr = rows[: args.eval], rows[args.eval :]
    for name, part in (("eval", ev), ("train", tr)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"total": len(rows), "eval": len(ev), "train": len(tr), "questions_per_state": len(PACK)}))


if __name__ == "__main__":
    main()
