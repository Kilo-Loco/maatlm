"""Rewrite the SURFACE of generated examples with a cheap LLM; labels never move.

The generator (`generator.py`) owns the truth: every target is computed from a
world model. Its prose is templated, though, so a model trained on it alone
learns the templates. This pass asks a cheap OpenAI-compatible model (OpenRouter
works: --base-url https://openrouter.ai/api/v1) to re-express each `state` and
each entry of `paraphrases` as natural text — a ticket, a chat, a memo — while
keeping every fact, number, date, name and hedge word ("confirmed", "most
likely", "unclear", "doubtful") intact. `targets` and `questions` are copied
through untouched.

    OPENAI_API_KEY=$OPENROUTER_API_KEY python -m opensysone.datasets.rewrite \
        --in data/gen/train.jsonl --out data/gen/train.rewritten.jsonl \
        --base-url https://openrouter.ai/api/v1 --model anthropic/claude-haiku-4.5 \
        --max-items 4000 --dry-run          # estimate cost first, then drop --dry-run

Credit-safety: --max-items caps the spend, --dry-run prints a token estimate and
exits, output is appended per item so an interrupted run keeps what it paid for,
and re-running with the same --out resumes (already-rewritten ids are skipped).
A rewrite is rejected (original kept) if any number, date fragment or hedge word
present in the source is missing from the rewrite.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from ..schema import state_to_text

SYSTEM = (
    "You rewrite short business texts so they read like something a real person wrote (a support ticket, a chat "
    "message, an internal note, an email). Keep EVERY fact, number, amount, date, name, plan, feature and rule exactly; "
    "keep hedge words with the same strength (confirmed/verified stays certain; 'most likely'/'very probable' stays "
    "likely; 'unclear'/'may or may not' stays unsure; 'doubtful'/'probably not' stays unlikely). Do not add facts, "
    "do not resolve uncertainty, do not drop anything, do not answer any question. Return only the rewritten text."
)

HEDGES = ["confirmed", "verified", "most likely", "very probable", "as far as we can tell", "unclear", "may or may not", "could not confirm", "doubtful", "probably not", "unlikely"]
NUM_RE = re.compile(r"\d+")


def _key(state: Any) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _facts(text: str) -> Set[str]:
    t = text.lower()
    facts = set(NUM_RE.findall(t))
    facts |= {h for h in HEDGES if h in t}
    return facts


def _accept(src: str, out: str) -> bool:
    if not out or len(out) < 0.4 * len(src) or len(out) > 3 * len(src) + 200:
        return False
    return _facts(src) <= _facts(out)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def rewrite_one(client, model: str, text: str, temperature: float) -> Optional[str]:
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        temperature=temperature,
        max_tokens=min(2048, 2 * _estimate_tokens(text) + 100),
    )
    return (r.choices[0].message.content or "").strip(), getattr(r, "usage", None)


async def run(args):
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.max_items:
        rows = rows[: args.max_items]
    done: Set[str] = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                done.add(json.loads(l).get("meta", {}).get("rewrite_of", ""))
    todo = [r for r in rows if _key(r["state"]) not in done]

    texts = [state_to_text(r["state"]) for r in todo] + [state_to_text(p) for r in todo for p in (r.get("paraphrases") or [])]
    in_tok = sum(_estimate_tokens(t) + len(SYSTEM) // 4 for t in texts)
    out_tok = sum(_estimate_tokens(t) for t in texts)
    print(json.dumps({"items": len(todo), "already_done": len(done), "calls": len(texts), "est_input_tokens": in_tok, "est_output_tokens": out_tok}))
    if args.dry_run or not todo:
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url)
    sem = asyncio.Semaphore(args.concurrency)
    used = {"prompt": 0, "completion": 0, "accepted": 0, "rejected": 0}
    out_f = open(args.out, "a")

    async def one(text: str) -> str:
        async with sem:
            try:
                new, usage = await rewrite_one(client, args.model, text, args.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {e}")
                return text
        if usage:
            used["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            used["completion"] += getattr(usage, "completion_tokens", 0) or 0
        if _accept(text, new):
            used["accepted"] += 1
            return new
        used["rejected"] += 1
        return text

    async def item(r: Dict[str, Any]):
        src = state_to_text(r["state"])
        paras = [state_to_text(p) for p in (r.get("paraphrases") or [])]
        outs = await asyncio.gather(one(src), *(one(p) for p in paras))
        new = dict(r)
        new["state"] = outs[0]
        # the original surface becomes one more paraphrase: same world, same targets
        new["paraphrases"] = [p for p in list(outs[1:]) + [src] if p != outs[0]] or None
        if new["paraphrases"] is None:
            new.pop("paraphrases", None)
        meta = dict(r.get("meta") or {})
        meta["rewrite_of"] = _key(r["state"])
        meta["rewritten"] = outs[0] != src
        new["meta"] = meta
        out_f.write(json.dumps(new, ensure_ascii=False) + "\n")
        out_f.flush()

    for i in range(0, len(todo), args.concurrency * 4):
        await asyncio.gather(*(item(r) for r in todo[i : i + args.concurrency * 4]))
        print(f"{min(i + args.concurrency * 4, len(todo))}/{len(todo)} tokens={used}", flush=True)
    out_f.close()
    print(json.dumps({"done": len(todo), "usage": used}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--max-items", type=int, default=0, help="cap on rows (0 = all)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true", help="print the token estimate and exit")
    args = ap.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
