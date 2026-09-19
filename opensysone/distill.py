"""Teacher distillation: label unlabeled (state, questions) requests with an
ensemble of frontier LLMs, producing soft probability targets.

    export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
    python -m opensysone.distill --in data/unlabeled.jsonl --out data/train_teacher.jsonl \
        --teacher openai:gpt-5.6 --teacher anthropic:claude-fable-5-1 --samples 2 --concurrency 8

Input lines need only {"state": ..., "questions": {...}}; existing "targets" are
overwritten. Each teacher is asked (with a strict JSON schema in the prompt) for
a probability distribution per question; samples and teachers are averaged.
Averaging several strong models is the same reference TypeSafe uses in their
published workflow evals — and it is a legitimate proxy for "ground truth
probabilities" when no outcome data exist. When you DO have outcomes (labels,
A/B results, human panels), prefer those: see datasets/convert_hf.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

from .data import Example, read_jsonl, write_jsonl
from .schema import rich_to_text, state_to_text

SYSTEM = (
    "You are a careful, well-calibrated judge. You will be given a STATE and several independent QUESTIONS. "
    "For each question return probabilities that reflect your honest uncertainty: if two options are both "
    "plausible, split the probability; if the state clearly determines the answer, be near-certain. "
    "Judge each question on its own. Return ONLY a JSON object."
)


def build_prompt(ex: Example) -> str:
    lines = [f"STATE:\n{state_to_text(ex.state)}\n", "QUESTIONS:"]
    schema = {}
    for qid, q in ex.questions.items():
        t = q["type"]
        lines.append(f"\n[{qid}] ({t}) {rich_to_text(q['instructions'])}")
        if t == "choice":
            for name, desc in q["criteria"].items():
                d = rich_to_text(desc)
                lines.append(f"  - {name}" + (f": {d}" if d else ""))
            schema[qid] = {"probabilities": {name: "float" for name in q["criteria"]}}
        elif t == "score":
            for i, desc in enumerate(q["criteria"]):
                lines.append(f"  - level {i}: {rich_to_text(desc)}")
            schema[qid] = {"probabilities": {str(i): "float" for i in range(len(q["criteria"]))}}
        else:
            if q.get("criteria"):
                for k, v in q["criteria"].items():
                    lines.append(f"  - {k}: {rich_to_text(v)}")
            schema[qid] = {"p_true": "float"}
    lines.append("\nReturn JSON exactly of the form (probabilities must sum to 1 per question):")
    lines.append(json.dumps(schema, indent=1))
    return "\n".join(lines)


def parse_teacher(ex: Example, text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for qid, q in ex.questions.items():
        a = obj.get(qid)
        if not isinstance(a, dict):
            return None
        if q["type"] == "noul":
            try:
                out[qid] = {"p": min(1.0, max(0.0, float(a["p_true"])))}
            except (KeyError, TypeError, ValueError):
                return None
        else:
            keys = list(q["criteria"].keys()) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            probs = a.get("probabilities", {})
            try:
                vec = [max(0.0, float(probs.get(k, 0.0))) for k in keys]
            except (TypeError, ValueError):
                return None
            s = sum(vec)
            if s <= 0:
                return None
            out[qid] = {"probabilities": [v / s for v in vec]}
    return out


async def call_openai(client, model: str, prompt: str, temperature: float) -> str:
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return r.choices[0].message.content or ""


async def call_anthropic(client, model: str, prompt: str, temperature: float) -> str:
    r = await client.messages.create(
        model=model, max_tokens=2048, system=SYSTEM, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(getattr(b, "text", "") for b in r.content)


def average(targets: List[Dict[str, Any]], ex: Example) -> Dict[str, Any]:
    out = {}
    for qid, q in ex.questions.items():
        if q["type"] == "noul":
            out[qid] = {"p": sum(t[qid]["p"] for t in targets) / len(targets)}
        else:
            n = len(q["criteria"])
            acc = [0.0] * n
            for t in targets:
                for i, v in enumerate(t[qid]["probabilities"]):
                    acc[i] += v
            out[qid] = {"probabilities": [v / len(targets) for v in acc]}
    return out


async def run(args):
    teachers = []
    for spec in args.teacher:
        provider, model = spec.split(":", 1)
        if provider == "openai":
            from openai import AsyncOpenAI
            teachers.append(("openai", model, AsyncOpenAI(base_url=args.openai_base_url)))
        elif provider == "anthropic":
            from anthropic import AsyncAnthropic
            teachers.append(("anthropic", model, AsyncAnthropic()))
        else:
            raise ValueError(f"unknown provider {provider}")
    examples = read_jsonl(args.inp) if _has_targets(args.inp) else _read_unlabeled(args.inp)
    sem = asyncio.Semaphore(args.concurrency)
    results: List[Optional[Example]] = [None] * len(examples)

    async def label(i: int, ex: Example):
        prompt = build_prompt(ex)
        got = []
        async with sem:
            for provider, model, client in teachers:
                for _ in range(args.samples):
                    try:
                        text = await (call_openai if provider == "openai" else call_anthropic)(client, model, prompt, args.temperature)
                    except Exception as e:  # noqa: BLE001
                        print(f"[warn] {provider}:{model} failed on {i}: {e}")
                        continue
                    t = parse_teacher(ex, text)
                    if t:
                        got.append(t)
        if got:
            results[i] = Example(ex.state, ex.questions, average(got, ex))
        if i % 50 == 0:
            print(f"labeled {i}/{len(examples)}", flush=True)

    await asyncio.gather(*(label(i, ex) for i, ex in enumerate(examples)))
    kept = [r for r in results if r is not None]
    write_jsonl(args.out, kept)
    print(f"wrote {len(kept)}/{len(examples)} to {args.out}")


def _has_targets(path: str) -> bool:
    with open(path) as f:
        first = f.readline()
    return '"targets"' in first


def _read_unlabeled(path: str) -> List[Example]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Example(d["state"], d["questions"], {}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", action="append", required=True, help="provider:model, repeatable")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    args = ap.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
