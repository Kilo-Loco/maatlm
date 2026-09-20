"""Convert public Hugging Face datasets into System One JSONL.

    pip install datasets
    python -m maatlm.datasets.convert_hf --out data/public --cap 20000 \
        --sets banking77 ag_news trec emotion sst5 yelp boolq snli anli

Each converter maps a labelled dataset onto one primitive with *descriptive*
criteria (the model reads the descriptions, never a label id). Where a dataset
ships multiple annotator votes (SNLI validation/test) we emit SOFT targets —
those are the most valuable rows for calibration. Everything else is one-hot.

The mix is deliberately broad: intent routing, topic, question type, emotion,
sentiment on ordinal scales, yes/no reading comprehension, and entailment.
Add your own converter by registering a function in CONVERTERS.

Dataset ids are those on the Hub at time of writing; if one has moved, edit
the id in its converter.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Callable, Dict, Iterable, List

from ..data import Example, write_jsonl


def _choice(instructions: str, criteria: Dict[str, str]) -> Dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _score(instructions: str, levels: List[str]) -> Dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _noul(instructions: str) -> Dict:
    return {"type": "noul", "instructions": instructions}


# ---------------------------------------------------------------- converters

def banking77(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77", split=split)  # PolyAI/banking77 is a script dataset (unsupported now)
    names = ds.features["label"].names if hasattr(ds.features["label"], "names") else sorted({r["label_text"] for r in ds})
    crit = {n: n.replace("_", " ") for n in names}
    q = _choice("What is the customer's banking intent?", crit)
    for r in ds:
        idx = r["label"] if hasattr(ds.features["label"], "names") else names.index(r["label_text"])
        yield Example(r["text"], {"intent": q}, {"intent": {"label": idx}})


def ag_news(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split=split)
    crit = {"world": "World news, politics, international affairs", "sports": "Sports", "business": "Business, markets, companies, economy", "sci_tech": "Science and technology"}
    q = _choice("Which section does this news article belong to?", crit)
    for r in ds:
        yield Example(r["text"], {"section": q}, {"section": {"label": r["label"]}})


def trec(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("CogComp/trec", split=split)
    crit = {
        "abbreviation": "Asks about an abbreviation or its expansion",
        "entity": "Asks about an entity: animal, colour, event, product, substance, etc.",
        "description": "Asks for a description, definition, reason or manner",
        "human": "Asks about a person or group of people",
        "location": "Asks about a place",
        "numeric": "Asks for a number, date, count, distance, money, etc.",
    }
    q = _choice("What kind of answer is this question looking for?", crit)
    for r in ds:
        yield Example(r["text"], {"answer_type": q}, {"answer_type": {"label": r["coarse_label"]}})


def emotion(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split", split=split)
    crit = {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}
    q = _choice("What is the dominant emotion expressed by the writer?", crit)
    for r in ds:
        yield Example(r["text"], {"emotion": q}, {"emotion": {"label": r["label"]}})


def sst5(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5", split=split)
    q = _score("How positive is this movie review sentence?", ["Very negative", "Negative", "Neutral", "Positive", "Very positive"])
    for r in ds:
        yield Example(r["text"], {"sentiment": q}, {"sentiment": {"label": r["label"]}})


def yelp(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full", split=split)
    q = _score("How many stars did the reviewer most likely give?", ["1 star: terrible", "2 stars: poor", "3 stars: okay", "4 stars: good", "5 stars: excellent"])
    for r in ds:
        yield Example(r["text"], {"stars": q}, {"stars": {"label": r["label"]}})


def boolq(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split=split)
    for r in ds:
        q = _noul(f"According to the passage, the answer to the question '{r['question']}' is yes.")
        yield Example({"passage": r["passage"]}, {"answer": q}, {"answer": {"p": 1.0 if r["answer"] else 0.0}})


def snli(split: str) -> Iterable[Example]:
    """Entailment as a 3-way choice; validation/test carry 5 annotator votes -> soft targets."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli", split=split)
    crit = {"entailment": "The hypothesis must be true given the premise", "neutral": "The hypothesis may or may not be true", "contradiction": "The hypothesis cannot be true given the premise"}
    for r in ds:
        if r["label"] < 0:
            continue
        state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
        q = _choice("What is the relationship between the premise and the hypothesis?", crit)
        votes = r.get("annotator_labels") or r.get("labels")
        if votes and len(votes) > 1:
            counts = [0.0, 0.0, 0.0]
            for v in votes:
                if 0 <= v <= 2:
                    counts[v] += 1
            tgt = {"probabilities": [c / sum(counts) for c in counts]}
        else:
            tgt = {"label": r["label"]}
        yield Example(state, {"relation": q}, {"relation": tgt})


def anli(split: str) -> Iterable[Example]:
    from datasets import load_dataset
    for rnd in ("r1", "r2", "r3"):
        ds = load_dataset("facebook/anli", split=f"{split}_{rnd}")
        crit = {"entailment": "The hypothesis must be true given the premise", "neutral": "The hypothesis may or may not be true", "contradiction": "The hypothesis cannot be true given the premise"}
        q = _choice("What is the relationship between the premise and the hypothesis?", crit)
        for r in ds:
            yield Example({"premise": r["premise"], "hypothesis": r["hypothesis"]}, {"relation": q}, {"relation": {"label": r["label"]}})


CONVERTERS: Dict[str, Callable[[str], Iterable[Example]]] = {
    "banking77": banking77,
    "ag_news": ag_news,
    "trec": trec,
    "emotion": emotion,
    "sst5": sst5,
    "yelp": yelp,
    "boolq": boolq,
    "snli": snli,
    "anli": anli,
}
SPLITS = {
    "train": {"default": "train"},
    "val": {"default": "test", "banking77": "test", "boolq": "validation", "snli": "validation", "anli": "dev", "emotion": "validation", "sst5": "validation", "trec": "test", "ag_news": "test", "yelp": "test"},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=list(CONVERTERS))
    ap.add_argument("--cap", type=int, default=20000, help="max train rows per dataset")
    ap.add_argument("--val-cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train, val = [], []
    for name in args.sets:
        fn = CONVERTERS[name]
        for kind, cap, sink in (("train", args.cap, train), ("val", args.val_cap, val)):
            split = SPLITS[kind].get(name, SPLITS[kind]["default"])
            try:
                rows = list(fn(split))
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {name}/{split}: {e}")
                continue
            rng.shuffle(rows)
            rows = rows[:cap]
            sink.extend(rows)
            print(f"{name:10s} {kind}: {len(rows)} rows")
    rng.shuffle(train)
    rng.shuffle(val)
    # carve a calibration split off val (never calibrate on training data)
    half = len(val) // 2
    write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "calib.jsonl"), val[:half])
    write_jsonl(os.path.join(args.out, "val.jsonl"), val[half:])
    print(f"train={len(train)} calib={half} val={len(val) - half} -> {args.out}")


if __name__ == "__main__":
    main()
