"""Training data: JSONL, one request per line, with per-question targets.

    {
      "state": "...",                                   # str | object | array
      "questions": {"dept": {"type": "choice", ...}, "sev": {"type": "score", ...}, "ref": {"type": "noul", ...}},
      "targets": {
        "dept": {"probabilities": [0.6, 0.38, 0.02]},   # soft target, option order as in criteria
        "sev":  {"label": 1},                            # or a hard label (index)
        "ref":  {"p": 0.95}                              # noul: probability the statement is true
      }
    }

Soft targets are what RLCD-style training wants: a teacher ensemble's
probabilities, or empirical frequencies from multiple annotators/outcomes.
Hard labels are accepted and treated as one-hot (optionally smoothed).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .layout import Layout
from .schema import parse_question


@dataclass
class Example:
    state: Any
    questions: Dict[str, Any]
    targets: Dict[str, Any]


def read_jsonl(path: str) -> List[Example]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Example(d["state"], d["questions"], d["targets"]))
    return out


def write_jsonl(path: str, examples: Sequence[Example]) -> None:
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps({"state": e.state, "questions": e.questions, "targets": e.targets}, ensure_ascii=False) + "\n")


def target_tensor(q: Dict[str, Any], t: Dict[str, Any], smoothing: float = 0.0) -> torch.Tensor:
    """Return the training target for one question: [n] probs for choice/score, [] for noul."""
    qtype = q["type"]
    if qtype == "noul":
        return torch.tensor(float(t["p"]))
    n = len(q["criteria"])
    if "probabilities" in t:
        p = torch.tensor([float(x) for x in t["probabilities"]])
        assert p.numel() == n, f"target has {p.numel()} entries, question has {n}"
        p = p.clamp_min(0)
        p = p / p.sum()
    else:
        p = torch.zeros(n)
        p[int(t["label"])] = 1.0
    if smoothing > 0:
        p = (1 - smoothing) * p + smoothing / n
    return p


class SystemOneDataset(Dataset):
    """Builds layouts lazily; also applies (optional) option-shuffle augmentation.

    Shuffling is *not* needed for invariance (the mask guarantees it) but it
    keeps the tokenizer from ever seeing a fixed option order, which helps
    when a teacher's labels were themselves order-biased.
    """

    def __init__(self, model, examples: Sequence[Example], shuffle_options: bool = False, smoothing: float = 0.0, seed: int = 0):
        self.model = model
        self.examples = list(examples)
        self.shuffle_options = shuffle_options
        self.smoothing = smoothing
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Tuple[Layout, List[torch.Tensor]]:
        ex = self.examples[i]
        questions, targets = {}, []
        for qid, q in ex.questions.items():
            q = dict(q)
            t = ex.targets[qid]
            tt = target_tensor(q, t, self.smoothing)
            if self.shuffle_options and q["type"] == "choice":
                items = list(q["criteria"].items())
                perm = list(range(len(items)))
                self.rng.shuffle(perm)
                q["criteria"] = dict(items[j] for j in perm)
                tt = tt[perm]
            questions[qid] = q
            targets.append(tt)
        return self.model.layout(ex.state, questions), targets


def collate_train(model):
    def _fn(items):
        layouts = [it[0] for it in items]
        targets = [it[1] for it in items]
        return model.collate(layouts), targets
    return _fn
