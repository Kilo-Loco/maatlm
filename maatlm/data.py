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
    paraphrases: Optional[List[Any]] = None  # alternative states with identical answers (consistency term)
    meta: Optional[Dict[str, Any]] = None  # free-form provenance (family, group, ...); never shown to the model


def read_jsonl(path: str) -> List[Example]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Example(d["state"], d["questions"], d["targets"], d.get("paraphrases"), d.get("meta")))
    return out


def write_jsonl(path: str, examples: Sequence[Example]) -> None:
    with open(path, "w") as f:
        for e in examples:
            d: Dict[str, Any] = {"state": e.state, "questions": e.questions, "targets": e.targets}
            if e.paraphrases:
                d["paraphrases"] = e.paraphrases
            if e.meta:
                d["meta"] = e.meta
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def target_tensor(q: Dict[str, Any], t: Dict[str, Any], smoothing: float = 0.0) -> torch.Tensor:
    """Return the training target for one question: [n] probs for choice/score, [] for noul.

    `smoothing` is applied ONLY to hard labels, never to genuine soft targets. A one-hot
    label from a public dataset is one annotator's opinion, not P=1.0, so softening it is a
    better estimate of truth and keeps the objective proper. A teacher ensemble's 0.6/0.4 or
    the generator's exact 0.8 already IS the truth — smoothing those would corrupt the signal
    the whole calibration story rests on."""
    qtype = q["type"]
    if qtype == "noul":
        p = float(t["p"])
        hard = p in (0.0, 1.0)
        if smoothing > 0 and hard:
            p = (1 - smoothing) * p + smoothing / 2
        return torch.tensor(p)
    n = len(q["criteria"])
    if "probabilities" in t:
        p = torch.tensor([float(x) for x in t["probabilities"]])
        assert p.numel() == n, f"target has {p.numel()} entries, question has {n}"
        p = p.clamp_min(0)
        return p / p.sum()          # genuine soft target: leave it alone
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

    def __init__(
        self,
        model,
        examples: Sequence[Example],
        shuffle_options: bool = False,
        smoothing: float = 0.0,
        seed: int = 0,
        paraphrases: bool = False,
    ):
        self.model = model
        self.examples = list(examples)
        self.shuffle_options = shuffle_options
        self.smoothing = smoothing
        self.paraphrases = paraphrases
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> List[Tuple[Layout, List[torch.Tensor]]]:
        """Returns 1 item, or 2 (state + one sampled paraphrase, same shuffled questions)."""
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
        out = [(self.model.layout(ex.state, questions), targets)]
        if self.paraphrases and ex.paraphrases:
            alt = self.rng.choice(ex.paraphrases)
            out.append((self.model.layout(alt, questions), targets))
        return out


def collate_train(model):
    """Flattens the per-example item lists; records (i, j) index pairs of paraphrases."""

    def _fn(items):
        layouts, targets, pairs = [], [], []
        for group in items:
            if isinstance(group, tuple):  # backwards compat: a single (layout, targets)
                group = [group]
            base = len(layouts)
            for lay, tg in group:
                layouts.append(lay)
                targets.append(tg)
            if len(group) == 2:
                pairs.append((base, base + 1))
        batch = model.collate(layouts)
        batch.pairs = pairs
        return batch, targets

    return _fn
