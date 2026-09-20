"""Tokenised layout of (state, questions) for one parallel forward pass.

The sequence is a *tree*, not a line:

    state ─┬─ question 1 ─┬─ option a
           │              ├─ option b
           │              └─ option c
           ├─ question 2 (noul) ─ [end]
           └─ question 3 ─┬─ level 0
                          ├─ level 1
                          └─ level 2

Every segment attends causally to itself and fully to its ancestors, and to
nothing else. Position ids restart at `parent.start + parent.len` for every
child, so each branch looks to the backbone exactly like "state, then this
question, then this option" — as if it had been sent alone.

Consequences (all verified in tests/):
  * questions are evaluated in isolation: adding/removing/reordering questions
    does not change any other question's answer;
  * options / levels are judged independently: reordering them permutes the
    output distribution exactly;
  * the whole request is one forward pass; there is no decoding loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    rich_to_text,
    state_to_text,
)

# ---- prompt templates -----------------------------------------------------
# Plain text, no new vocabulary: a pretrained base model already has sensible
# priors for "Answer (yes/no):" so training starts from a zero-shot classifier
# rather than from a random head.

STATE_TPL = "### State\n{state}\n\n"
QUESTION_TPL = "### Question\n{instructions}\n"
OPTION_TPL = "Candidate answer: {name}{desc}\nIs this candidate the correct answer? Answer (yes/no):"
LEVEL_TPL = "Candidate level: {desc}\nDoes the state match this level? Answer (yes/no):"
NOUL_TPL = "### Statement\n{instructions}\n{criteria}Is this statement true of the state? Answer (yes/no):"


@dataclass
class QuestionHeads:
    qid: str
    qtype: str  # choice | score | noul
    positions: List[int]  # token index whose hidden state feeds the yes/no logit
    names: List[str] = field(default_factory=list)  # option names (choice) / level ids (score)
    legend: Dict[str, Any] = field(default_factory=dict)  # score only


@dataclass
class Layout:
    input_ids: torch.Tensor  # [L]
    position_ids: torch.Tensor  # [L]
    attn_mask: torch.Tensor  # [L, L] bool, True = may attend
    heads: List[QuestionHeads]

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[0])


class _Seg:
    __slots__ = ("ids", "parent", "start_pos")

    def __init__(self, ids: List[int], parent: Optional[int]):
        self.ids = ids
        self.parent = parent
        self.start_pos = 0


def _tok(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_layout(
    tokenizer,
    state,
    questions: Dict[str, Question],
    max_state_tokens: Optional[int] = None,
) -> Layout:
    segs: List[_Seg] = []
    heads: List[QuestionHeads] = []

    def add(ids: List[int], parent: Optional[int]) -> int:
        segs.append(_Seg(ids, parent))
        return len(segs) - 1

    state_ids = _tok(tokenizer, STATE_TPL.format(state=state_to_text(state)))
    if max_state_tokens is not None and len(state_ids) > max_state_tokens:
        state_ids = state_ids[:max_state_tokens]
    root = add(state_ids, None)

    for qid, q in questions.items():
        if isinstance(q, (ChoiceQuestion, ScoreQuestion)):
            q_seg = add(_tok(tokenizer, QUESTION_TPL.format(instructions=rich_to_text(q.instructions))), root)
            qh = QuestionHeads(qid=qid, qtype=q.type, positions=[])
            if isinstance(q, ChoiceQuestion):
                for name, desc in q.criteria.items():
                    d = rich_to_text(desc)
                    text = OPTION_TPL.format(name=name, desc=(f": {d}" if d else ""))
                    add(_tok(tokenizer, text), q_seg)
                    qh.positions.append(len(segs) - 1)  # segment index for now; resolved below
                    qh.names.append(name)
            else:
                for i, desc in enumerate(q.criteria):
                    text = LEVEL_TPL.format(desc=rich_to_text(desc))
                    add(_tok(tokenizer, text), q_seg)
                    qh.positions.append(len(segs) - 1)
                    qh.names.append(str(i))
                    qh.legend[str(i)] = desc
            heads.append(qh)
        elif isinstance(q, NoulQuestion):
            crit = ""
            if q.criteria:
                crit = "".join(f"{k}: {rich_to_text(v)}\n" for k, v in q.criteria.items())
            text = NOUL_TPL.format(instructions=rich_to_text(q.instructions), criteria=crit)
            add(_tok(tokenizer, text), root)
            heads.append(QuestionHeads(qid=qid, qtype="noul", positions=[len(segs) - 1]))
        else:  # pragma: no cover
            raise TypeError(type(q))

    # flatten
    n_seg = len(segs)
    seg_start_tok = [0] * n_seg
    seg_len = [len(s.ids) for s in segs]
    tok = 0
    for i, s in enumerate(segs):
        seg_start_tok[i] = tok
        tok += len(s.ids)
        if s.parent is None:
            s.start_pos = 0
        else:
            p = segs[s.parent]
            s.start_pos = p.start_pos + len(p.ids)
    L = tok

    input_ids = torch.tensor([t for s in segs for t in s.ids], dtype=torch.long)
    position_ids = torch.tensor(
        [s.start_pos + k for s in segs for k in range(len(s.ids))], dtype=torch.long
    )
    seg_id = torch.tensor([i for i, s in enumerate(segs) for _ in s.ids], dtype=torch.long)

    # ancestor-or-self matrix A[s, t]
    A = torch.zeros(n_seg, n_seg, dtype=torch.bool)
    for i, s in enumerate(segs):
        j: Optional[int] = i
        while j is not None:
            A[i, j] = True
            j = segs[j].parent
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    attn = A[seg_id][:, seg_id] & causal

    # resolve head positions: last token of the segment
    for qh in heads:
        qh.positions = [seg_start_tok[s] + seg_len[s] - 1 for s in qh.positions]

    return Layout(input_ids=input_ids, position_ids=position_ids, attn_mask=attn, heads=heads)


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, L]
    position_ids: torch.Tensor  # [B, L]
    attn_mask: torch.Tensor  # [B, 1, L, L] bool
    layouts: List[Layout]

    pairs: Optional[List[tuple]] = None  # (i, j) paraphrase pairs, set by data.collate_train

    def to(self, device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            position_ids=self.position_ids.to(device),
            attn_mask=self.attn_mask.to(device),
            layouts=self.layouts,
            pairs=self.pairs,
        )


def collate(layouts: Sequence[Layout], pad_token_id: int) -> Batch:
    B = len(layouts)
    L = max(l.n_tokens for l in layouts)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    attn = torch.zeros((B, 1, L, L), dtype=torch.bool)
    for b, l in enumerate(layouts):
        n = l.n_tokens
        input_ids[b, :n] = l.input_ids
        position_ids[b, :n] = l.position_ids
        attn[b, 0, :n, :n] = l.attn_mask
        if n < L:  # pad rows attend to themselves only (avoids all-masked softmax rows)
            idx = torch.arange(n, L)
            attn[b, 0, idx, idx] = True
    return Batch(input_ids=input_ids, position_ids=position_ids, attn_mask=attn, layouts=list(layouts))
