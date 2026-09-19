"""SystemOneModel: a decoder-only backbone driven as a parallel, typed decision model.

Inference is exactly one forward pass:

  hidden = backbone(input_ids, position_ids, tree_attention_mask)
  for every option / level / statement:
      logit = <hidden[end_token], w_yes - w_no>        # reuse of the LM head
  choice / score : softmax over the option logits (per question)
  noul           : sigmoid of the single logit

No tokens are generated, so the output can only ever be a distribution over the
declared options (or a single probability) — that is the type-safety guarantee.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .confidence import CONFIDENCE_FNS
from .layout import Batch, Layout, build_layout, collate
from .schema import (
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    parse_question,
)

SYSONE_CONFIG = "opensysone_config.json"
TYPES = ("choice", "score", "noul")


class SystemOneModel(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer,
        yes_token: str = " yes",
        no_token: str = " no",
        confidence: str = "margin",
        max_state_tokens: Optional[int] = 4096,
        model_name: str = "opensysone-latest",
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.yes_id = tokenizer(yes_token, add_special_tokens=False)["input_ids"][0]
        self.no_id = tokenizer(no_token, add_special_tokens=False)["input_ids"][0]
        assert self.yes_id != self.no_id, "yes/no tokens collapse to the same id"
        self.confidence_name = confidence
        self.confidence_fn = CONFIDENCE_FNS[confidence]
        self.max_state_tokens = max_state_tokens
        self.model_name = model_name
        # per-type temperature (log-parameterised), fitted post-hoc by calibrate.py
        self.log_temp = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in TYPES})
        self.decision_head = self._DecisionHead(self)

    # ------------------------------------------------------------------ head
    class _DecisionHead(nn.Module):
        """Wraps lm_head rows for `yes` and `no`; tied to the backbone (trains with it)."""

        def __init__(self, outer: "SystemOneModel"):
            super().__init__()
            self._outer = [outer]  # avoid registering as a submodule twice

        def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [N, H] -> [N]
            outer = self._outer[0]
            lm_head = outer.backbone.get_output_embeddings()
            w = lm_head.weight[[outer.yes_id, outer.no_id]].to(h.dtype)  # [2, H]
            logits = h @ w.t()
            if getattr(lm_head, "bias", None) is not None:
                logits = logits + lm_head.bias[[outer.yes_id, outer.no_id]].to(h.dtype)
            return (logits[:, 0] - logits[:, 1]).float()

    # --------------------------------------------------------------- helpers
    def temperature(self, qtype: str) -> torch.Tensor:
        return self.log_temp[qtype].exp()

    def layout(self, state, questions: Dict[str, Any]) -> Layout:
        qs = {k: parse_question(v) for k, v in questions.items()}
        return build_layout(self.tokenizer, state, qs, max_state_tokens=self.max_state_tokens)

    def collate(self, layouts: Sequence[Layout]) -> Batch:
        return collate(layouts, pad_token_id=self.tokenizer.pad_token_id)

    def _additive_mask(self, bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        neg = torch.finfo(dtype).min
        return torch.zeros(bool_mask.shape, dtype=dtype, device=bool_mask.device).masked_fill(~bool_mask, neg)

    # --------------------------------------------------------------- forward
    def _inner(self) -> nn.Module:
        """The decoder stack without the LM head (works through a peft wrapper too)."""
        bb = self.backbone
        if hasattr(bb, "get_base_model"):
            bb = bb.get_base_model()
        return getattr(bb, bb.base_model_prefix)

    def hidden_states(self, batch: Batch) -> torch.Tensor:
        dtype = next(self.backbone.parameters()).dtype
        out = self._inner()(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            attention_mask=self._additive_mask(batch.attn_mask, dtype),
            use_cache=False,
        )
        return out.last_hidden_state  # [B, L, H]

    def forward(self, batch: Batch) -> List[List[torch.Tensor]]:
        """Raw (un-tempered) logits. Returns per layout, per question:
        choice/score -> tensor [n_options]; noul -> tensor [] (scalar)."""
        h = self.hidden_states(batch)
        # gather all head positions in one go
        b_idx, t_idx, spans = [], [], []
        for b, lay in enumerate(batch.layouts):
            for qh in lay.heads:
                spans.append((b, len(qh.positions)))
                b_idx.extend([b] * len(qh.positions))
                t_idx.extend(qh.positions)
        gathered = h[torch.tensor(b_idx, device=h.device), torch.tensor(t_idx, device=h.device)]  # [N, H]
        flat = self.decision_head(gathered)  # [N]
        out: List[List[torch.Tensor]] = [[] for _ in batch.layouts]
        k = 0
        for (b, n), qh in zip(spans, (qh for lay in batch.layouts for qh in lay.heads)):
            piece = flat[k : k + n]
            k += n
            out[b].append(piece[0] if qh.qtype == "noul" else piece)
        return out

    # ---------------------------------------------------------- probabilities
    def probabilities(self, logits: torch.Tensor, qtype: str) -> torch.Tensor:
        t = self.temperature(qtype)
        if qtype == "noul":
            return torch.sigmoid(logits / t)
        return F.softmax(logits / t, dim=-1)

    @torch.no_grad()
    def predict(self, state, questions: Dict[str, Any], model_name: Optional[str] = None) -> SystemOneResponse:
        return self.predict_batch([(state, questions)], model_name=model_name)[0]

    @torch.no_grad()
    def predict_batch(
        self, requests: Sequence[Tuple[Any, Dict[str, Any]]], model_name: Optional[str] = None
    ) -> List[SystemOneResponse]:
        self.eval()
        device = next(self.backbone.parameters()).device
        layouts = [self.layout(s, q) for s, q in requests]
        batch = self.collate(layouts).to(device)
        raw = self.forward(batch)
        responses = []
        for lay, per_q in zip(layouts, raw):
            answers: Dict[str, Any] = {}
            for qh, logits in zip(lay.heads, per_q):
                p = self.probabilities(logits, qh.qtype).cpu()
                if qh.qtype == "noul":
                    answers[qh.qid] = NoulAnswer(noul=round(float(p), 4))
                    continue
                probs = [round(float(x), 4) for x in p.tolist()]
                conf = round(self.confidence_fn(probs), 4)
                if qh.qtype == "choice":
                    best = qh.names[int(torch.argmax(p))]
                    answers[qh.qid] = ChoiceAnswer(
                        choice=best, confidence=conf, probabilities=dict(zip(qh.names, probs))
                    )
                else:
                    score = sum(i * pi for i, pi in enumerate(probs))
                    answers[qh.qid] = ScoreAnswer(
                        score=round(score, 4),
                        confidence=conf,
                        legend=qh.legend,
                        probabilities=dict(zip(qh.names, probs)),
                    )
            responses.append(
                SystemOneResponse(
                    model=model_name or self.model_name,
                    answers=answers,
                    usage=Usage(input_tokens=lay.n_tokens, output_tokens=0),
                )
            )
        return responses

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        cfg = {
            "yes_id": self.yes_id,
            "no_id": self.no_id,
            "confidence": self.confidence_name,
            "max_state_tokens": self.max_state_tokens,
            "model_name": self.model_name,
            "log_temp": {t: float(self.log_temp[t].detach()) for t in TYPES},
        }
        with open(os.path.join(path, SYSONE_CONFIG), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: str = "sdpa",
        device: Optional[str] = None,
        **kwargs,
    ) -> "SystemOneModel":
        backbone = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch_dtype, attn_implementation=attn_implementation
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        cfg_path = os.path.join(path, SYSONE_CONFIG)
        cfg: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        m = cls(
            backbone,
            tokenizer,
            confidence=cfg.get("confidence", kwargs.pop("confidence", "margin")),
            max_state_tokens=cfg.get("max_state_tokens", kwargs.pop("max_state_tokens", 4096)),
            model_name=cfg.get("model_name", kwargs.pop("model_name", "opensysone-latest")),
            **kwargs,
        )
        for t, v in cfg.get("log_temp", {}).items():
            with torch.no_grad():
                m.log_temp[t].fill_(v)
        if device:
            m.to(device)
        return m
