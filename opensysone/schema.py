"""Request / response schema, mirroring the TypeSafe `POST /v1/systemone` shape.

Three primitives:
  - choice: pick one option from a named set (<= 255 options)
  - score : ordered levels (2..10), answer is the probability-weighted level index
  - noul  : probability that a statement is true

Type safety is structural: the model can only emit a distribution over the
options / levels you declared, or a single probability. There is no string
output to parse, so a type error is impossible by construction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# instructions / criteria values may be a string, an object or an array (like TypeSafe)
Rich = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def rich_to_text(value: Rich) -> str:
    """Render a string/object/array instruction or criterion as plain text the model reads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}: {rich_to_text(v)}")
        return "\n".join(parts)
    if isinstance(value, list):
        return "\n".join(f"- {rich_to_text(v)}" for v in value)
    return str(value)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Rich
    criteria: Dict[str, Rich]  # option name -> description (may be null)

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v: Dict[str, Rich]) -> Dict[str, Rich]:
        if not (1 <= len(v) <= MAX_CHOICE_OPTIONS):
            raise ValueError(f"choice needs 1..{MAX_CHOICE_OPTIONS} options, got {len(v)}")
        for k in v:
            if not isinstance(k, str) or not k.strip():
                raise ValueError("option names must be non-empty strings")
        return v

    @property
    def option_names(self) -> List[str]:
        return list(self.criteria.keys())


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: Rich
    criteria: List[Rich]  # ordered level descriptions, low -> high

    @field_validator("criteria")
    @classmethod
    def _check_levels(cls, v: List[Rich]) -> List[Rich]:
        if not (MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS):
            raise ValueError(
                f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels, got {len(v)}"
            )
        return v


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: Rich
    criteria: Optional[Dict[str, Rich]] = None  # optional {"true": ..., "false": ...}


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]

State = Union[str, Dict[str, Any], List[Any]]


def state_to_text(state: State) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


class SystemOneRequest(BaseModel):
    state: State
    questions: Dict[str, Question]
    model: str = "opensysone-latest"

    @model_validator(mode="after")
    def _non_empty(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: Dict[str, Rich]
    probabilities: Dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0  # there is no generation; kept for API compatibility


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Answer]
    usage: Usage


def parse_question(obj: Union[Question, Dict[str, Any]]) -> Question:
    if isinstance(obj, (ChoiceQuestion, ScoreQuestion, NoulQuestion)):
        return obj
    t = obj.get("type")
    if t == "choice":
        return ChoiceQuestion(**obj)
    if t == "score":
        return ScoreQuestion(**obj)
    if t == "noul":
        return NoulQuestion(**obj)
    raise ValueError(f"unknown question type: {t!r}")
