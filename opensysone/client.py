"""Minimal Python client mirroring the TypeSafe SDK surface.

    from opensysone.client import Client, Choice, Score, Noul
    c = Client("http://localhost:8000")
    r = c.system_one(state=ticket, questions={"dept": Choice("Which team?", {"billing": None, "returns": None})})
    r.answers["dept"].choice, r.answers["dept"].confidence
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import SystemOneResponse


@dataclass
class Choice:
    instructions: Any
    criteria: Dict[str, Any]
    type: str = "choice"


@dataclass
class Score:
    instructions: Any
    criteria: List[Any]
    type: str = "score"


@dataclass
class Noul:
    instructions: Any
    criteria: Optional[Dict[str, Any]] = None
    type: str = "noul"


class Client:
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "opensysone-latest", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def system_one(self, state: Any, questions: Dict[str, Any]) -> SystemOneResponse:
        qs = {k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in questions.items()}
        body = json.dumps({"state": state, "questions": qs, "model": self.model}).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return SystemOneResponse(**json.loads(resp.read()))
