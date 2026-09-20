import os

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


def test_server_roundtrip(monkeypatch):
    monkeypatch.setenv("MAATLM_MODEL", "tiny")
    from maatlm import server

    with TestClient(server.app) as c:
        r = c.post(
            "/v1/systemone",
            json={
                "state": {"ticket": "Two charges on my card and the shoes never came."},
                "questions": {
                    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}},
                    "urgent": {"type": "noul", "instructions": "This is urgent."},
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["answers"]) == {"dept", "urgent"}
        assert body["answers"]["dept"]["choice"] in ("billing", "shipping")
        assert 0 <= body["answers"]["urgent"]["noul"] <= 1
        # schema violations are rejected before they reach the model
        bad = c.post("/v1/systemone", json={"state": "x", "questions": {"s": {"type": "score", "instructions": "x", "criteria": ["one"]}}})
        assert bad.status_code == 422
