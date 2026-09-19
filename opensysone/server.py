"""HTTP API compatible with the TypeSafe request/response shape.

    OPENSYSONE_MODEL=runs/x/final uvicorn opensysone.server:app --host 0.0.0.0 --port 8000

    POST /v1/systemone
    {"state": "...", "questions": {"id": {"type": "choice", "instructions": "...", "criteria": {...}}}}

Requests arriving within a short window are micro-batched into one forward pass.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

import torch
from fastapi import FastAPI, HTTPException

from .model import SystemOneModel
from .schema import SystemOneRequest, SystemOneResponse

app = FastAPI(title="opensysone", version="0.1.0")
_model: SystemOneModel | None = None
_queue: "asyncio.Queue[Tuple[SystemOneRequest, asyncio.Future]]" = None  # type: ignore
BATCH_WINDOW_S = float(os.environ.get("OPENSYSONE_BATCH_WINDOW", "0.005"))
MAX_BATCH = int(os.environ.get("OPENSYSONE_MAX_BATCH", "32"))


def _load() -> SystemOneModel:
    path = os.environ.get("OPENSYSONE_MODEL")
    if not path:
        raise RuntimeError("set OPENSYSONE_MODEL to a checkpoint directory (or 'tiny' for the offline test model)")
    if path == "tiny":
        from .tiny import tiny_model
        return tiny_model()
    device = os.environ.get("OPENSYSONE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else None
    return SystemOneModel.from_pretrained(path, torch_dtype=dtype, device=device)


async def _worker():
    loop = asyncio.get_running_loop()
    while True:
        req, fut = await _queue.get()
        items = [(req, fut)]
        deadline = time.monotonic() + BATCH_WINDOW_S
        while len(items) < MAX_BATCH:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                items.append(await asyncio.wait_for(_queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        try:
            pairs = [(r.state, {k: v for k, v in r.questions.items()}) for r, _ in items]
            outs = await loop.run_in_executor(None, lambda: _model.predict_batch(pairs))
            for (r, f), o in zip(items, outs):
                o.model = r.model if r.model != "opensysone-latest" else _model.model_name
                if not f.done():
                    f.set_result(o)
        except Exception as e:  # noqa: BLE001
            for _, f in items:
                if not f.done():
                    f.set_exception(e)


@app.on_event("startup")
async def _startup():
    global _model, _queue
    _model = _load()
    _queue = asyncio.Queue()
    asyncio.create_task(_worker())


@app.get("/health")
async def health():
    return {"ok": True, "model": _model.model_name if _model else None}


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def systemone(req: SystemOneRequest):
    if _model is None:
        raise HTTPException(503, "model not loaded")
    fut = asyncio.get_running_loop().create_future()
    await _queue.put((req, fut))
    try:
        return await fut
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
