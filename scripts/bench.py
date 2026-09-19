"""Latency / throughput micro-benchmark for a checkpoint.

    python scripts/bench.py --model runs/x/final --questions 8 --options 20 --state-tokens 500 --batch 16
"""

from __future__ import annotations

import argparse
import time

import torch

from opensysone.model import SystemOneModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--options", type=int, default=20)
    ap.add_argument("--state-tokens", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SystemOneModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else None, device=device)
    state = " ".join(["The customer wrote a long message about a delayed order and a duplicate charge."] * (args.state_tokens // 16))
    qs = {
        f"q{i}": {"type": "choice", "instructions": f"Question number {i} about the state?", "criteria": {f"opt{j}": f"description of option {j}" for j in range(args.options)}}
        for i in range(args.questions)
    }
    lay = m.layout(state, qs)
    print(f"tokens per request: {lay.n_tokens}")
    for bs in (1, args.batch):
        reqs = [(state, qs)] * bs
        m.predict_batch(reqs)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(args.iters):
            m.predict_batch(reqs)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t) / args.iters
        print(f"batch {bs:3d}: {dt*1000:7.1f} ms/batch  {dt*1000/bs:7.1f} ms/request  {bs*args.questions/dt:8.0f} decisions/s")


if __name__ == "__main__":
    main()
