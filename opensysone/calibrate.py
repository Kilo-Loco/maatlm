"""Post-hoc temperature scaling, one temperature per primitive type.

    python -m opensysone.calibrate --model runs/x/final --data data/calib.jsonl [--out runs/x/calibrated]

Fits T_choice, T_score, T_noul by minimising NLL on a held-out set (never the
training set), writes them into the checkpoint's opensysone_config.json, and
prints before/after ECE. Temperature scaling cannot change the argmax, so
accuracy is untouched; it only fixes systematic over/under-confidence.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import read_jsonl
from .evaluate import apply_temps, collect
from .metrics import summarize
from .model import SystemOneModel


def fit_temperature(recs: List[Dict], qtype: str, iters: int = 300) -> float:
    rs = [r for r in recs if r["qtype"] == qtype]
    if not rs:
        return 1.0
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = log_t.exp()
        loss = 0.0
        for r in rs:
            if qtype == "noul":
                loss = loss + F.binary_cross_entropy_with_logits(r["logits"] / T, r["target"].float())
            else:
                loss = loss - (r["target"] * F.log_softmax(r["logits"] / T, dim=-1)).sum()
        loss = loss / len(rs)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", help="where to save the calibrated checkpoint (default: in place)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SystemOneModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.bf16 else None, device=device)
    recs = collect(m, read_jsonl(args.data), batch_size=args.batch, device=device)

    before = summarize(apply_temps(recs, {}))
    temps = {t: fit_temperature(recs, t) for t in ("choice", "score", "noul")}
    after = summarize(apply_temps(recs, temps))
    for t in temps:
        with torch.no_grad():
            m.log_temp[t].fill_(float(torch.tensor(temps[t]).log()))
    m.save(args.out or args.model)
    print(json.dumps({
        "temperatures": temps,
        "ece_before": {k: v["ece"] for k, v in before.items()},
        "ece_after": {k: v["ece"] for k, v in after.items()},
        "nll_before": {k: v["nll"] for k, v in before.items()},
        "nll_after": {k: v["nll"] for k, v in after.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
