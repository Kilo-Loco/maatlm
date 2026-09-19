"""Evaluate a checkpoint on a JSONL file: accuracy, NLL, Brier, ECE, reliability bins.

    python -m opensysone.evaluate --model runs/x/final --data data/val.jsonl [--out report.json]

Also usable as a library: `collect(model, examples)` returns raw logits + targets,
which `calibrate.py` reuses for temperature fitting.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SystemOneDataset, collate_train, read_jsonl
from .metrics import summarize
from .model import SystemOneModel


@torch.no_grad()
def collect(m: SystemOneModel, examples, batch_size: int = 8, device=None) -> List[Dict]:
    """Return per-question records: {qtype, logits (tensor), target (tensor)} — untempered."""
    m.eval()
    device = device or next(m.backbone.parameters()).device
    ds = SystemOneDataset(m, examples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_train(m))
    recs = []
    for batch, targets in dl:
        raw = m(batch.to(device))
        for per_q, per_t, lay in zip(raw, targets, batch.layouts):
            for logits, t, qh in zip(per_q, per_t, lay.heads):
                recs.append({"qtype": qh.qtype, "logits": logits.detach().float().cpu(), "target": t})
    return recs


def apply_temps(recs: List[Dict], temps: Dict[str, float]) -> List[Dict]:
    out = []
    for r in recs:
        T = temps.get(r["qtype"], 1.0)
        if r["qtype"] == "noul":
            p = torch.sigmoid(r["logits"] / T).item()
            out.append({"qtype": "noul", "probs": p, "target": float(r["target"])})
        else:
            p = F.softmax(r["logits"] / T, dim=-1).tolist()
            out.append({"qtype": r["qtype"], "probs": p, "target": r["target"].tolist()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SystemOneModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.bf16 else None, device=device)
    recs = collect(m, read_jsonl(args.data), batch_size=args.batch, device=device)
    temps = {t: float(m.temperature(t).detach()) for t in ("choice", "score", "noul")}
    report = summarize(apply_temps(recs, temps))
    report["temperatures"] = temps
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bins"} if isinstance(v, dict) else v for k, v in report.items()}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
