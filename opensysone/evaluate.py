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
                recs.append({"qtype": qh.qtype, "n": len(qh.positions), "logits": logits.detach().float().cpu(), "target": t})
    return recs


def temp_for(temps: Dict[str, float], qtype: str, n: int) -> float:
    """Finest available temperature: 'choice:3' beats 'choice' beats 1.0."""
    return temps.get(f"{qtype}:{n}", temps.get(qtype, 1.0))


def apply_temps(recs: List[Dict], temps: Dict[str, float]) -> List[Dict]:
    out = []
    for r in recs:
        T = temp_for(temps, r["qtype"], r.get("n", 1))
        if r["qtype"] == "noul":
            p = torch.sigmoid(r["logits"] / T).item()
            out.append({"qtype": "noul", "probs": p, "target": float(r["target"])})
        else:
            p = F.softmax(r["logits"] / T, dim=-1).tolist()
            out.append({"qtype": r["qtype"], "probs": p, "target": r["target"].tolist()})
    return out


@torch.no_grad()
def paraphrase_consistency(m: SystemOneModel, examples, batch_size: int = 8, device=None) -> Dict:
    """For examples carrying `paraphrases`: mean JS divergence and argmax agreement
    between the state's answers and each paraphrase's answers (tempered probabilities)."""
    from .losses import js_divergence

    pairs = [(e.state, alt, e.questions) for e in examples if e.paraphrases for alt in e.paraphrases]
    if not pairs:
        return {}
    m.eval()
    js, agree = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        reqs = [(s, q) for s, _, q in chunk] + [(a, q) for _, a, q in chunk]
        resp = m.predict_batch(reqs)
        for r1, r2 in zip(resp[: len(chunk)], resp[len(chunk) :]):
            for qid, a in r1.answers.items():
                b = r2.answers[qid]
                if a.type == "noul":
                    p, q = torch.tensor([a.noul, 1 - a.noul]), torch.tensor([b.noul, 1 - b.noul])
                    agree.append(float((a.noul >= 0.5) == (b.noul >= 0.5)))
                else:
                    keys = list(a.probabilities)
                    p, q = torch.tensor([a.probabilities[k] for k in keys]), torch.tensor([b.probabilities[k] for k in keys])
                    agree.append(float(max(a.probabilities, key=a.probabilities.get) == max(b.probabilities, key=b.probabilities.get)))
                js.append(float(js_divergence(p, q)))
    return {"pairs": len(js), "js_divergence": sum(js) / len(js), "argmax_agreement": sum(agree) / len(agree)}


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
    examples = read_jsonl(args.data)
    recs = collect(m, examples, batch_size=args.batch, device=device)
    temps = m.temperatures()
    report = summarize(apply_temps(recs, temps))
    report["temperatures"] = temps
    cons = paraphrase_consistency(m, examples, batch_size=args.batch, device=device)
    if cons:
        report["paraphrase_consistency"] = cons
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bins"} if isinstance(v, dict) else v for k, v in report.items()}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
