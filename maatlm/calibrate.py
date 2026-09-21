"""Post-hoc temperature scaling, one temperature per primitive type AND per
(primitive, option count) where the calibration set has enough examples.

    python -m maatlm.calibrate --model runs/x/final --data data/calib.jsonl [--out runs/x/calibrated]

Fits T_choice, T_score, T_noul (plus T_choice:3, T_score:5, ... when >= --min-count
records exist for that option count; softmax sharpness depends on how many options
share the mass, so a single per-type temperature is systematically off for the
counts it was not fitted on) by minimising NLL on a held-out set (never the
training set), writes them into the checkpoint's maatlm_config.json, and
prints before/after ECE. Temperature scaling cannot change the argmax, so
accuracy is untouched; it only fixes systematic over/under-confidence.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import read_jsonl
from .evaluate import apply_temps, collect
from .metrics import summarize
from .model import SystemOneModel


def fit_temperature(recs: List[Dict], qtype: str, iters: int = 300, n: Optional[int] = None) -> float:
    rs = [r for r in recs if r["qtype"] == qtype and (n is None or r.get("n") == n)]
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
    ap.add_argument("--min-count", type=int, default=50, help="records needed to fit a per-option-count temperature")
    ap.add_argument("--ood-data", help="held-out set NOT used for fitting; reports whether the fit generalises")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SystemOneModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.bf16 else None, device=device)
    recs = collect(m, read_jsonl(args.data), batch_size=args.batch, device=device)

    before = summarize(apply_temps(recs, {}))
    temps = {t: fit_temperature(recs, t) for t in ("choice", "score", "noul")}
    counts: Dict[str, int] = {}
    for r in recs:
        if r["qtype"] != "noul":
            counts[f"{r['qtype']}:{r['n']}"] = counts.get(f"{r['qtype']}:{r['n']}", 0) + 1
    for key, c in sorted(counts.items()):
        if c >= args.min_count:
            qt, n = key.split(":")
            T = fit_temperature(recs, qt, n=int(n))
            if 0.05 < T < 20.0:  # a fit pinned at the clamp means the logits carry no signal; keep the type-level value
                temps[key] = T
    after = summarize(apply_temps(recs, temps))
    for key, T in temps.items():
        m.set_temperature(key, T)
    m.save(args.out or args.model)

    ood = {}
    if args.ood_data:
        ood_recs = collect(m, read_jsonl(args.ood_data), batch_size=args.batch, device=device)
        b, a2 = summarize(apply_temps(ood_recs, {})), summarize(apply_temps(ood_recs, temps))
        ood = {
            "file": args.ood_data,
            "n": {k: v["n"] for k, v in a2.items()},
            "ece_before": {k: round(v["ece"], 4) for k, v in b.items()},
            "ece_after": {k: round(v["ece"], 4) for k, v in a2.items()},
        }
        worse = [k for k in ood["ece_after"] if ood["ece_after"][k] > ood["ece_before"][k] + 0.01]
        if worse:
            print(f"!! calibration got WORSE out of distribution for {worse} — the fitting split is "
                  "probably too close to the training distribution")

    print(json.dumps({
        "temperatures": temps,
        "out_of_distribution": ood,
        "ece_before": {k: v["ece"] for k, v in before.items()},
        "ece_after": {k: v["ece"] for k, v in after.items()},
        "nll_before": {k: v["nll"] for k, v in before.items()},
        "nll_after": {k: v["nll"] for k, v in after.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
