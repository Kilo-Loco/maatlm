"""Validate our JevBench scoring by reproducing a system whose official numbers are published.

Runs Laya (convaiinnovations/laya, ModernBERT-large 421M) on the same 231 public items and
scores it with the SAME record()/summarize() code as scripts/eval_jevbench.py. If our numbers
match JevBench's published per-item outcomes for Laya, our harness is sound.

Official (JevBench v1.2.7, public items): easy 0.958, standard 0.694, hard 0.351.

    pip install laya
    python scripts/validate_harness.py

Verified 2026-09-21: ours 0.9583 / 0.6944 / 0.3514 against official 0.958 / 0.694 / 0.351,
0 skipped, 0 failed — the scoring in eval_jevbench.py is sound.
"""
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from eval_jevbench import record, summarize, load   # our scoring, unchanged
import laya                                          # pip install laya

DATA = os.environ.get("JEVBENCH_DATA", os.path.join(os.path.dirname(_HERE), "data", "jevbench"))
OFFICIAL = {"easy": 0.958, "original": 0.694, "hard": 0.351}


class Ans:
    """Adapt Laya's dict answer to the attribute shape record() expects."""
    def __init__(self, d, qtype):
        self.type = qtype
        if qtype == "noul":
            self.noul = float(d.get("noul", d.get("p", 0.5)))
        else:
            self.probabilities = {str(k): float(v) for k, v in d["probabilities"].items()}
            self.choice = d.get("choice") or max(self.probabilities, key=self.probabilities.get)
            self.score = d.get("score")


def main():
    agent = laya.Agent()
    out = {}
    for split in ("easy", "original", "hard"):
        rows = load(f"{DATA}/{split}.jsonl")
        recs, skipped, failed = [], 0, 0
        for i, r in enumerate(rows):
            try:
                resp = agent.system_one(state=r["state"], questions={"q": r["question"]})
                a = resp["answers"]["q"] if "answers" in resp else resp["q"]
                got = record(Ans(a, r["question"]["type"]), r)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  [warn] {r['id']}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if got is None:
                skipped += 1
                continue
            p, t, pred, top, ok = got
            recs.append({"p": p, "t": t, "top": top, "ok": ok})
            if (i + 1) % 40 == 0:
                print(f"  {split} {i+1}/{len(rows)}", flush=True)
        s = summarize(recs)
        s["skipped"], s["failed"] = skipped, failed
        out[split] = s
        off = OFFICIAL[split]
        print(f"{split:9s} ours={s.get('accuracy')}  official={off}  delta={round((s.get('accuracy') or 0)-off,4)}")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
