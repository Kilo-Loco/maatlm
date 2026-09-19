# opensysone — guide for Claude Code

Open-source "System One" decision model in the spirit of TypeSafe AI's Jev:
state + typed questions in → typed decisions with calibrated probabilities out,
in ONE parallel forward pass, no text generation. Read `README.md` first for the
design; this file is the working contract.

## Commands

```bash
pip install -e ".[train,distill,dev]"          # first time
python -m pytest tests -q                       # structural invariants + server round-trip; runs OFFLINE in ~5 s
python -m opensysone.datasets.synthetic --out data/synth --n 3000            # offline data with known true probs
python -m opensysone.train --tiny --train data/synth/train.jsonl --val data/synth/val.jsonl \
    --out runs/tiny --epochs 3 --batch 16 --lr 3e-3 --workers 0 --attn eager   # CPU smoke run (~6 min)
python -m opensysone.calibrate --model runs/tiny/final --data data/synth/calib.jsonl --out runs/tiny/calibrated
python -m opensysone.evaluate  --model runs/tiny/calibrated --data data/synth/val.jsonl
OPENSYSONE_MODEL=runs/tiny/calibrated uvicorn opensysone.server:app --port 8000
bash scripts/setup_gpu.sh && bash scripts/train.sh                          # real run on a GPU box (see sizing table in train.sh)
```

Expected smoke numbers after the CPU run above: choice Brier ≈ 6e-4 / ECE < 0.01,
noul Brier ≈ 1e-5. If a change makes these materially worse, it's a regression.

## Invariants — do not break

These are the whole point of the project and are enforced by `tests/test_invariants.py`:

1. **Isolation**: a question's answer never depends on which other questions are in the request.
2. **Option equivariance**: permuting a choice's options permutes `probabilities` exactly.
3. **One forward pass**: `SystemOneModel.forward` calls the backbone once per batch; never add a decode loop.
4. **Type safety**: outputs are only a softmax over declared options / levels or a single sigmoid. No strings.
5. **Question ids are never shown to the model** (only instructions + criteria are tokenised).
6. **Levels/options are judged independently** — an option's tokens never attend to sibling options.

Anything touching `layout.py` (mask, position ids, templates) or `model.py::forward`
must keep all tests green. Run them before and after.

## Architecture in one paragraph

`layout.py` tokenises the request as a tree (state → question → option), builds a
4-D boolean attention mask (attend to self causally + all ancestors) and restarts
position ids per branch. `model.py` runs the HF decoder stack once with that mask,
gathers the hidden state at each option's last token, and takes `⟨h, w_yes − w_no⟩`
from the LM head as the logit. `losses.py` trains on a proper scoring rule
(log / Brier) against soft targets. `calibrate.py` fits one temperature per
primitive on a held-out split. `server.py` exposes the TypeSafe-compatible
`POST /v1/systemone` with micro-batching.

## Conventions

- Python 3.10+, type hints, pydantic v2 for the API schema. No new frameworks.
- Keep the request/response shape byte-compatible with the TypeSafe docs
  (`choice`/`score`/`noul`, `probabilities`, `confidence`, `legend`, `usage`).
- Attention implementation must be `sdpa` or `eager`; flash-attn cannot take the tree mask.
- Temperatures (`log_temp`) are frozen during training and fitted only in `calibrate.py`, only on held-out data.
- Never calibrate or report metrics on the training split.
- `tiny.py` (random backbone, offline tokenizer) is for tests only; real runs start from a pretrained `--base`.
- Dataset ids in `datasets/convert_hf.py` are best-effort; if one 404s, fix the id, don't delete the converter.
- Data/runs are gitignored; checkpoints are HF `save_pretrained` dirs plus `opensysone_config.json`.

## Known gaps / good next tasks (roughly in priority order)

1. **Validate on real weights.** The pretrained path (`from_pretrained` + 4-D mask through a real Qwen3)
   has only been exercised structurally — HF was unreachable where this was built. First GPU
   task: `bash scripts/train.sh` with `BASE=Qwen/Qwen3-0.6B-Base LORA=0 EPOCHS=0.2` and confirm
   loss falls and `evaluate` ECE is sane. Fix anything that breaks in `model.py::hidden_states`.
2. **Zero-shot baseline**: run `evaluate.py` on the untrained base model to record the step-0
   numbers (the yes/no head should already beat chance).
3. **Memory for 255-option questions**: the dense `[B,1,L,L]` mask is O(L²). Options: chunk
   options across multiple forwards sharing a KV cache of the state, or a block-sparse mask via
   `torch.nn.attention.flex_attention` (mask_mod from `seg_id` + ancestor matrix).
4. **KV-cache the state**: many requests reuse the same state with different questions
   (and vice versa). Prefill the state once, run branches against the cache.
5. **Paraphrase-consistency term**: TypeSafe advertises "similar answers for similar inputs".
   Add an optional loss that penalises divergence between a state and its paraphrase
   (needs a paraphrase source — the distill script can generate them).
6. **Multi-annotator datasets** for calibration: ChaosNLI, GoEmotions (multi-label votes),
   Jigsaw toxicity rater counts. Add converters emitting `probabilities` targets.
7. **Encoder backbone option** (ModernBERT): bidirectional-within-segment mask variant in
   `layout.py`, new head in `model.py`. Cheaper inference.
8. **Server hardening**: request size limits, auth header, Prometheus metrics, ONNX/torch.compile export.
9. **Confidence formula**: `confidence.py::margin` approximates TypeSafe's published examples;
   revisit once real reliability data exists (e.g. fit confidence → P(argmax correct)).

## Don'ts

- Don't add a text-generation path "for explanations". Out of scope by design.
- Don't shuffle options at inference time — it's unnecessary (equivariance is structural) and costs latency.
- Don't fit temperatures inside `train.py`.
- Don't commit `data/` or `runs/`.
