# maatlm — guide for Claude Code

Named for Ma'at, the feather the heart is weighed against. Package, env vars and
checkpoint config are `maatlm` / `MAATLM_*` / `maatlm_config.json`; the HTTP path
`/v1/systemone` is deliberately unchanged for TypeSafe wire compatibility.

Open-source "System One" decision model in the spirit of TypeSafe AI's Jev:
state + typed questions in → typed decisions with calibrated probabilities out,
in ONE parallel forward pass, no text generation. Read `README.md` first for the
design; this file is the working contract.

## Commands

```bash
pip install -e ".[train,distill,dev]"          # first time
python -m pytest tests -q                       # structural invariants + server round-trip; runs OFFLINE in ~5 s
python -m maatlm.datasets.synthetic --out data/synth --n 3000            # offline toy data with known true probs
python -m maatlm.datasets.generator --out data/gen --n 20000               # the real synthetic engine: 6 families, exact targets,
                                                                               #   contrastive twins, paraphrases, traps (see generator.py)
python -m maatlm.train --tiny --train data/synth/train.jsonl --val data/synth/val.jsonl \
    --out runs/tiny --epochs 3 --batch 16 --lr 3e-3 --workers 0 --attn eager   # CPU smoke run (~6 min)
python -m maatlm.calibrate --model runs/tiny/final --data data/synth/calib.jsonl --out runs/tiny/calibrated
python -m maatlm.evaluate  --model runs/tiny/calibrated --data data/synth/val.jsonl
MAATLM_MODEL=runs/tiny/calibrated uvicorn maatlm.server:app --port 8000
python -m maatlm.datasets.unlabeled --out data/real --n 1500 --eval 300   # real tickets + 10-question pack (needs HF)
python -m maatlm.distill ... --dry-run / --max-items N                     # teacher labels via any OpenAI-compatible URL
python -m maatlm.datasets.rewrite ... --dry-run / --max-items N            # cheap-LLM surface rewrites, labels untouched
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
7. **The backbone must honour an arbitrary attention mask.** Invariants 1, 2 and 6 are enforced by
   the 4-D tree mask, which layers with linear/recurrent/SSM attention silently ignore.
   `model.check_backbone_attention` refuses those at load. Measured 2026-09-21: Qwen3.5-4B-Base
   (24 linear + 8 full) drifts 0.193 on isolation and 0.205 on option permutation — **Qwen3.5 is
   disqualified at every size**. Full attention (Qwen3) and sliding-window attention (Gemma 4)
   both pass with drift 0.0000. Escape hatch for hybrid backbones: per-branch forward passes
   against a cached state prefix (no mask needed) — which is also what on-device needs.

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
  Keys are per primitive (`choice`) and per primitive+option count (`choice:3`); the finer key wins.
- Objective = proper scoring rule (log/Brier) + `--rps` ranked-probability term for ordinal `score`
  questions + `--consistency` JS term between a state and its `paraphrases`. All three are proper /
  minimised at the true distribution; RL is not needed for a one-step decision (see losses.py).
- `confidence` default is TypeSafe's published formula `(n·p_max − 1)/(n − 1)` (`confidence.top1`).
- The yes/no head reads `" Yes"`/`" No"` (falls back to lowercase if the tokenizer splits them).
- `max_state_tokens` defaults to 32768, matching Jev's state limit; the dense mask is O(L²) so batch smaller at that length.
- Never calibrate or report metrics on the training split.
- `tiny.py` (random backbone, offline tokenizer) is for tests only; real runs start from a pretrained `--base`.
- Base backbones beat instruct ones for the raw-text yes/no readout: instruct models want to emit
  " Answer" rather than " Yes". Zero-shot on 231 JevBench public items (easy/std/hard):
  Qwen3-4B-Base 0.938/0.528/0.469 · Qwen3.5-4B-Base 0.708/0.472/0.396 · gemma-4-E2B-it 0.333/0.222/0.351.
  Zero-shot screening therefore cannot rank instruct backbones fairly — they need training first.
- Dataset ids in `datasets/convert_hf.py` are best-effort; if one 404s, fix the id, don't delete the converter.
- Paid-API scripts (`distill.py`, `datasets/rewrite.py`) must stay credit-safe: dry-run estimate, `--max-items`,
  append-as-you-go output, resume on rerun. `data/real/eval.jsonl` is the fixed reference eval — never train on it,
  never label JevBench items.
- Data/runs are gitignored; checkpoints are HF `save_pretrained` dirs plus `maatlm_config.json`.

## Known gaps / good next tasks (roughly in priority order)

1. **Verify the calibration fix.** `setup_gpu.sh` used to fit temperatures on generator-only data,
   which the model had memorised, so hard-tier ECE went 0.132 (untrained) -> 0.289 (trained). The
   split is now public+generator and `calibrate.py --ood-data` reports whether a fit generalises,
   but this has NOT yet been re-measured on a trained checkpoint.
2. **Train Gemma 4 E2B** (~2.3B effective, Apache-2.0, MLX-supported, invariants verified) — the
   on-device candidate. Its zero-shot score is meaningless (instruct format), so only a trained
   run can rank it; system-one-open reaches 0.732 with this backbone.
3. **Memory for 255-option questions**: the dense `[B,1,L,L]` mask is O(L²). Options: chunk
   options across multiple forwards sharing a KV cache of the state, or a block-sparse mask via
   `torch.nn.attention.flex_attention` (mask_mod from `seg_id` + ancestor matrix).
4. **KV-cache the state**: many requests reuse the same state with different questions
   (and vice versa). Prefill the state once, run branches against the cache.
5. **LLM-written surface forms for the generator**: `datasets/generator.py` renders worlds with
   templates; an optional teacher rewrite of `state`/`paraphrases` (label untouched, since the
   generator owns it) would widen the surface distribution. The consistency loss already exists.
6. **Multi-annotator datasets** for calibration: ChaosNLI, GoEmotions (multi-label votes),
   Jigsaw toxicity rater counts. Add converters emitting `probabilities` targets.
7. **Encoder backbone option** (ModernBERT): bidirectional-within-segment mask variant in
   `layout.py`, new head in `model.py`. Cheaper inference.
8. **Server hardening**: request size limits, auth header, Prometheus metrics, ONNX/torch.compile export.
9. **JevBench**: `data/jevbench/*.jsonl` (public easy/original/hard) is in the TypeSafe wire format;
   step-0 Qwen3-0.6B scores 0.75 / 0.38 / 0.40 argmax accuracy. Add a converter + adapter so
   `evaluate.py` can report it directly; the hard tier was authored by Opus 5 and GPT-5.6, so
   distilling from those teachers inflates it — always report generator-truth ECE alongside.

## Don'ts

- Don't add a text-generation path "for explanations". Out of scope by design.
- Don't shuffle options at inference time — it's unnecessary (equivariance is structural) and costs latency.
- Don't fit temperatures inside `train.py`.
- Don't commit `data/` or `runs/`.
