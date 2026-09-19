# opensysone

An open-source **System One** style decision model, in the spirit of
[TypeSafe AI's Jev](https://typesafe.ai): send a *state* and a set of *typed
questions*, get back typed decisions with calibrated probabilities — in a single
parallel forward pass, with no text generation and therefore no possibility of a
type error.

TypeSafe has published *what* their model does (the API contract, the three
primitives, the calibration claims) but not *how* (architecture, sampler, RLCD).
This repo is an independent design that satisfies the same contract. It is not a
replication of their weights or their unpublished algorithm, and a 1–4B model
trained on public data will not match their reported intelligence.

```python
from opensysone.client import Client, Choice, Score, Noul

r = Client("http://localhost:8000").system_one(
    state="Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    questions={
        "department": Choice("Which team should handle this?", {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems"}),
        "frustration": Score("How frustrated is the customer?", [
            "Calm, just stating facts", "Frustrated but civil", "Very angry"]),
        "asks_refund": Noul("The customer explicitly asks for a refund."),
    })
r.answers["department"].choice, r.answers["department"].confidence, r.answers["department"].probabilities
# ('returns', 0.39, {'returns': 0.6, 'billing': 0.38, 'shipping': 0.02})
```

The HTTP shape (`POST /v1/systemone`, `answers[id].{choice|score|noul, probabilities, confidence}`,
`usage`) mirrors TypeSafe's docs, so code written against one runs against the other.

## How it works

**One forward pass, a tree instead of a line.** The request is tokenised as a
tree: the state is the root, each question is a child of the state, and each
option / level is a child of its question. A custom 4-D attention mask lets every
token attend causally to itself and fully to its ancestors — and to nothing
else. Position ids restart for every branch, so each option sees exactly what it
would see if it had been sent alone. All branches are computed at once in one
call to the backbone (`opensysone/layout.py`).

This gives, structurally and for any weights (see `tests/test_invariants.py`):

* **isolation** — adding, removing or reordering questions never changes another question's answer;
* **exact permutation equivariance of options** — no positional bias, no "first option wins";
* **constant-ish latency in the number of questions** — it is one batched forward pass;
* **type safety** — the only outputs are a softmax over the options you declared, or one sigmoid.

**Decision head reuses the LM head.** Each option ends in `Answer (yes/no):` and
its logit is `⟨h_end, w_yes − w_no⟩` from the pretrained language-model head.
At step zero the model is therefore already a zero-shot classifier; training
sharpens and calibrates it rather than learning a head from scratch.

**Primitives.**

| type   | model output                                  | answer fields                                    |
| ------ | --------------------------------------------- | ------------------------------------------------ |
| choice | softmax over ≤255 options                     | `choice`, `probabilities`, `confidence`          |
| score  | softmax over 2–10 ordered levels              | `score` (= Σ i·pᵢ), `probabilities`, `legend`, `confidence` |
| noul   | sigmoid                                       | `noul` ∈ [0,1]                                   |

**Training objective = proper scoring rule.** The model is trained to minimise
log loss (or Brier) against *probabilistic* targets. For a one-step decision,
"RL with a calibration reward" is exactly this: the policy *is* the output
distribution, so the expected reward under a proper scoring rule is a
differentiable function of the logits. Targets can be one-hot labels,
annotator-vote frequencies (SNLI ships these), outcome rates, or a frontier-LLM
ensemble's probabilities (`opensysone/distill.py` — the same reference TypeSafe
uses in their workflow evals).

**Calibration.** After training, one temperature per primitive is fitted on a
held-out split (`calibrate.py`); `evaluate.py` reports accuracy, NLL, Brier,
ECE and reliability bins. `confidence` is a pluggable statistic of the
distribution (`confidence.py`; default is a margin rule that matches TypeSafe's
published examples closely).

## Layout

```
opensysone/
  schema.py        request/response models, limits (255 options, 2–10 levels)
  layout.py        tree tokenisation, 4-D attention mask, restarted position ids
  model.py         SystemOneModel: backbone + yes/no decision head + temperatures, predict(), save/load
  confidence.py    confidence statistics
  losses.py        proper scoring rules (log, brier)
  data.py          JSONL format, dataset, option-shuffle augmentation
  train.py         trainer (full FT or LoRA, bf16, grad checkpointing, cosine LR)
  calibrate.py     temperature scaling on a held-out split
  evaluate.py      metrics + reliability bins
  metrics.py       ECE / Brier / NLL (soft-target aware)
  distill.py       teacher-ensemble labelling via OpenAI / Anthropic APIs
  server.py        FastAPI, micro-batching, TypeSafe-compatible endpoint
  client.py        tiny Python client
  tiny.py          offline random backbone for tests / smoke runs
  datasets/
    synthetic.py   generator with known ground-truth probabilities
    convert_hf.py  banking77, ag_news, trec, emotion, sst5, yelp, boolq, snli (soft), anli
scripts/
  setup_gpu.sh     one-time setup on a rented GPU
  train.sh         train -> calibrate -> evaluate recipe with sizing table
  bench.py         latency / decisions-per-second
tests/             structural invariants + server round-trip (run offline)
```

## Data format

One request per line; targets per question id:

```json
{"state": "...", 
 "questions": {"dept": {"type":"choice","instructions":"...","criteria":{"a":"...","b":"..."}},
               "sev":  {"type":"score","instructions":"...","criteria":["low","mid","high"]},
               "ref":  {"type":"noul","instructions":"..."}},
 "targets":   {"dept": {"probabilities":[0.6,0.4]}, "sev": {"label":1}, "ref": {"p":0.95}}}
```

## Running it on a rented GPU (Vast.ai / RunPod)

1. Rent a box with a PyTorch CUDA image. For a first run a 24 GB card (3090/4090)
   is enough for `Qwen3-1.7B-Base` with LoRA; see the sizing table in `scripts/train.sh`.
2. Copy this repo over (`scp -r opensysone root@host:` or push it to git and clone).
3. ```bash
   bash scripts/setup_gpu.sh        # installs, runs tests, downloads + converts public data
   bash scripts/train.sh            # train -> calibrate -> evaluate   (1.7B LoRA: ~1–2 h on a 4090)
   OPENSYSONE_MODEL=runs/qwen3-1.7b-base-sysone/final uvicorn opensysone.server:app --host 0.0.0.0 --port 8000
   python scripts/bench.py --model runs/qwen3-1.7b-base-sysone/final
   ```
4. Optional, and where most of the quality comes from: label *your own* states with
   a teacher ensemble and train on those.
   ```bash
   export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
   python -m opensysone.distill --in my_unlabeled.jsonl --out data/mine/train.jsonl \
       --teacher openai:<model> --teacher anthropic:<model> --samples 2
   DATA=data/mine bash scripts/train.sh
   ```
   `my_unlabeled.jsonl` only needs `state` and `questions`.

Everything except the GPU steps runs offline: `python -m pytest tests` and the
synthetic smoke run (`python -m opensysone.datasets.synthetic --out data/synth --n 3000`,
then `python -m opensysone.train --tiny ...`) work on a laptop CPU.

## Backbone choice

Any HF causal LM works (`--base`). Defaults are the Qwen3 *base* (not instruct)
checkpoints: base models haven't been through RLHF, so they have not had their
output distribution narrowed by preference optimisation — the "mode dropping"
TypeSafe argues against — which is the right starting point for a model whose
job is to report honest probabilities. Encoder models (ModernBERT etc.) would
also fit this layout with a small change to `model.py` (bidirectional within a
segment instead of causal), and are cheaper at inference; decoder-only was chosen
because the pretrained LM head gives a strong zero-shot start.

`attn_implementation` must be `sdpa` or `eager` (flash-attn cannot take an
arbitrary mask). SDPA with a dense mask is O(L²) memory in the sequence length;
with 8 questions × 20 options × ~25 tokens plus a 1k-token state that's ~5k
tokens — fine. For 255-option questions on long states, batch smaller.

## What this is and isn't

* It follows the published *approach*: typed decisions instead of strings,
  parallel evaluation of isolated questions, calibrated probabilities from a
  proper-scoring objective, confidence as a function of the distribution, and
  the same API contract.
* It is **not** TypeSafe's architecture, sampler or RLCD algorithm — those are
  unpublished. Where the docs were specific (options are judged independently,
  levels never see their neighbours, question ids are never shown to the model,
  score = probability-weighted level) the layout follows them exactly.
* Speed: no decoding means one forward pass per request, so latency is that of
  a single prefill. On a modern GPU a 1.7B model answers an 8-question request
  in tens of milliseconds and batches to thousands of decisions per second;
  measure with `scripts/bench.py` rather than trusting any number here.
* Quality is bounded by the backbone size and the targets. Hard public labels
  teach the task; *soft* targets (annotator votes, outcome rates, teacher
  ensembles) are what teach calibration. Budget for the distillation step if you
  care about the confidence numbers.

License: Apache-2.0.
