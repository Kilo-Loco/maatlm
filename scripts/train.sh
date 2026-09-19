#!/usr/bin/env bash
# Full recipe: train -> calibrate -> evaluate -> serve.
#
#   BASE=Qwen/Qwen3-1.7B-Base bash scripts/train.sh            # default, fits a 24GB card with LoRA
#   BASE=Qwen/Qwen3-4B-Base LORA=16 bash scripts/train.sh      # 4B on a 48-80GB card
#   BASE=Qwen/Qwen3-0.6B-Base LORA=0 bash scripts/train.sh     # full fine-tune of the small one
#
# Rough sizing (bf16, batch 8 x accum 4, state <= 1k tokens):
#   0.6B full FT  : ~14 GB      -> RTX 3090/4090
#   1.7B LoRA r16 : ~12 GB      -> RTX 3090/4090
#   1.7B full FT  : ~28 GB      -> A6000 / L40S / A100-40
#   4B   LoRA r16 : ~22 GB      -> A6000 / L40S / A100
#   4B   full FT  : ~64 GB      -> A100-80 / H100
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${BASE:-Qwen/Qwen3-1.7B-Base}
DATA=${DATA:-data/public}
OUT=${OUT:-runs/$(basename "$BASE" | tr '[:upper:]' '[:lower:]')-sysone}
LORA=${LORA:-16}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-4}
LR=${LR:-$([ "$LORA" = "0" ] && echo 1e-5 || echo 1e-4)}

python -m opensysone.train \
  --base "$BASE" --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --out "$OUT" \
  --epochs "$EPOCHS" --batch "$BATCH" --grad-accum "$ACCUM" --lr "$LR" \
  --lora "$LORA" --bf16 --grad-checkpoint --shuffle-options --rule log \
  --eval-every 250 --workers 4

# temperature scaling on the held-out calibration split (never on train)
python -m opensysone.calibrate --model "$OUT/final" --data "$DATA/calib.jsonl" --bf16

# report on the untouched validation split
python -m opensysone.evaluate --model "$OUT/final" --data "$DATA/val.jsonl" --bf16 --out "$OUT/report.json"

echo
echo "serve with:  OPENSYSONE_MODEL=$OUT/final uvicorn opensysone.server:app --host 0.0.0.0 --port 8000"
