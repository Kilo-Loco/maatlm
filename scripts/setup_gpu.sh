#!/usr/bin/env bash
# One-time setup on a fresh Vast.ai / RunPod box (PyTorch CUDA image recommended,
# e.g. "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" or RunPod's PyTorch template).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -U pip
python -m pip install -e ".[train,distill,dev]"
# flash-attn is NOT used: it cannot take the arbitrary tree mask. SDPA is the path.

python - <<'EOF'
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

# sanity: structural invariants on the tiny random model (no downloads)
python -m pytest tests -q

# pull the public data mix (needs network; ~10-20 min the first time)
python -m maatlm.datasets.convert_hf --out data/public --cap 20000 --val-cap 2000
python -m maatlm.datasets.generator  --out data/gen --n 20000 --seed 0
# merged train set: public breadth + generator truth (val/calib stay separate so ECE-vs-truth is reportable)
# the real teacher-labelled rows (data/real, tracked in git) are the only messy-input data: upweight x3
mkdir -p data/mix && { cat data/public/train.jsonl data/gen/train.jsonl; for i in 1 2 3; do cat data/real/train.labeled.jsonl; done; } \
  | shuf --random-source=<(yes) > data/mix/train.jsonl \
  && cp data/gen/calib.jsonl data/mix/calib.jsonl && cat data/public/val.jsonl data/gen/val.jsonl > data/mix/val.jsonl
wc -l data/mix/*.jsonl
