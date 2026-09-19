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
python -m opensysone.datasets.convert_hf --out data/public --cap 20000 --val-cap 2000
