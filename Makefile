.PHONY: install test smoke serve gpu
install:
	pip install -e ".[train,distill,dev]"
test:
	python -m pytest tests -q
smoke:
	python -m opensysone.datasets.synthetic --out data/synth --n 3000
	python -m opensysone.train --tiny --train data/synth/train.jsonl --val data/synth/val.jsonl --out runs/tiny --epochs 3 --batch 16 --lr 3e-3 --workers 0 --attn eager
	python -m opensysone.calibrate --model runs/tiny/final --data data/synth/calib.jsonl --out runs/tiny/calibrated
	python -m opensysone.evaluate --model runs/tiny/calibrated --data data/synth/val.jsonl
serve:
	OPENSYSONE_MODEL=$${MODEL:-runs/tiny/calibrated} uvicorn opensysone.server:app --host 0.0.0.0 --port 8000
gpu:
	bash scripts/setup_gpu.sh && bash scripts/train.sh
