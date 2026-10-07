.PHONY: install test lint train eval serve

DATA ?= data/LA

install:
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest -q

lint:
	ruff check .

# Quick baseline: LFCC + GMM (the official ASVspoof-style baseline)
train:
	python -m vsd.train --data-root $(DATA) --features lfcc --backend gmm --out models/lfcc_gmm.joblib

eval:
	python -m vsd.evaluate --model models/lfcc_gmm.joblib --data-root $(DATA) --split eval --out-dir results/lfcc_gmm

serve:
	VSD_MODEL_PATH=models/lfcc_gmm.joblib uvicorn vsd.api:app --port 8000
