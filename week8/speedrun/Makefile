PY ?= python3
CONFIG ?= config-baseline.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.PHONY: prepare train eval sample reproduce
prepare:
	$(PY) prepare.py
train: prepare
	$(PY) train.py $(CONFIG)
	$(PY) train.py config-narrow.json --out artifacts/comparison/narrow
eval:
	$(PY) eval.py
sample:
	$(PY) sample.py
reproduce:
	$(PY) reproduce.py all
