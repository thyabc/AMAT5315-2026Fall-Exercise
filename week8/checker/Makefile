PYTHON ?= python3
.PHONY: test
test:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) test_checker.py
