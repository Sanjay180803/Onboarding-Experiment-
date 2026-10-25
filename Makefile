PY ?= python3

# single-threaded BLAS/OpenMP: results do not depend on machine core count
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.PHONY: all data design analyze validate report test clean

all: test data design analyze validate report

data:
	$(PY) sim/simulate.py --out data

design:
	$(PY) analysis/01_design.py

analyze:
	$(PY) analysis/02_analyze.py

validate:
	$(PY) analysis/03_validate.py

report:
	$(PY) analysis/04_report.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf data/*.parquet data/_truth reports/*.json reports/*.csv reports/*.parquet reports/figures/*.png
