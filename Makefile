PYTHON ?= python3
.PHONY: test smoke doctor dry-run verify all

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) -m scripts.http_contract_smoke
	$(PYTHON) -m scripts.systemone_pipeline_smoke

verify:
	$(PYTHON) -m scripts.verify_dataset

doctor:
	$(PYTHON) -m scripts.check_environment

dry-run:
	$(PYTHON) benchmark.py --dry-run --limit 4

all: test smoke verify doctor dry-run

.PHONY: make-hard split probability-smoke verify-hard
make-hard:
	$(PYTHON) -m scripts.make_hard_dataset
split:
	$(PYTHON) -m scripts.make_calibration_split
probability-smoke:
	$(PYTHON) -m scripts.probability_smoke
verify-hard: verify
