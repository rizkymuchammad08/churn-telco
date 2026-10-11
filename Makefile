PY ?= python

.PHONY: help install freeze all all-sample data features train monitoring ablation roi render dashboard lint test clean \
        data-sample features-sample train-sample ablation-sample \
        check-p1 check-p2 check-p3 check-p4 check-p5 check-p6

help:
	@echo "Targets: install freeze all all-sample data features train monitoring roi render lint test clean check-p1..check-p6"

install:
	$(PY) -m pip install -c constraints.txt -e ".[dev]"

# Writes constraints.txt from the current environment. Run it only in a clean virtualenv created
# for this project, after `make install` and `make all`; it pins everything that is installed.
freeze:
	$(PY) -m pip freeze --exclude-editable > constraints.txt

# Full pipeline on the full data (paths.raw). Entry point is locked.
all: data features train monitoring ablation roi render

# Fast path on the sample (paths.sample) for local iteration. Not a source of official report figures.
all-sample: data-sample features-sample train-sample monitoring ablation-sample roi render

data:
	$(PY) -m src.data.validate

features:
	$(PY) -m src.features.build

train:
	$(PY) -m src.models.train

# `train` writes reports/monitoring.json; this checks that it covers exactly the saved model's features
# (no feature without thresholds, none extra). Fails with E_MONITOR_* when a feature or a monitoring setting
# changed and `train` has not been run again.
monitoring:
	$(PY) -m src.models.monitoring --require-model

data-sample:
	$(PY) -m src.data.validate --sample

features-sample:
	$(PY) -m src.features.build --sample

train-sample:
	$(PY) -m src.models.train --sample

# Diagnostic: with vs without tenure and TotalCharges. It is a numeric source for the rendered documents
# (reports/ablation.json), so `make all` runs it before `render`; the official model is not affected.
ablation:
	$(PY) -m src.models.ablation

ablation-sample:
	$(PY) -m src.models.ablation --sample

roi:
	$(PY) -m src.reporting.roi

render:
	$(PY) -m src.reporting.render

# Needs the dashboard extra: pip install -e ".[dashboard]" (make install already includes it).
dashboard:
	$(PY) -m streamlit run src/reporting/dashboard.py

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# make check-pN = lint + pytest -m pN + phase-specific step + handoff manifest check.
define CHECK_COMMON
	$(MAKE) lint
	$(PY) -m pytest -m $(1)
endef

check-p1:
	$(call CHECK_COMMON,p1)
	$(MAKE) -n all > /dev/null
	$(PY) scripts/check_handoff.py handoff/p1.yaml

check-p2:
	$(call CHECK_COMMON,p2)
	$(MAKE) data
	$(PY) scripts/check_handoff.py handoff/p2.yaml

check-p3:
	$(call CHECK_COMMON,p3)
	$(MAKE) features
	$(PY) scripts/check_handoff.py handoff/p3.yaml

check-p4:
	$(call CHECK_COMMON,p4)
	$(PY) -m src.models.tuning --check
	$(MAKE) train
	$(MAKE) monitoring
	$(PY) scripts/check_handoff.py handoff/p4.yaml

check-p5:
	$(call CHECK_COMMON,p5)
	$(PY) scripts/check_handoff.py handoff/p5.yaml

check-p6:
	$(call CHECK_COMMON,p6)
	$(MAKE) render
	$(PY) scripts/scan_placeholders.py README.md CHANGELOG.md docs reports
	$(PY) scripts/check_handoff.py handoff/p6.yaml
