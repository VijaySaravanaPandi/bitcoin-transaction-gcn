# =============================================================================
# Vanilla GCN — Makefile
# =============================================================================
.DEFAULT_GOAL := help
PYTHON        := python
PIP           := pip
SRC           := src
TESTS         := tests
SCRIPTS       := scripts
NOTEBOOKS     := notebooks

.PHONY: help install install-dev test lint format train evaluate notebook clean

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:       ## Install runtime dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:   ## Install all dependencies (runtime + dev)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

test:          ## Run the test suite
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

test-cov:      ## Run tests with coverage report
	$(PYTHON) -m pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing

lint:          ## Lint source with ruff
	$(PYTHON) -m ruff check $(SRC) $(SCRIPTS) $(TESTS)

format:        ## Format source with black
	$(PYTHON) -m black $(SRC) $(SCRIPTS) $(TESTS)
	$(PYTHON) -m ruff check --fix $(SRC) $(SCRIPTS) $(TESTS)

train:         ## Run training with the default config
	$(PYTHON) $(SCRIPTS)/train.py --config configs/default.yaml

evaluate:      ## Run evaluation on the default checkpoint
	$(PYTHON) $(SCRIPTS)/evaluate.py --config configs/default.yaml

oversmoothing: ## Run the over-smoothing experiment
	$(PYTHON) $(SCRIPTS)/run_oversmoothing.py --config configs/default.yaml

notebook:      ## Start Jupyter Lab in the notebooks/ directory
	jupyter lab $(NOTEBOOKS)/

clean:         ## Remove generated artefacts (keep .gitkeep)
	find outputs -type f ! -name ".gitkeep" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
