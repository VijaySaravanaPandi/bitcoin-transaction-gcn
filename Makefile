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
PYTHONPATH    := $(SRC)

.PHONY: help install install-dev \
        test test-numpy test-all test-cov \
        lint format \
        train train-numpy evaluate oversmoothing \
		analyze-risk evaluate-links \
        notebook colab clean

# ── Help ─────────────────────────────────────────────────────────────────────

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────────────────

install:       ## Install runtime dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:   ## Install all dependencies (runtime + dev)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

# ── Testing ──────────────────────────────────────────────────────────────────

test:          ## Run PyTorch GCN tests only (51 tests)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest $(TESTS) -v --tb=short \
	  --ignore=$(TESTS)/test_numpy_gcn.py

test-numpy:    ## Run pure NumPy GCN tests only (35 tests)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest $(TESTS)/test_numpy_gcn.py -v --tb=short

test-all:      ## Run ALL tests — PyTorch + NumPy (86 tests)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest $(TESTS) -v --tb=short

test-cov:      ## Run all tests with HTML coverage report
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest $(TESTS) \
	  --cov=$(SRC) --cov-report=term-missing --cov-report=html

# ── Code quality ─────────────────────────────────────────────────────────────

lint:          ## Lint source with ruff
	$(PYTHON) -m ruff check $(SRC) $(SCRIPTS) $(TESTS)

format:        ## Auto-format with black + ruff --fix
	$(PYTHON) -m black $(SRC) $(SCRIPTS) $(TESTS)
	$(PYTHON) -m ruff check --fix $(SRC) $(SCRIPTS) $(TESTS)

# ── Training & evaluation ────────────────────────────────────────────────────

train:         ## Train PyTorch GCN (default config)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/train.py \
	  --config configs/default.yaml

train-numpy:   ## Train pure NumPy GCN (synthetic data, no PyTorch)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/train_numpy.py

evaluate:      ## Evaluate PyTorch GCN from checkpoint
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/evaluate.py \
	  --config configs/default.yaml

analyze-risk:  ## Rank fraud risk and cluster learned embeddings
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/analyze_risk.py \
	  --config configs/default.yaml

evaluate-links: ## Evaluate embedding-based link prediction
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/evaluate_links.py \
	  --config configs/default.yaml

oversmoothing: ## Run over-smoothing depth sweep
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) $(SCRIPTS)/run_oversmoothing.py \
	  --config configs/default.yaml

# ── Notebooks ────────────────────────────────────────────────────────────────

notebook:      ## Start Jupyter Lab (all notebooks)
	jupyter lab $(NOTEBOOKS)/

colab:         ## Open GCN_Colab_Main.ipynb in browser
	$(PYTHON) -c "import webbrowser; webbrowser.open('$(NOTEBOOKS)/GCN_Colab_Main.ipynb')"

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:         ## Remove generated artefacts (preserves .gitkeep)
	find outputs -type f ! -name ".gitkeep" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
