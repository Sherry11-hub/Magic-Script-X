# Makefile — Event & Script Copilot
# Usage:  make <target>
# Requires: Python 3.10+, pip

PYTHON    := python3
VENV      := .venv
PIP       := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST    := $(VENV)/bin/pytest

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  🎤 Generative AI Event & Script Copilot"
	@echo "  ════════════════════════════════════════"
	@echo "  make setup      — Create venv + install all dependencies"
	@echo "  make run        — Launch the Streamlit app"
	@echo "  make test       — Run the test suite"
	@echo "  make lint       — Run ruff linter"
	@echo "  make clean      — Remove venv, caches, and ChromaDB data"
	@echo "  make reset-kb   — Delete only the ChromaDB vector store"
	@echo "  make env        — Copy .env.example → .env (if .env missing)"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: setup
setup: $(VENV)/bin/activate
	@echo "✅  Environment ready. Run: make run"

$(VENV)/bin/activate: requirements.txt
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "📥 Installing dependencies (this may take a few minutes)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Environment file
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: env
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅  .env created from .env.example — add your OPENAI_API_KEY"; \
	else \
		echo "ℹ️   .env already exists — not overwriting."; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run
run:
	@if [ ! -f .env ]; then $(MAKE) env; fi
	$(STREAMLIT) run app.py

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTEST) tests/ -v --tb=short

# ─────────────────────────────────────────────────────────────────────────────
# Lint
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(VENV)/bin/ruff check . --fix

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "🧹 Removing virtual environment, caches, and ChromaDB data..."
	rm -rf $(VENV) __pycache__ core/__pycache__ ui/__pycache__ \
	       .pytest_cache .ruff_cache chroma_db .model_cache \
	       **/__pycache__ **/*.pyc
	@echo "✅  Clean."

.PHONY: reset-kb
reset-kb:
	@echo "⚠️  Deleting ChromaDB vector store (./chroma_db)..."
	rm -rf chroma_db
	@echo "✅  Knowledge base cleared. Re-upload documents on next run."
