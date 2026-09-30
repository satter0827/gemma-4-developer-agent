export UV_PROJECT_ENVIRONMENT := $(CURDIR)/artifacts/venv
export UV_CACHE_DIR := $(CURDIR)/artifacts/cache/uv
export PYTHONPYCACHEPREFIX := $(CURDIR)/artifacts/cache/python
export TMPDIR := $(CURDIR)/artifacts/tmp

.PHONY: setup check cli

setup:
	mkdir -p artifacts/tmp
	uv sync --frozen --no-editable --reinstall-package gemma-agent-dev

check: setup
	uv run --no-sync ruff format --check src tests kaggle
	uv run --no-sync ruff check src tests kaggle
	uv run --no-sync mypy
	uv run --no-sync pytest -q

cli:
	uv run --no-sync gemma-dev $(ARGS)
