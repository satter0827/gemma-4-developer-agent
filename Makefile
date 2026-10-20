export UV_PROJECT_ENVIRONMENT := $(CURDIR)/artifacts/venv
export UV_CACHE_DIR := $(CURDIR)/artifacts/cache/uv
export PYTHONPYCACHEPREFIX := $(CURDIR)/artifacts/cache/python
export TMPDIR := $(CURDIR)/artifacts/tmp

.PHONY: setup check verify coverage cli

setup:
	mkdir -p artifacts/tmp
	uv sync --frozen --no-editable --reinstall-package gemma-agent-dev

check: setup verify

verify:
	uv run --no-sync ruff format --check src tests kaggle scripts
	uv run --no-sync ruff check src tests kaggle scripts
	uv run --no-sync mypy
	uv run --no-sync python scripts/check_architecture.py
	$(MAKE) --no-print-directory coverage

coverage:
	rm -rf artifacts/coverage
	mkdir -p artifacts/coverage
	uv run --no-sync coverage run -m pytest -q
	uv run --no-sync coverage json
	uv run --no-sync coverage html
	uv run --no-sync coverage report
	uv run --no-sync python scripts/check_coverage.py

cli:
	uv run --no-sync gemma-dev $(ARGS)
