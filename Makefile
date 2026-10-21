export UV_PROJECT_ENVIRONMENT := $(CURDIR)/artifacts/venv
export UV_CACHE_DIR := $(CURDIR)/artifacts/cache/uv
export PYTHONPYCACHEPREFIX := $(CURDIR)/artifacts/cache/python
export TMPDIR := $(CURDIR)/artifacts/tmp

.DEFAULT_GOAL := help
.NOTPARALLEL:

PYTHON_SOURCES := src tests scripts agent/skills
UV_SYNC := uv sync --frozen --no-editable --reinstall-package gemma-agent-dev
RUN := uv run --no-sync

.PHONY: help setup setup-submission format check verify coverage cli

help:
	@printf '%s\n' \
	  'setup             開発環境を準備・更新する' \
	  'setup-submission  提出SDKを含む開発環境を準備・更新する' \
	  'format            Pythonコードを整形する' \
	  'check             開発環境を更新して全品質検査を実行する' \
	  'verify            準備済み環境で全品質検査を実行する' \
	  'coverage          テストとカバレッジ基準を確認する' \
	  'cli ARGS=...      開発用CLIを実行する'

setup:
	mkdir -p "$(TMPDIR)"
	$(UV_SYNC)

setup-submission:
	mkdir -p "$(TMPDIR)"
	$(UV_SYNC) --group submission

format:
	$(RUN) ruff format $(PYTHON_SOURCES)

check: setup
	$(MAKE) --no-print-directory verify

verify:
	$(RUN) ruff format --check $(PYTHON_SOURCES)
	$(RUN) ruff check $(PYTHON_SOURCES)
	$(RUN) mypy
	$(RUN) python scripts/check_architecture.py
	$(MAKE) --no-print-directory coverage

coverage:
	rm -rf artifacts/coverage
	mkdir -p artifacts/coverage
	$(RUN) coverage run -m pytest -q
	$(RUN) coverage json
	$(RUN) coverage html
	$(RUN) coverage report
	$(RUN) python scripts/check_coverage.py

cli:
	$(RUN) gemma-dev $(ARGS)
