.PHONY: setup check view

setup:
	uv sync --frozen --no-editable --reinstall-package gemma-agent-dev

check: setup
	uv run --no-sync ruff format --check src tests kaggle
	uv run --no-sync ruff check src tests kaggle
	uv run --no-sync mypy
	uv run --no-sync pytest -q

view:
	uv run --no-sync gemma-dev view --results results
