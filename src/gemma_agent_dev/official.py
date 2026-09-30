"""Validate submissions using the installed competition compiler and registries."""

from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_validator(config: Any) -> Callable[[Path], None]:
    from adk_submission import compile_submission, discover_adapters, validate_directory
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    from swegemma.context import SwegemmaContext
    from swegemma.models.discovery import validate_single_declared_model
    from swegemma.models.registry import resolve_swegemma_adapter

    context = SwegemmaContext(budget=config.budget, harness=config.harness)

    def validate(directory: Path) -> None:
        validate_directory(directory, limits=config.limits)
        if validate_single_declared_model(directory) != "gemma-4-31b-it-qat-w4a16-ct":
            raise ValueError("Submission must declare the competition model")
        adapters = discover_adapters(directory, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        compile_submission(
            submission_dir=directory,
            tool_registry=context.create_tools(),
            model_registry=config.models,
            limits=config.limits,
            generation_constraints=config.generation_constraints,
            adapter_manifest=adapters,
            adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(
                base, info, config.models
            ),
        )

    return validate
