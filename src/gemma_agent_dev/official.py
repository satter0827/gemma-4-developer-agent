"""Connect submissions to the current official compiler and inference registry."""

from collections.abc import Callable
from pathlib import Path
from typing import Any


def expected_model(runtime: dict[str, Any]) -> str:
    name = str(runtime["model_source"]).split("/")[-2]
    if Path(runtime["paths"]["model"]).parent.name != name:
        raise ValueError("Model source and mounted model path disagree")
    return name


def validate_model(directory: Path, expected: str) -> None:
    from adk_submission import discover_declared_models

    declared = discover_declared_models(directory)
    if declared != {expected}:
        raise ValueError(f"Expected model {expected!r}, found {sorted(declared)}")


def create_server(directory: Path, runtime: dict[str, Any]) -> Any:
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    validate_model(directory, expected_model(runtime))
    adapters = discover_adapters(directory, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    return VllmServer(
        VllmConfig(model=runtime["paths"]["model"], **runtime["vllm"]),
        adapter_manifest=adapters,
    )


def make_validator(config: Any, expected: str) -> Callable[[Path], None]:
    from adk_submission import compile_submission, discover_adapters, validate_directory
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    from swegemma.context import SwegemmaContext

    context = SwegemmaContext(budget=config.budget, harness=config.harness)

    def validate(directory: Path) -> None:
        validate_directory(directory, limits=config.limits)
        validate_model(directory, expected)
        adapters = discover_adapters(directory, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        compile_submission(
            submission_dir=directory,
            tool_registry=context.create_tools(),
            model_registry=config.models,
            limits=config.limits,
            generation_constraints=config.generation_constraints,
            adapter_manifest=adapters,
            adapter_resolver_fn=lambda _base, info: config.models.get(info.name),
        )

    return validate
