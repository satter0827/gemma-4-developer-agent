"""Connect submissions to the current official compiler and inference registry."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.common.events import logged


@logged("model.start")
def start_model(server: Any, runtime: dict[str, Any]) -> float:
    import litellm
    import torch

    if torch.cuda.device_count() != runtime["vllm"]["tensor_parallel_size"]:
        raise RuntimeError("Unexpected GPU count")
    litellm.drop_params = True
    started = time.monotonic()
    server.start()
    return time.monotonic() - started


@logged("model.stop")
def stop_model(server: Any) -> None:
    server.stop()


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


async def evaluate(
    config: Any, sources: dict[str, Any] | None = None, dependency_dir: Path | None = None
) -> Any:
    from swegemma.evaluate import Evaluator

    errors: list[str] = []

    class EvaluationErrors(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            errors.append(record.getMessage())

    logger = logging.getLogger("swegemma.evaluate")
    handler = EvaluationErrors(level=logging.ERROR)
    logger.addHandler(handler)
    try:
        evaluator = Evaluator(config)
        if sources is not None:
            from gemma_agent_dev.diagnostics.sandbox import observe_sources

            check = observe_sources(evaluator.sandbox, sources, config.results_dir, dependency_dir)
        result = await evaluator.run()
        if sources is not None:
            check()
    finally:
        logger.removeHandler(handler)
        handler.close()
    if errors:
        raise RuntimeError(f"Official evaluation reported errors: {errors}")
    return result


def select_tasks(tasks: list[Any], ids: list[str]) -> list[Any]:
    selected_ids = [task.instance_id for task in tasks if task.instance_id in ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate selected task data")
    available = {task.instance_id: task for task in tasks}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(available):
        raise ValueError("Unknown, empty, or duplicate task IDs")
    selected = [available[task_id] for task_id in ids]
    if any(not task.patch.strip() or not task.test_patch.strip() for task in selected):
        raise ValueError("Every selected task must have reference and test patches")
    return selected


async def verify_controls(
    config: Any,
    tasks: list[Any],
    output: Path,
    sources: dict[str, Any] | None = None,
    dependency_dir: Path | None = None,
) -> bool:
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.evaluate import Evaluator
    from swegemma.harness.verification import verify_task

    selected = select_tasks(tasks, config.task_ids)
    records = []
    for task in selected:
        evaluator = Evaluator(config)
        if sources is not None:
            from gemma_agent_dev.diagnostics.sandbox import observe_sources

            check = observe_sources(
                evaluator.sandbox,
                {task.instance_id: sources[task.instance_id]},
                output,
                dependency_dir,
            )
        snapshot, base, patch = resolve_task_snapshot_paths(
            config.snapshots_dir,
            task.instance_id,
            task.repo,
        )
        outcomes = {}
        for label, solution in (("unmodified", ""), ("reference", task.patch)):
            result = await verify_task(
                evaluator.sandbox,
                config,
                task,
                snapshot,
                base_snapshot_path=base,
                patch_path=patch,
                agent_patch=solution,
                start_time=time.perf_counter(),
            )
            (output / f"control-{task.instance_id}-{label}.log").write_text(
                result.test_output,
                encoding="utf-8",
            )
            outcomes[label] = {
                "resolved": result.resolved,
                "exit_code": result.test_exit_code,
                "error": result.error,
                "duration_seconds": result.duration_seconds,
            }
        records.append({"instance_id": task.instance_id, "repo": task.repo, **outcomes})
        write_json(output / "controls.json", records)
        if sources is not None:
            check()
    return bool(records) and all(
        not record["unmodified"]["resolved"] and record["reference"]["resolved"]
        for record in records
    )


def generation_contract(directory: Path) -> list[dict[str, Any]]:
    """Read every selected LLM node through the pinned SDK's include resolver."""
    import json

    from adk_submission.yaml_loader import load_yaml

    contracts = {}
    for path in directory.rglob("*.yaml"):
        data = load_yaml(path, directory)
        if isinstance(data, dict) and "model" in data:
            value = {"model": data["model"], "generation": data.get("generate_content_config")}
            if value["generation"] is None:
                raise ValueError("Every experiment agent must declare sampling explicitly")
            contracts[json.dumps(value, sort_keys=True)] = value
    if not contracts:
        raise ValueError("Experiment has no declared LLM agents")
    return [contracts[key] for key in sorted(contracts)]
