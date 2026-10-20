"""Run the official harness on Kaggle; GPU dependencies stay off the local machine."""

import asyncio
import glob
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash, inventory, write_json


def install_wheelhouse(directory: Path, output: Path) -> None:
    environment = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TRANSFORMERS_NO_TF": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    os.environ.update(environment)
    wheels = sorted(p for p in directory.glob("*.whl") if "cutlass" not in p.name.lower())
    if not wheels:
        raise FileNotFoundError(f"No official wheels in {directory}")
    write_json(output / "wheelhouse.json", {p.name: file_hash(p) for p in wheels})
    for pattern in (
        "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
        "/usr/local/lib/python*/site-packages/*cutlass*.pth",
    ):
        for name in glob.glob(pattern):
            Path(name).unlink(missing_ok=True)
    staging = output / "wheel-links"
    staging.mkdir()
    for wheel in wheels:
        name = wheel.name.replace("cu128", "+cu128") if "+" not in wheel.name else wheel.name
        (staging / name).symlink_to(wheel)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                *map(str, sorted(staging.glob("*.whl"))),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(staging)
    importlib.invalidate_caches()


def select_tasks(tasks: list[Any], ids: list[str]) -> list[Any]:
    available = {task.instance_id: task for task in tasks}
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(available):
        raise ValueError("Unknown, empty, or duplicate task IDs")
    selected = [available[task_id] for task_id in ids]
    if any(not task.patch.strip() or not task.test_patch.strip() for task in selected):
        raise ValueError("Every selected task must have reference and test patches")
    return selected


async def verify_controls(config: Any, tasks: list[Any], output: Path) -> bool:
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.evaluate import Evaluator
    from swegemma.harness.verification import verify_task

    evaluator = Evaluator(config)
    selected = select_tasks(tasks, config.task_ids)
    records = []
    for task in selected:
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
    return bool(records) and all(
        not record["unmodified"]["resolved"] and record["reference"]["resolved"]
        for record in records
    )


def validate_evaluation(output: Path, result: Any) -> None:
    from gemma_agent_dev.results import load_run

    saved = load_run(output, require_evaluation=True)
    if saved.problems or result.total != len(saved.task_ids) or result.resolved != saved.resolved:
        raise RuntimeError(f"Evaluation artifacts are incomplete or inconsistent: {saved.problems}")


def run(root: Path, provenance: dict[str, Any], *, verify_only: bool = False) -> None:
    config = json.loads((root / "configs/runtime.json").read_text())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(config["paths"]["output"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "verification" if verify_only else "evaluation",
        "run_id": run_id,
        "status": "running",
        "phase": "bootstrap",
        "started_at": datetime.now(UTC).isoformat(),
        "task_ids": config["task_ids"],
        **provenance,
        "runtime_config": config,
        "timings": {},
        "environment": {"python": platform.python_version(), "sandbox": "subprocess"},
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    server = None
    start = time.perf_counter()

    def phase(name: str) -> None:
        manifest["phase"] = name
        write_json(manifest_path, manifest)
        print(f"[{run_id}] {name}", flush=True)

    try:
        install_wheelhouse(Path(config["paths"]["wheelhouse"]), output)
        manifest["timings"]["bootstrap_seconds"] = time.perf_counter() - start
        packages = {name: importlib.metadata.version(name) for name in config["packages"]}
        manifest["environment"]["packages"] = packages
        if packages != config["packages"]:
            raise ValueError(f"Official package versions changed: {packages}")
        import litellm
        import torch
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import EvalConfig
        from swegemma.models import load_tasks

        from gemma_agent_dev.config import EvaluationSettings
        from gemma_agent_dev.official import create_server, evaluate, expected_model, make_validator
        from gemma_agent_dev.packaging import package_submission

        phase("prepare_submission")
        data = Path(config["paths"]["data"])
        submission = output / "submission"
        shutil.copytree(data / "sample_submission", submission)
        shutil.copytree(root / "agent", submission, dirs_exist_ok=True)
        settings = EvaluationSettings.load(submission / "eval_config.yaml")
        manifest["evaluation"] = settings.as_dict()
        manifest["submission_files"] = inventory(submission)
        manifest["tasks_sha256"] = file_hash(data / "tasks.jsonl")
        tasks = load_tasks(data / "tasks.jsonl")
        ids = config["task_ids"]
        select_tasks(tasks, ids)
        server = create_server(submission, config)
        manifest["vllm_config"] = asdict(server.config)
        manifest["verification_timeout_seconds"] = config["verification_timeout_seconds"]
        models = server.create_model_registry(
            aliases=[expected_model(config)], model_prefix="openai/", api_key="EMPTY"
        )
        evaluation = EvalConfig(
            tasks_path=data / "tasks.jsonl",
            snapshots_dir=data / "snapshots",
            results_dir=output,
            submission_dir=submission,
            models=models,
            sandbox="subprocess",
            wheels_dir=data / "wheels",
            task_ids=ids,
            graph_dir=str(data / "graphs"),
            embeddings_dir=str(data / "embeddings"),
            adapter_manifest=server.adapter_manifest,
            concurrency=1,
            display_mode="quiet",
            verification_timeout_seconds=config["verification_timeout_seconds"],
            context_cache_config=ContextCacheConfig(**config["context_cache"]),
            events_compaction_config=EventsCompactionConfig(**config["compaction"]),
            **settings.as_dict(),
        )
        phase("validate_package")
        package_submission(
            submission,
            output / "submission.zip",
            make_validator(evaluation, expected_model(config)),
        )
        phase("verify_controls")
        controls_start = time.perf_counter()
        manifest["controls_passed"] = asyncio.run(verify_controls(evaluation, tasks, output))
        manifest["timings"]["controls_seconds"] = time.perf_counter() - controls_start
        if not manifest["controls_passed"]:
            raise RuntimeError("Reference verification failed; inspect controls.json and logs")
        if verify_only:
            manifest["status"] = "completed"
            manifest["phase"] = "finished"
            return
        phase("start_model")
        gpu_count = torch.cuda.device_count()
        manifest["environment"]["gpus"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        if gpu_count != config["vllm"]["tensor_parallel_size"]:
            raise RuntimeError(
                f"Expected {config['vllm']['tensor_parallel_size']} GPUs, got {gpu_count}"
            )
        litellm.drop_params = True
        model_start = time.perf_counter()
        server.start()
        manifest["timings"]["model_startup_seconds"] = time.perf_counter() - model_start
        phase("evaluate")
        evaluation_start = time.perf_counter()
        result = asyncio.run(evaluate(evaluation))
        manifest["timings"]["evaluation_seconds"] = time.perf_counter() - evaluation_start
        if inventory(submission) != manifest["submission_files"]:
            raise ValueError("Evaluated submission changed after archive validation")
        phase("validate_results")
        validate_evaluation(output, result)
        manifest["status"] = "completed"
        manifest["phase"] = "finished"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["timings"]["total_seconds"] = time.perf_counter() - start
        write_json(manifest_path, manifest)
        if server is not None:
            try:
                server.stop()
            except Exception as cleanup_error:
                manifest["cleanup_error"] = str(cleanup_error)
            log_path = Path(server.log_path)
            if log_path.is_file():
                shutil.copyfile(log_path, output / "vllm.log")
                log_path.unlink()
        write_json(manifest_path, manifest)
