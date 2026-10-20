"""Run the official harness on Kaggle; GPU dependencies stay off the local machine."""

import asyncio
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.candidates import extract_archive, validate_references
from gemma_agent_dev.config import RuntimeSettings
from gemma_agent_dev.environment import install_wheelhouse
from gemma_agent_dev.official import select_tasks, verify_controls


def validate_evaluation(output: Path, result: Any) -> dict[str, str]:
    from gemma_agent_dev.results import load_run

    saved = load_run(output, require_evaluation=True)
    if saved.problems or result.total != len(saved.task_ids) or result.resolved != saved.resolved:
        raise RuntimeError(f"Evaluation artifacts are incomplete or inconsistent: {saved.problems}")

    hashes = {}
    for task, row in saved.rows.items():
        expected = [f"logs/{task}.log", f"traces/trace_{task}.json"]
        if row["agent_patch_size"] > 0:
            expected.extend([f"patches/{task}.patch", f"test_outputs/{task}.log"])
        for name in expected:
            path = output / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing evaluation artifact: {name}")
            if path.suffix == ".json":
                json.loads(path.read_text())
            hashes[name] = file_hash(path)
    for name in ("summary.json", "task_results.jsonl"):
        hashes[name] = file_hash(output / name)
    return hashes


def run(root: Path, provenance: dict[str, Any], *, verify_only: bool = False) -> None:
    config = RuntimeSettings.load(root / "configs/runtime.json").as_dict()
    run_id = new_run_id()
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
    failure: BaseException | None = None
    export = Path(config["paths"]["submission_export"])
    export_temporary = export.with_suffix(".zip.tmp")

    def phase(name: str) -> None:
        manifest["phase"] = name
        write_json(manifest_path, manifest)
        print(f"[{run_id}] {name}", flush=True)

    try:
        export.unlink(missing_ok=True)
        export_temporary.unlink(missing_ok=True)
        install_wheelhouse(
            Path(config["paths"]["wheelhouse"]), output, config["vllm"]["startup_timeout"]
        )
        manifest["timings"]["bootstrap_seconds"] = time.perf_counter() - start
        packages = {name: importlib.metadata.version(name) for name in config["packages"]}
        manifest["environment"]["packages"] = packages
        if packages != config["packages"]:
            raise ValueError(f"Official package versions changed: {packages}")
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import EvalConfig
        from swegemma.deduplication import resolve_task_snapshot_paths
        from swegemma.models import load_tasks

        from gemma_agent_dev.config import EvaluationSettings
        from gemma_agent_dev.official import create_server, evaluate, expected_model, make_validator

        phase("prepare_submission")
        data = Path(config["paths"]["data"])
        submission = output / "submission"
        candidate = json.loads((root / "candidate.json").read_text())
        if (
            file_hash(root / "candidate.zip") != candidate["sha256"]
            or candidate["sha256"] != provenance["candidate_sha256"]
        ):
            raise ValueError("Candidate archive checksum mismatch")
        extract_archive(root / "candidate.zip", submission, candidate["max_unpacked_bytes"])
        if validate_references(submission) != candidate["files"]:
            raise ValueError("Candidate files changed")
        settings = EvaluationSettings.load(submission / "eval_config.yaml")
        manifest["evaluation"] = settings.as_dict()
        manifest["submission_files"] = inventory(submission)
        manifest["tasks_sha256"] = file_hash(data / "tasks.jsonl")
        if manifest["tasks_sha256"] != candidate["tasks_sha256"]:
            raise ValueError("Task data differs from preflight")
        tasks = load_tasks(data / "tasks.jsonl")
        ids = config["task_ids"]
        manifest["snapshots"] = {}
        for task in select_tasks(tasks, ids):
            snapshot, base, delta = resolve_task_snapshot_paths(
                data / "snapshots", task.instance_id, task.repo
            )
            digest = file_hash(snapshot)
            if (
                base is not None
                or delta is not None
                or digest != candidate["snapshots"][task.instance_id]
            ):
                raise ValueError("Snapshot differs from preflight")
            manifest["snapshots"][task.instance_id] = digest
        server = create_server(submission, config)
        # The official server removes its temporary log when stopping.
        (output / "vllm.log").hardlink_to(server.log_path)
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
            concurrency=config["concurrency"],
            display_mode="quiet",
            verification_timeout_seconds=config["verification_timeout_seconds"],
            context_cache_config=ContextCacheConfig(**config["context_cache"]),
            events_compaction_config=EventsCompactionConfig(**config["compaction"]),
            **settings.as_dict(),
        )
        phase("validate_package")
        make_validator(evaluation, expected_model(config))(submission)
        shutil.copyfile(root / "candidate.zip", output / "submission.zip")
        manifest["candidate_sha256"] = file_hash(output / "submission.zip")
        phase("verify_controls")
        controls_start = time.perf_counter()
        manifest["controls_passed"] = asyncio.run(verify_controls(evaluation, tasks, output))
        manifest["timings"]["controls_seconds"] = time.perf_counter() - controls_start
        if not manifest["controls_passed"]:
            raise RuntimeError("Reference verification failed; inspect controls.json and logs")
        if not verify_only:
            import litellm
            import torch

            phase("start_model")
            gpu_count = torch.cuda.device_count()
            manifest["environment"]["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(gpu_count)
            ]
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
            phase("validate_results")
            manifest["evaluation_artifacts"] = validate_evaluation(output, result)
        if inventory(submission) != manifest["submission_files"]:
            raise ValueError("Validated submission changed")
    except BaseException as error:
        failure = error
        manifest["error"] = f"{type(error).__name__}: {error}"
    # Cleanup must run even when manifest storage is unavailable.
    if server is not None:
        try:
            server.stop()
        except BaseException as error:
            manifest["cleanup_error"] = f"{type(error).__name__}: {error}"
            failure = failure or error
        finally:
            try:
                Path(server.log_path).unlink(missing_ok=True)
            except OSError as error:
                manifest["cleanup_error"] = str(error)
                failure = failure or error
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["timings"]["total_seconds"] = time.perf_counter() - start
    manifest["status"] = "failed" if failure else "completed"
    manifest["phase"] = manifest["phase"] if failure else "finished"
    try:
        write_json(manifest_path, manifest)
        if failure is None:
            shutil.copyfile(output / "submission.zip", export_temporary)
            if file_hash(export_temporary) != provenance["candidate_sha256"]:
                raise ValueError("Exported candidate changed")
            export_temporary.replace(export)
    except BaseException as error:
        failure = failure or error
        manifest.update(status="failed", error=f"{type(failure).__name__}: {failure}")
        try:
            write_json(manifest_path, manifest)
        except OSError as save_error:
            print(f"Cannot save failure manifest: {save_error}", file=sys.stderr)
    if failure is not None:
        try:
            export_temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(f"Cannot remove failed export: {cleanup_error}", file=sys.stderr)
        raise failure
