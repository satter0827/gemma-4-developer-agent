"""Run isolated task/variant cells while sharing only one model server."""

import asyncio
import importlib.metadata
import json
import os
import platform
import shutil
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import (
    declared_skills,
    extract_archive,
    generation_contract,
    validate_references,
)
from gemma_agent_dev.candidates.changes import ChangeScope, validate_change, validate_generation
from gemma_agent_dev.common.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.common.events import bundle_session, context, event, logged
from gemma_agent_dev.common.processes import time_limit
from gemma_agent_dev.evaluation.analysis import emit_tool_events
from gemma_agent_dev.evaluation.config import EvaluationSettings, RuntimeSettings
from gemma_agent_dev.evaluation.control_evidence import control_identity, copy_native_controls
from gemma_agent_dev.evaluation.official import (
    create_server,
    evaluate,
    expected_model,
    make_validator,
    select_tasks,
    start_model,
    stop_model,
    verify_controls,
)
from gemma_agent_dev.evaluation.results import load_run
from gemma_agent_dev.evaluation.runtime import validate_evaluation
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR, SessionPolicy
from gemma_agent_dev.notebooks.environment import install_wheelhouse


def model_conditions(runtime: dict[str, Any], output: Path) -> dict[str, Any]:
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        available = None
    model = Path(runtime["paths"]["model"])
    return {
        "source": runtime["model_source"],
        "path": str(model),
        "load_strategy": "default",
        "placement": "configured_path",
        "physical_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        "available_memory_bytes": available,
        "output_disk": shutil.disk_usage(output)._asdict(),
        "model_disk": shutil.disk_usage(model)._asdict() if model.exists() else None,
    }


def evaluation_config(
    runtime: dict[str, Any], server: Any, submission: Path, output: Path, ids: list[str]
) -> Any:
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig

    data = Path(runtime["paths"]["data"])
    return EvalConfig(
        tasks_path=data / "tasks.jsonl",
        snapshots_dir=data / "snapshots",
        results_dir=output,
        submission_dir=submission,
        models=server.create_model_registry(
            aliases=[expected_model(runtime)], model_prefix="openai/", api_key="EMPTY"
        ),
        sandbox="subprocess",
        wheels_dir=data / "wheels",
        task_ids=ids,
        graph_dir=str(data / "graphs"),
        embeddings_dir=str(data / "embeddings"),
        adapter_manifest=server.adapter_manifest,
        concurrency=runtime["concurrency"],
        display_mode="quiet",
        verification_timeout_seconds=runtime["verification_timeout_seconds"],
        context_cache_config=ContextCacheConfig(**runtime["context_cache"]),
        events_compaction_config=EventsCompactionConfig(**runtime["compaction"]),
        **EvaluationSettings.load(submission / "eval_config.yaml").as_dict(),
    )


def prepare_candidates(
    root: Path, output: Path, spec: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Path]:
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.models import load_tasks

    data = Path(runtime["paths"]["data"])
    if file_hash(data / "tasks.jsonl") != spec["tasks_sha256"]:
        raise ValueError("Task inventory changed")
    tasks = load_tasks(data / "tasks.jsonl")
    for task in select_tasks(tasks, spec["definition"]["task_ids"]):
        snapshot, base, delta = resolve_task_snapshot_paths(
            data / "snapshots", task.instance_id, task.repo
        )
        if (
            base is not None
            or delta is not None
            or file_hash(snapshot) != spec["snapshots"][task.instance_id]
        ):
            raise ValueError("Experiment snapshot changed")
    submissions = {}
    common = None
    allowed = (
        ChangeScope.parse(spec["definition"]["change"]).generation_fields
        if "change" in spec["definition"]
        else []
    )
    for name, candidate in spec["variants"].items():
        archive = root / name / "submission.zip"
        if file_hash(archive) != candidate["sha256"]:
            raise ValueError("Experiment candidate changed")
        submission = output / "submissions" / name
        extract_archive(archive, submission, candidate["max_unpacked_bytes"])
        files = validate_references(submission)
        if files != candidate["files"]:
            raise ValueError("Experiment candidate files changed")
        conditions = {
            k: v for k, v in files.items() if k.startswith("adapters/") or k == "eval_config.yaml"
        }
        conditions["@generation"] = json.dumps(
            sorted(
                {
                    json.dumps(
                        {
                            "model": contract["model"],
                            "generation": {
                                k: v for k, v in contract["generation"].items() if k not in allowed
                            },
                        },
                        sort_keys=True,
                    )
                    for contract in generation_contract(submission)
                }
            )
        )
        if common is not None and common != conditions:
            raise ValueError("Shared sessions require identical adapters, sampling and evaluation")
        common = conditions
        submissions[name] = submission
    if "change" in spec["definition"]:
        validate_change(
            ChangeScope.parse(spec["definition"]["change"]),
            submissions,
            {name: item["files"] for name, item in spec["variants"].items()},
        )
    else:
        validate_generation(submissions, [])
    return submissions


@logged("evaluation.cell")
def run_cell(
    root: Path,
    output: Path,
    runtime: dict[str, Any],
    spec: dict[str, Any],
    server: Any,
    submission: Path,
    cell: dict[str, str],
    provenance: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True)
    task, variant = cell["task_id"], cell["variant"]
    candidate = spec["variants"][variant]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "evaluation",
        "run_id": output.parent.parent.name + "-" + output.name,
        "status": "running",
        "phase": "evaluate",
        "task_ids": [task],
        **provenance,
        "experiment_id": spec["definition"]["experiment_id"],
        "definition_sha256": spec["definition_sha256"],
        "variant": variant,
        "candidate_sha256": candidate["sha256"],
        "submission_files": candidate["files"],
        "runtime_config": runtime,
        "evaluation": EvaluationSettings.load(submission / "eval_config.yaml").as_dict(),
        "generation": generation_contract(submission),
        "tasks_sha256": spec["tasks_sha256"],
        "snapshots": {task: spec["snapshots"][task]},
        "source_setup": {task: spec["sources"][task]},
        "controls_passed": True,
        "wheelhouse_hashes": runtime["wheelhouse_hashes"],
        "vllm_config": asdict(server.config),
        "verification_timeout_seconds": runtime["verification_timeout_seconds"],
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "environment": {"python": platform.python_version(), "sandbox": "subprocess"},
        "controls": controls,
        "timings": {},
    }
    path = output / "manifest.json"
    write_json(path, manifest)
    start = time.monotonic()
    failure: BaseException | None = None
    try:
        configuration = evaluation_config(runtime, server, submission, output, [task])
        # A new EvalConfig, model registry and Evaluator create a fresh conversation and sandbox.
        result = asyncio.run(
            evaluate(configuration, {task: spec["sources"][task]}, root / "sandbox-wheels")
        )
        manifest["evaluation_artifacts"] = validate_evaluation(output, result)
        if "max_artifact_bytes" in spec:
            emit_tool_events(load_run(output), task, spec["max_artifact_bytes"])
        if inventory(submission) != candidate["files"]:
            raise ValueError("Validated candidate changed during evaluation")
        manifest["status"] = "completed"
        event("evaluation.outcome", resolved=result.resolved == result.total, count=result.total)
    except BaseException as error:
        failure = error
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        manifest["timings"]["total_seconds"] = time.monotonic() - start
        try:
            write_json(path, manifest)
        except OSError as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Cannot save cell evidence: {error}")
    if failure is not None:
        raise failure
    return cell_reference(output, cell)


def cell_reference(output: Path, cell: dict[str, str]) -> dict[str, Any]:
    path = output / "manifest.json"
    return {
        **cell,
        "manifest": str(path.relative_to(output.parent.parent)),
        "sha256": file_hash(path),
    }


def _run(
    root: Path,
    provenance: dict[str, Any],
    *,
    verify_only: bool = False,
    bindings: dict[str, str] | None = None,
    output: Path,
) -> Path:
    runtime = RuntimeSettings.load(root / "runtime.json").as_dict()
    spec = json.loads((root / "experiment.json").read_text())
    policy = SessionPolicy.parse(spec["definition"]["session"])
    runtime["vllm"]["extra_args"] = ["--no-enable-prefix-caching"]
    if bindings:
        if not verify_only or set(bindings) != {"data", "output", "native_controls"}:
            raise ValueError("Local path bindings are restricted to CPU verification")
        runtime["paths"].update({k: v for k, v in bindings.items() if k != "native_controls"})
    started = time.monotonic()
    job_started = min(started, float(os.environ.get("GEMMA_JOB_STARTED_MONOTONIC", str(started))))
    deadline = min(
        started + policy.job_timeout_seconds,
        float(
            os.environ.get(
                "GEMMA_JOB_DEADLINE_MONOTONIC", str(started + policy.job_timeout_seconds)
            )
        ),
    )
    path = output / "session.json"
    report: dict[str, Any] = {
        "kind": "experiment-verification" if verify_only else "experiment-session",
        "status": "running",
        **provenance,
        "definition_sha256": spec["definition_sha256"],
        "definition": spec["definition"],
        "experiment_id": spec["definition"]["experiment_id"],
        "candidate_files": {name: value["files"] for name, value in spec["variants"].items()},
        "planned_order": spec["definition"]["planned_order"],
        "task_ids": spec["definition"]["task_ids"],
        "variants": {name: value["sha256"] for name, value in spec["variants"].items()},
        "worker_completion_required": bool(
            provenance.get("job_id")
            and Path(os.environ.get("GEMMA_PROCESS_REGISTRY", "")).parent.name
            == provenance["job_id"]
        ),
        "actual_order": [],
        "cells": [],
        "runtime_config": runtime,
        "evidence_inputs": {k: spec[k] for k in ("tasks_sha256", "snapshots", "sources")},
        "sdk_packages": runtime["packages"],
        "environment": {"python": platform.python_version(), "sandbox": "subprocess"},
        "control_engine": "native-python" if bindings else "official-sdk",
        "timings": {"bootstrap_seconds": started - job_started},
    }
    write_json(path, report)
    server = None
    failure: BaseException | None = None
    evaluation_started: float | None = None
    try:
        report["completion_budget"] = policy.completion(
            spec["definition"]["evaluation"], runtime, len(report["task_ids"])
        )
        if not verify_only:
            policy.require_completion(
                spec["definition"]["evaluation"], runtime, len(report["task_ids"])
            )
        guard: AbstractContextManager[None] = nullcontext()
        if not verify_only:
            assert policy.preparation_timeout_seconds is not None
            guard = time_limit(
                policy.preparation_timeout_seconds - (time.monotonic() - job_started)
            )
        with guard:
            if not bindings:
                preparation_remaining = None
                if not verify_only:
                    assert policy.preparation_timeout_seconds is not None
                    preparation_remaining = policy.preparation_timeout_seconds - (
                        time.monotonic() - job_started
                    )
                install_wheelhouse(
                    Path(runtime["paths"]["wheelhouse"]),
                    output,
                    runtime["vllm"]["startup_timeout"],
                    runtime["wheelhouse_hashes"],
                    preparation_timeout=preparation_remaining,
                )
            report["timings"]["dependencies_seconds"] = time.monotonic() - started
            inputs_started = time.monotonic()
            if {name: importlib.metadata.version(name) for name in runtime["packages"]} != runtime[
                "packages"
            ]:
                raise ValueError("Official SDK versions changed")
            submissions = prepare_candidates(root, output, spec, runtime)
            report["generation"] = {
                name: generation_contract(folder) for name, folder in submissions.items()
            }
            report["candidate_archives"] = {}
            for name, candidate in spec["variants"].items():
                archive = output / "archives" / name / "submission.zip"
                archive.parent.mkdir(parents=True)
                shutil.copyfile(root / name / "submission.zip", archive)
                report["candidate_archives"][name] = {
                    "path": archive.relative_to(output).as_posix(),
                    "sha256": file_hash(archive),
                }
                if report["candidate_archives"][name]["sha256"] != candidate["sha256"]:
                    raise ValueError("Archived experiment candidate changed")
            server = create_server(next(iter(submissions.values())), runtime)
            (output / "vllm.log").hardlink_to(server.log_path)
            if verify_only:
                from gemma_agent_dev.evaluation.sdk_worker import request_contract
                from gemma_agent_dev.evaluation.sequential_checks import run as check_sequential
                from gemma_agent_dev.evaluation.skill_checks import run as check_skill

                for name, submission in submissions.items():
                    report.setdefault("generation_requests", {})[name] = [
                        asyncio.run(request_contract(contract["generation"]))
                        for contract in report["generation"][name]
                    ]
                    if "sub_agents/verify-v1.yaml" in spec["variants"][name]["files"]:
                        report["sequential_contract"] = check_sequential(
                            submission, output / "sequential-contract.json"
                        )
                    if declared_skills(submission):
                        report.setdefault("skill_contracts", {})[name] = check_skill(
                            submission,
                            output / "contracts" / name / "skill",
                            runtime["verification_timeout_seconds"],
                        )
            from swegemma.models import load_tasks

            tasks = load_tasks(Path(runtime["paths"]["data"]) / "tasks.jsonl")
            folder = output / "controls/shared"
            folder.mkdir(parents=True)
            configurations = []
            for submission in submissions.values():
                configuration = evaluation_config(
                    runtime, server, submission, folder, spec["definition"]["task_ids"]
                )
                make_validator(configuration, expected_model(runtime))(submission)
                configurations.append(configuration)
            report["timings"]["inputs_seconds"] = time.monotonic() - inputs_started
            controls_started = time.monotonic()
            configuration = configurations[0]
            if bindings:
                copy_native_controls(
                    Path(bindings["native_controls"]),
                    spec["native_controls_sha256"],
                    spec["definition"]["task_ids"],
                    folder,
                )
            elif not asyncio.run(
                verify_controls(
                    configuration, tasks, folder, spec["sources"], root / "sandbox-wheels"
                )
            ):
                raise RuntimeError("Experiment controls failed")
            report["control_evidence"] = {
                "directory": "controls/shared",
                "condition_sha256": control_identity(report),
                "files": inventory(folder),
            }
            report["timings"]["controls_seconds"] = time.monotonic() - controls_started
            write_json(path, report)
        report["timings"]["cpu_seconds"] = time.monotonic() - job_started
        if not verify_only:
            report["model_preparation"] = model_conditions(runtime, output)
            with time_limit(runtime["vllm"]["startup_timeout"]):
                report["timings"]["model_startup_seconds"] = start_model(server, runtime)
            evaluation_started = time.monotonic()
            order = report["planned_order"]
            for offset in range(0, len(order), VARIANTS_PER_PAIR):
                remaining = deadline - time.monotonic()
                if remaining < policy.pair_seconds(
                    spec["definition"]["evaluation"], runtime["verification_timeout_seconds"]
                ):
                    report["stop_reason"] = "insufficient_time_for_complete_pair"
                    break
                for index, cell in enumerate(
                    order[offset : offset + VARIANTS_PER_PAIR], start=offset
                ):
                    report["actual_order"].append(cell)
                    write_json(path, report)
                    cell_output = (
                        output / "cells" / f"{index:03d}-{cell['task_id']}-{cell['variant']}"
                    )
                    try:
                        with (
                            time_limit(policy.evaluation_timeout_seconds)
                            if policy.evaluation_timeout_seconds is not None
                            else nullcontext(),
                            context(
                                task_id=cell["task_id"],
                                variant=cell["variant"],
                                candidate_sha256=spec["variants"][cell["variant"]]["sha256"],
                            ),
                        ):
                            record = run_cell(
                                root,
                                cell_output,
                                runtime,
                                spec,
                                server,
                                submissions[cell["variant"]],
                                cell,
                                provenance,
                                report["control_evidence"],
                            )
                        report["cells"].append(record)
                    except BaseException:
                        cell_path = cell_output / "manifest.json"
                        if cell_path.is_file():
                            report["cells"].append(cell_reference(cell_output, cell))
                        raise
                    write_json(path, report)
        report["status"] = (
            "completed"
            if verify_only or len(report["cells"]) == len(report["planned_order"])
            else "incomplete"
        )
    except BaseException as error:
        failure = error
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if isinstance(error, TimeoutError):
            report["stop_reason"] = (
                "evaluation_deadline_exceeded"
                if evaluation_started is not None
                else "model_startup_deadline_exceeded"
                if "cpu_seconds" in report["timings"]
                else "preparation_deadline_exceeded"
            )
    finally:
        if evaluation_started is not None:
            report["timings"]["evaluation_seconds"] = time.monotonic() - evaluation_started
        cleanup_started = time.monotonic()
        if server is not None:
            try:
                stop_model(server)
            except BaseException as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
        report["timings"]["cleanup_seconds"] = time.monotonic() - cleanup_started
        report["timings"]["total_seconds"] = time.monotonic() - job_started
        if failure is not None:
            report["status"] = "failed"
        try:
            write_json(path, report)
        except OSError as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Cannot save session evidence: {error}")
    if failure is not None:
        raise failure
    return path


def run(
    root: Path,
    provenance: dict[str, Any],
    *,
    verify_only: bool = False,
    bindings: dict[str, str] | None = None,
) -> Path:
    runtime = RuntimeSettings.load(root / "runtime.json")
    directory = Path(bindings["output"]) if bindings else Path(runtime.paths.output)
    output = directory / new_run_id()
    output.mkdir(parents=True)
    with bundle_session(root, output, provenance), context(session_id=output.name):
        try:
            return _run(root, provenance, verify_only=verify_only, bindings=bindings, output=output)
        except BaseException:
            event("session.exception", level="ERROR", exception=True)
            raise
