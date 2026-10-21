"""Replay patches on Kaggle CPU with official verification."""

import asyncio
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.common.events import bundle_session, phase
from gemma_agent_dev.common.junit import diagnostic_evidence
from gemma_agent_dev.evaluation.config import RuntimeSettings


def _run(root: Path, provenance: dict[str, Any], output: Path) -> None:
    from gemma_agent_dev.notebooks.environment import install_wheelhouse

    spec = json.loads((root / "diagnostics.json").read_text())
    runtime_file = root / "runtime.json"
    write_json(runtime_file, {**spec["runtime"], "task_ids": spec["task_ids"]})
    runtime = RuntimeSettings.load(runtime_file)
    if set(runtime.task_ids) != set(spec["snapshots"]):
        raise ValueError("Diagnostic task list and snapshots disagree")
    report: dict[str, Any] = {
        "kind": "diagnostic",
        "engine": "official-verify-task",
        "status": "running",
        "fingerprint": spec["fingerprint"],
        "provenance": provenance,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tasks_sha256": spec["tasks_sha256"],
        "snapshots": spec["snapshots"],
        "timeout_seconds": spec["timeout_seconds"],
        "lock_sha256": spec["lock_sha256"],
        "task_ids": runtime.task_ids,
        "cases": [],
    }
    write_json(output / "report.json", report)
    try:
        install_wheelhouse(
            Path(runtime.paths.wheelhouse),
            output,
            runtime.vllm.startup_timeout,
            runtime.wheelhouse_hashes,
        )
        versions = {name: importlib.metadata.version(name) for name in runtime.packages}
        if versions != runtime.packages:
            raise ValueError(f"Official package versions changed: {versions}")
        report["packages"] = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }
        spec["dependency_dir"] = str(root / "sandbox-wheels")
        asyncio.run(verify_cases(runtime, spec, output, report))
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "report.json", report)


async def verify_cases(
    runtime: RuntimeSettings, spec: dict[str, Any], output: Path, report: dict[str, Any]
) -> None:
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.evaluate import Evaluator
    from swegemma.harness.verification import verify_task
    from swegemma.models import load_tasks

    from gemma_agent_dev.evaluation.official import select_tasks

    data = Path(runtime.paths.data)
    if file_hash(data / "tasks.jsonl") != spec["tasks_sha256"]:
        raise ValueError("Task data differs from local diagnostics")
    tasks = {
        t.instance_id: t for t in select_tasks(load_tasks(data / "tasks.jsonl"), runtime.task_ids)
    }
    expected_cases = [
        (task_id, name)
        for task_id in runtime.task_ids
        for name in (
            ["reference", "unmodified"]
            + (
                ["generated"]
                if any(c["task_id"] == task_id and c["case"] == "generated" for c in spec["cases"])
                else []
            )
            + (
                ["probe"]
                if any(c["task_id"] == task_id and c["case"] == "probe" for c in spec["cases"])
                else []
            )
        )
    ]
    if [(c["task_id"], c["case"]) for c in spec["cases"]] != expected_cases:
        raise ValueError("Diagnostic cases are incomplete, duplicated or out of order")
    config = EvalConfig(
        tasks_path=data / "tasks.jsonl",
        snapshots_dir=data / "snapshots",
        results_dir=output,
        submission_dir=output / "unused-submission",
        models=ModelRegistry(),
        sandbox="subprocess",
        wheels_dir=data / "wheels",
        task_ids=runtime.task_ids,
        timeout_seconds=spec["timeout_seconds"],
        verification_timeout_seconds=spec["timeout_seconds"],
        display_mode="quiet",
    )
    evaluator = Evaluator(config)
    from gemma_agent_dev.diagnostics.sandbox import observe_sources

    check_sources = observe_sources(
        evaluator.sandbox, spec["sources"], output, Path(spec["dependency_dir"])
    )
    original_exec = evaluator.sandbox.exec
    active_case: list[Path] = []

    def capture_xml(*args: Any, **kwargs: Any) -> Any:
        result = original_exec(*args, **kwargs)
        invoked = args[1] if len(args) > 1 else kwargs.get("command", "")
        xml_path = re.search(r"--junitxml=(/tmp/_swegemma_junit_[a-f0-9]+\.xml)", invoked)
        if active_case and xml_path:
            xml = original_exec(args[0], f"cat {xml_path[1]} 2>/dev/null || true")
            if "<testsuite" in xml.stdout:
                (active_case[0] / "junit.xml").write_text(xml.stdout)
        if active_case and "<testsuite" in result.stdout and result.stdout.lstrip().startswith("<"):
            (active_case[0] / "junit.xml").write_text(result.stdout)
        return result

    evaluator.sandbox.exec = capture_xml
    for case in spec["cases"]:
        task_id, name = case["task_id"], case["case"]
        if task_id not in runtime.task_ids:
            raise ValueError("Unexpected diagnostic task")
        if hashlib.sha256(case["patch"].encode()).hexdigest() != case["patch_sha256"]:
            raise ValueError("Diagnostic patch checksum mismatch")
        snapshot, base, delta = resolve_task_snapshot_paths(
            data / "snapshots", task_id, tasks[task_id].repo
        )
        if (
            base is not None
            or delta is not None
            or file_hash(snapshot) != spec["snapshots"][task_id]
        ):
            raise ValueError("Snapshot differs from local diagnostics")
        destination = output / task_id / name
        destination.mkdir(parents=True)
        active_case[:] = [destination]
        (destination / "candidate.patch").write_text(case["patch"])
        record = {
            **{k: v for k, v in case.items() if k != "patch"},
            "test_patch_sha256": hashlib.sha256(tasks[task_id].test_patch.encode()).hexdigest(),
            "passed": False,
            "matched": False,
        }
        start = time.perf_counter()
        try:
            result = await verify_task(
                evaluator.sandbox,
                config,
                tasks[task_id],
                snapshot,
                agent_patch=case["patch"],
                start_time=start,
            )
            (destination / "test.log").write_text(result.test_output)
            record.update(
                passed=result.resolved, exit_code=result.test_exit_code, error=result.error
            )
            if result.error is not None:
                raise RuntimeError(f"Official verification failed: {result.error}")
            outcome, counts = diagnostic_evidence(
                result.test_exit_code, destination / "junit.xml", case["expected_outcome"]
            )
            record.update(outcome=outcome, junit=counts)
            if result.resolved != (result.test_exit_code == 0):
                raise ValueError("Official decision and test exit code disagree")
            record["matched"] = outcome == case["expected_outcome"]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if not (destination / "test.log").exists():
                (destination / "test.log").write_text(record["error"] + "\n")
        finally:
            record["duration_seconds"] = time.perf_counter() - start
            record["artifacts"] = inventory(destination)
            write_json(destination / "case.json", record)
            report["cases"].append(record)
            write_json(output / "report.json", report)
        check_sources()
        print(f"{task_id}/{name}: {record}", flush=True)
        if not record["matched"]:
            raise RuntimeError(f"Official diagnostic outcome mismatch: {task_id}/{name}")


def run(root: Path, provenance: dict[str, Any]) -> None:
    from gemma_agent_dev.common.artifacts import new_run_id

    spec = json.loads((root / "diagnostics.json").read_text())
    output = Path(spec["runtime"]["paths"]["output"]).parent / "diagnostics" / new_run_id()
    output.mkdir(parents=True, exist_ok=False)
    with bundle_session(root, output, provenance), phase("diagnostic.remote"):
        _run(root, provenance, output)
