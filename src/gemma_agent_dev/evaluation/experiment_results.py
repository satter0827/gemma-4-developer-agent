"""Compare complete task pairs without turning missing evidence into failures."""

import copy
import json
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import generation_contract
from gemma_agent_dev.candidates.changes import ChangeScope, validate_change
from gemma_agent_dev.common.artifacts import content_hash, file_hash, inventory
from gemma_agent_dev.common.events import load_health
from gemma_agent_dev.evaluation.analysis import (
    SCORABLE_TERMINATIONS,
    Behavior,
    behavior,
    implementation_changed,
    task_evidence,
    termination_reason,
    unknown_behavior,
)
from gemma_agent_dev.evaluation.control_evidence import session_controls
from gemma_agent_dev.evaluation.results import (
    comparison_differences,
    load_run,
    require_evaluation_evidence,
)
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR


class TaskObservation(Behavior):
    task_id: str
    variant: str
    run_id: str | None
    resolved: bool | None
    duration_seconds: float | None
    status: str
    termination: str
    implementation_changed: bool | None
    tool_argument_errors: int | None


def validate_worker_completion(path: Path, report: dict[str, Any]) -> None:
    if not report.get("worker_completion_required"):
        return
    evidence = report.get("worker_evidence", {})
    if report.get("status") != "completed" and set(evidence) != {
        "watchdog.json",
        "processes.jsonl",
    }:
        return
    job = report.get("job_id")
    if not isinstance(job, str) or Path(job).name != job or job in {"", ".", ".."} or "\\" in job:
        raise ValueError("Invalid worker identity")
    if set(evidence) != {"watchdog.json", "processes.jsonl"}:
        raise ValueError("Worker completion evidence is missing")
    artifact_root = path.parent.parent.parent
    for name, item in evidence.items():
        expected = f"artifacts/jobs/{job}/{name}"
        if item.get("path") != expected:
            raise ValueError("Worker evidence belongs to another job")
        target = artifact_root.parent / expected
        if file_hash(target) != item["sha256"]:
            raise ValueError("Worker completion evidence changed")
    receipt = json.loads((artifact_root / "jobs" / job / "watchdog.json").read_text())
    if any(
        receipt.get("provenance", {}).get(key) != report.get(key)
        for key in ("job_id", "source_sha256", "definition_sha256")
        if key in report
    ):
        raise ValueError("Worker completion identity changed")
    if report["status"] == "completed" and (
        receipt.get("status") != "completed" or receipt.get("cleanup_error")
    ):
        raise ValueError("Worker cleanup did not complete")


def validate_inputs(report: dict[str, Any]) -> None:
    """New sessions bind observations to the actual frozen inputs and SDK contract."""
    expected = copy.deepcopy(report["definition"]["runtime"])
    actual = copy.deepcopy(report["runtime_config"])
    expected["vllm"]["extra_args"] = ["--no-enable-prefix-caching"]
    if report["control_engine"] == "native-python":
        for field in ("data", "output"):
            actual["paths"][field] = expected["paths"][field]
    inputs = report["evidence_inputs"]
    if (
        expected != actual
        or report["sdk_packages"] != expected["packages"]
        or inputs["tasks_sha256"] != report["definition"]["pool"]["tasks_sha256"]
        or set(inputs["snapshots"]) != set(report["task_ids"])
        or set(inputs["sources"]) != set(report["task_ids"])
    ):
        raise ValueError("Session conditions differ from the frozen definition")


def observation(
    run: Any, task: str, variant: str, max_bytes: int, *, attempted: bool = False
) -> TaskObservation:
    if run is None:
        return {
            "task_id": task,
            "variant": variant,
            "run_id": None,
            "resolved": None,
            "duration_seconds": None,
            "status": "incomplete" if attempted else "not_run",
            "termination": "unknown",
            "implementation_changed": None,
            "tool_argument_errors": None,
            **unknown_behavior(),
        }
    row = run.rows.get(task)
    evidence = task_evidence(run, task, max_bytes)
    termination = termination_reason(run, task, evidence)
    return {
        **behavior(run, task, max_bytes),
        "task_id": task,
        "variant": variant,
        "run_id": run.manifest["run_id"],
        "resolved": row["resolved"] if row and run.complete else None,
        "duration_seconds": row.get("duration_seconds") if row else None,
        "status": "completed" if run.complete else "incomplete",
        "termination": termination,
        "implementation_changed": implementation_changed(run, task, max_bytes),
        "tool_argument_errors": len(evidence["tool_argument_errors"])
        if evidence["trace_error"] is None and not evidence["ambiguous_tool_observations"]
        else None,
    }


def summarize(path: Path, *, max_bytes: int) -> dict[str, Any]:
    session = json.loads(path.read_text())
    if session.get("kind") != "experiment-session":
        raise ValueError("Expected an agent experiment, not CPU verification")
    validate_worker_completion(path, session)
    planned = session["planned_order"]
    definition = session.get("definition")
    if definition is not None:
        if (
            content_hash(definition) != session.get("definition_sha256")
            or definition["experiment_id"] != session["experiment_id"]
            or definition["planned_order"] != planned
        ):
            raise ValueError("Session definition identity changed")
        if "evidence_inputs" in session:
            validate_inputs(session)
        candidate_files = session["candidate_files"]
        folders = {name: path.parent / "submissions" / name for name in session["variants"]}
        if "generation" in session and session["generation"] != {
            name: generation_contract(folder) for name, folder in folders.items()
        }:
            raise ValueError("Recorded generation differs from the evaluated candidates")
        if "change" in definition:
            if any(inventory(folder) != candidate_files[name] for name, folder in folders.items()):
                raise ValueError("Frozen candidate evidence changed")
            validate_change(ChangeScope.parse(definition["change"]), folders, candidate_files)
    tasks = list(dict.fromkeys(cell["task_id"] for cell in planned))
    variants = list(dict.fromkeys(cell["variant"] for cell in planned))
    if (
        len(variants) != VARIANTS_PER_PAIR
        or len(planned) != len(tasks) * len(variants)
        or len({(c["task_id"], c["variant"]) for c in planned}) != len(planned)
    ):
        raise ValueError("Invalid paired execution plan")
    if session["actual_order"] != planned[: len(session["actual_order"])]:
        raise ValueError("Actual order disagrees with the frozen plan")
    if "control_evidence" in session:
        for variant in variants:
            session_controls(path, session, variant)
    cells = {}
    run_ids: set[str] = set()
    for cell in session["cells"]:
        key = (cell["task_id"], cell["variant"])
        manifest = (path.parent / cell["manifest"]).resolve()
        if (
            not manifest.is_relative_to(path.parent.resolve())
            or key in cells
            or file_hash(manifest) != cell["sha256"]
            or key not in [(c["task_id"], c["variant"]) for c in session["actual_order"]]
        ):
            raise ValueError("Cell evidence changed or is duplicated")
        run = load_run(manifest.parent)
        if run.manifest["run_id"] in run_ids:
            raise ValueError("Duplicate cell run identity")
        run_ids.add(run.manifest["run_id"])
        if run.manifest["status"] == "completed":
            require_evaluation_evidence(run)
        if (
            run.task_ids != [cell["task_id"]]
            or run.manifest.get("source_sha256") != session["source_sha256"]
            or run.manifest.get("variant") != cell["variant"]
            or run.manifest.get("candidate_sha256") != session["variants"].get(cell["variant"])
        ):
            raise ValueError("Cell identity differs from session")
        if definition is not None and (
            run.manifest.get("experiment_id") != session["experiment_id"]
            or run.manifest.get("definition_sha256") != session["definition_sha256"]
            or run.manifest["submission_files"] != session["candidate_files"][cell["variant"]]
        ):
            raise ValueError("Cell experiment definition or files differ")
        if "evidence_inputs" in session:
            inputs = session["evidence_inputs"]
            task = cell["task_id"]
            manifest_inputs = {
                "tasks_sha256": inputs["tasks_sha256"],
                "snapshots": {task: inputs["snapshots"][task]},
                "source_setup": {task: inputs["sources"][task]},
                "runtime_config": session["runtime_config"],
                "evaluation": definition["evaluation"],
                **(
                    {"generation": session["generation"][cell["variant"]]}
                    if "generation" in session
                    else {}
                ),
                "verification_timeout_seconds": definition["runtime"][
                    "verification_timeout_seconds"
                ],
                "wheelhouse_hashes": definition["runtime"]["wheelhouse_hashes"],
            }
            if (
                any(run.manifest.get(k) != v for k, v in manifest_inputs.items())
                or any(
                    run.manifest.get("dependencies", {}).get(k) != v
                    for k, v in session["sdk_packages"].items()
                )
                or run.manifest.get("job_id") != session.get("job_id")
            ):
                raise ValueError("Cell inputs differ from the frozen session")
        if (
            "control_evidence" in session
            and run.manifest.get("controls") != session["control_evidence"]
        ):
            raise ValueError("Cell shared controls differ from session")
        cells[key] = run
    observations = {
        (task, variant): observation(
            cells.get((task, variant)),
            task,
            variant,
            max_bytes,
            attempted={"task_id": task, "variant": variant} in session["actual_order"],
        )
        for task in tasks
        for variant in variants
    }
    counts = dict(
        both=0, baseline_only=0, candidate_only=0, neither=0, errors=0, incomplete=0, not_run=0
    )
    rows = []
    for task in tasks:
        pair = [cells.get((task, v)) for v in variants]
        attempted = any(cell["task_id"] == task for cell in session["actual_order"])
        if not attempted:
            outcome = "not_run"
        elif not all(pair) or any(run is not None and not run.complete for run in pair):
            outcome = "incomplete"
        else:
            left, right = pair
            assert left is not None and right is not None
            differences = comparison_differences(left, right)
            if set(differences) - {"submission_files"}:
                raise ValueError(f"Paired execution conditions differ: {differences}")
            if any(
                observations[(task, variant)]["termination"] not in SCORABLE_TERMINATIONS
                for variant in variants
            ):
                outcome = "errors"
            else:
                a, b = (run.rows[task]["resolved"] for run in (left, right))
                outcome = (
                    "both"
                    if a and b
                    else "baseline_only"
                    if a
                    else "candidate_only"
                    if b
                    else "neither"
                )
        counts[outcome] += 1
        rows.append({"task_id": task, "outcome": outcome})
    return {
        "experiment_id": session["experiment_id"],
        "logging": load_health(path.parent),
        "status": session["status"],
        "variants": variants,
        "counts": counts,
        "tasks": rows,
        "formal_score": None,
        "cell_errors": sum(
            not run.complete or observations[key]["termination"] not in SCORABLE_TERMINATIONS
            for key, run in cells.items()
        ),
        "observations": list(observations.values()),
        "quota_hours": session.get("quota_hours"),
        "timings": session.get("timings", {}),
        "model_preparation": session.get("model_preparation"),
    }


def load_cpu(
    path: Path, digest: str, definition_hash: str, task_ids: list[str], variants: dict[str, str]
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("kind") != "experiment-verification"
        or report.get("status") != "completed"
        or report.get("source_sha256") != digest
        or report.get("definition_sha256") != definition_hash
        or report.get("task_ids") != task_ids
        or report.get("variants") != variants
    ):
        raise ValueError("CPU proof does not match the complete experiment bundle")
    if (
        report.get("cpu_worker_completion_required")
        and report.get("cpu_worker_completed") is not True
    ):
        raise ValueError("CPU worker cleanup did not complete")
    validate_worker_completion(path, report)
    if "evidence_inputs" in report:
        validate_inputs(report)
    for variant in variants:
        session_controls(path, report, variant)
    return dict(report)
