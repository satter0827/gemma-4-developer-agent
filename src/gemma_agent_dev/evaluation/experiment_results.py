"""Compare complete task pairs without turning missing evidence into failures."""

import json
from pathlib import Path
from typing import Any, TypedDict

from gemma_agent_dev.candidates.changes import ChangeScope, validate_change
from gemma_agent_dev.common.artifacts import file_hash, inventory
from gemma_agent_dev.evaluation.analysis import implementation_changed, task_evidence
from gemma_agent_dev.evaluation.experiments import content_hash
from gemma_agent_dev.evaluation.results import (
    comparison_differences,
    load_run,
    require_evaluation_evidence,
)
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR


class TaskObservation(TypedDict):
    task_id: str
    variant: str
    run_id: str | None
    resolved: bool | None
    duration_seconds: float | None
    status: str
    termination: str
    implementation_changed: bool | None
    tool_argument_errors: int | None


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
        }
    row = run.rows.get(task)
    evidence = task_evidence(run, task, max_bytes)
    termination = next(
        (
            key
            for key in (
                "context_window_exceeded",
                "time_limit_reached",
                "tool_limit_reached",
                "verification_timeout",
            )
            if evidence.get(key)
        ),
        "execution_error"
        if row and row.get("error")
        else "completed"
        if run.complete
        else "unknown",
    )
    return {
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
    planned = session["planned_order"]
    definition = session.get("definition")
    if definition is not None:
        if (
            content_hash(definition) != session.get("definition_sha256")
            or definition["experiment_id"] != session["experiment_id"]
            or definition["planned_order"] != planned
        ):
            raise ValueError("Session definition identity changed")
        candidate_files = session["candidate_files"]
        folders = {name: path.parent / "submissions" / name for name in session["variants"]}
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
        cells[key] = run
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
            if any(run.rows[task].get("error") for run in (left, right)):
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
        "status": session["status"],
        "variants": variants,
        "counts": counts,
        "tasks": rows,
        "formal_score": None,
        "cell_errors": sum(
            not run.complete or bool(run.rows.get(task, {}).get("error"))
            for (task, _), run in cells.items()
        ),
        "observations": [
            observation(
                cells.get((task, variant)),
                task,
                variant,
                max_bytes,
                attempted={"task_id": task, "variant": variant} in session["actual_order"],
            )
            for task in tasks
            for variant in variants
        ],
        "quota_hours": session.get("quota_hours"),
    }
