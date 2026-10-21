"""Compare complete task pairs without turning missing evidence into failures."""

import json
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.evaluation.results import (
    comparison_differences,
    load_run,
    require_evaluation_evidence,
)
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR


def summarize(path: Path) -> dict[str, Any]:
    session = json.loads(path.read_text())
    if session.get("kind") != "experiment-session":
        raise ValueError("Expected an agent experiment, not CPU verification")
    planned = session["planned_order"]
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
        if run.manifest["status"] == "completed":
            require_evaluation_evidence(run)
        if (
            run.task_ids != [cell["task_id"]]
            or run.manifest.get("source_sha256") != session["source_sha256"]
            or run.manifest.get("variant") != cell["variant"]
            or run.manifest.get("candidate_sha256") != session["variants"].get(cell["variant"])
        ):
            raise ValueError("Cell identity differs from session")
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
    }
