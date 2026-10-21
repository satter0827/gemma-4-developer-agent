import json
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.evaluation.experiment_results import summarize


@pytest.fixture
def paired_run(make_run: Any, tmp_path: Path) -> Path:
    planned = [{"task_id": "task_z", "variant": "base"}, {"task_id": "task_z", "variant": "new"}]
    cells = []
    for index, variant in enumerate(["base", "new"]):
        folder = make_run(variant, ids=("task_z",), completed=1)
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.update(
            variant=variant,
            source_sha256="bundle",
            candidate_sha256=variant,
            evaluation_artifacts={
                name: file_hash(folder / name) for name in ["summary.json", "task_results.jsonl"]
            },
        )
        write_json(folder / "manifest.json", manifest)
        cells.append(
            {
                **planned[index],
                "manifest": variant + "/manifest.json",
                "sha256": file_hash(folder / "manifest.json"),
            }
        )
    p = tmp_path / "session.json"
    write_json(
        p,
        {
            "kind": "experiment-session",
            "experiment_id": "trial",
            "status": "completed",
            "source_sha256": "bundle",
            "variants": {"base": "base", "new": "new"},
            "planned_order": planned,
            "actual_order": planned,
            "cells": cells,
        },
    )
    return p


def update_outcome(
    session: Path, variant: str, *, resolved: bool, error: str | None = None
) -> None:
    folder = session.parent / variant
    row = json.loads((folder / "task_results.jsonl").read_text())
    row.update(resolved=resolved, test_exit_code=0 if resolved else 1, error=error)
    (folder / "task_results.jsonl").write_text(json.dumps(row) + "\n")
    write_json(folder / "summary.json", {"total_tasks": 1, "resolved": int(resolved)})
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["evaluation_artifacts"] = {
        name: file_hash(folder / name) for name in ["summary.json", "task_results.jsonl"]
    }
    write_json(folder / "manifest.json", manifest)
    data = json.loads(session.read_text())
    for cell in data["cells"]:
        if cell["variant"] == variant:
            cell["sha256"] = file_hash(folder / "manifest.json")
    write_json(session, data)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (True, True, "both"),
        (True, False, "baseline_only"),
        (False, True, "candidate_only"),
        (False, False, "neither"),
    ],
)
def test_only_matching_task_pairs_compared(
    paired_run: Path, left: bool, right: bool, expected: str
) -> None:
    update_outcome(paired_run, "base", resolved=left)
    update_outcome(paired_run, "new", resolved=right)
    summary = summarize(paired_run, max_bytes=2000000)
    assert summary["counts"][expected] == 1 and sum(summary["counts"].values()) == 1


def test_errors_partial_and_not_run_are_separate(paired_run: Path) -> None:
    update_outcome(paired_run, "new", resolved=False, error="tool execution failed")
    summary = summarize(paired_run, max_bytes=2000000)
    assert summary["counts"]["errors"] == 1 and summary["cell_errors"] == 1
    data = json.loads(paired_run.read_text())
    data["cells"] = data["cells"][:1]
    data["actual_order"] = data["actual_order"][:1]
    write_json(paired_run, data)
    assert summarize(paired_run, max_bytes=2000000)["counts"]["incomplete"] == 1
    data["cells"] = []
    data["actual_order"] = []
    write_json(paired_run, data)
    assert summarize(paired_run, max_bytes=2000000)["counts"]["not_run"] == 1


@pytest.mark.parametrize(
    "damage", ["kind", "plan", "order", "duplicate", "hash", "identity", "conditions", "result"]
)
def test_changed_or_incomparable_evidence_is_rejected(paired_run: Path, damage: str) -> None:
    data = json.loads(paired_run.read_text())
    if damage == "kind":
        data["kind"] = "verification"
    elif damage == "plan":
        data["planned_order"] *= 2
    elif damage == "order":
        data["actual_order"].reverse()
    elif damage == "duplicate":
        data["cells"] *= 2
    elif damage == "hash":
        data["cells"][0]["sha256"] = "changed"
    elif damage == "result":
        paired_run.parent.joinpath("base/summary.json").write_text("{}")
    else:
        p = paired_run.parent / "new/manifest.json"
        m = json.loads(p.read_text())
        if damage == "identity":
            m["variant"] = "other"
        else:
            m["evaluation"] = {"max_time_minutes": 2}
        write_json(p, m)
        data["cells"][1]["sha256"] = file_hash(p)
    write_json(paired_run, data)
    with pytest.raises((ValueError, KeyError)):
        summarize(paired_run, max_bytes=2000000)


@pytest.mark.parametrize(
    "error,termination,invalid",
    [
        ("Agent exceeded session timeout (5 min)", "time_limit_reached", False),
        ("Agent exceeded tool call budget (30 calls)", "tool_limit_reached", False),
        ("ContextWindowExceededError: limit reached", "context_window_exceeded", False),
        ("Max number of llm calls limit of `50` exceeded", "turn_limit_reached", False),
        ("Verification timed out", "verification_timeout", True),
        ("Sandbox execution error", "execution_error", True),
    ],
)
def test_generation_limits_remain_official_unsolved_results(
    paired_run: Path, error: str, termination: str, invalid: bool
) -> None:
    update_outcome(paired_run, "base", resolved=False, error=error)
    summary = summarize(paired_run, max_bytes=2000000)
    assert summary["observations"][0]["termination"] == termination
    assert summary["observations"][0]["resolved"] is False
    assert summary["counts"]["errors" if invalid else "candidate_only"] == 1
    assert summary["cell_errors"] == int(invalid)
