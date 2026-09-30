import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gemma_agent_dev.results import comparison_differences, load_run, read_artifact


def test_complete_and_partial_results_have_different_score_semantics(
    make_run: Callable[..., Path],
) -> None:
    complete = load_run(make_run())
    partial = load_run(make_run("partial", completed=1, status="running"))
    assert complete.resolution_rate == 0.5
    assert partial.resolved == 1 and len(partial.rows) == 1
    assert partial.resolution_rate is None
    assert "incomplete" in comparison_differences(complete, partial)


@pytest.mark.parametrize("damage", ["duplicate", "truncated", "missing", "unknown", "invalid"])
def test_corrupt_records_cannot_produce_a_comparable_score(
    make_run: Callable[..., Path],
    damage: str,
) -> None:
    path = make_run()
    records = path / "task_results.jsonl"
    first = records.read_text().splitlines()[0]
    if damage == "duplicate":
        records.write_text(records.read_text() + first + "\n")
    elif damage == "truncated":
        records.write_text(records.read_text() + '{"instance_id":')
    elif damage == "missing":
        records.unlink()
    elif damage == "unknown":
        records.write_text(records.read_text().replace("task_a", "task_c"))
    else:
        records.write_text(records.read_text().replace('"resolved": true', '"resolved": "true"'))
    result = load_run(path)
    assert result.problems
    assert result.resolution_rate is None
    if damage == "duplicate":
        assert "task_a" not in result.rows


def test_summary_mismatch_is_not_ignored(make_run: Callable[..., Path]) -> None:
    path = make_run()
    (path / "summary.json").write_text('{"total_tasks": 2, "resolved": 2}')
    assert not load_run(path).complete


def test_comparison_checks_tasks_and_conditions(make_run: Callable[..., Path]) -> None:
    a = load_run(make_run())
    b_path = make_run("run-b")
    assert comparison_differences(a, load_run(b_path)) == []
    manifest = json.loads((b_path / "manifest.json").read_text())
    manifest["evaluation"]["max_time_minutes"] = 5
    (b_path / "manifest.json").write_text(json.dumps(manifest))
    assert comparison_differences(a, load_run(b_path)) == ["evaluation"]
    c = load_run(make_run("run-c", ids=("task_a", "task_c")))
    assert "task_ids" in comparison_differences(a, c)


def test_artifacts_cannot_escape_result_directory(make_run: Callable[..., Path]) -> None:
    run = load_run(make_run())
    with pytest.raises(ValueError, match="escapes"):
        read_artifact(run, "../secret.txt")
