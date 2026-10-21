import json
from collections.abc import Callable
from pathlib import Path

import pytest

from gemma_agent_dev.evaluation.results import comparison_differences, load_run, read_artifact


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
        read_artifact(run, "../secret.txt", 100)


def test_pre_evaluation_failure_is_readable_without_official_results(
    make_run: Callable[..., Path],
) -> None:
    path = make_run(status="failed", completed=0)
    (path / "task_results.jsonl").unlink()
    (path / "summary.json").unlink()
    result = load_run(path)
    assert not result.problems and not result.complete and result.resolution_rate is None
    assert load_run(path, require_evaluation=True).problems


def test_reference_verification_has_no_agent_score(make_run: Callable[..., Path]) -> None:
    path = make_run()
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(kind="verification", controls_passed=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "task_results.jsonl").unlink()
    (path / "summary.json").unlink()
    controls = [
        dict(
            instance_id=task,
            unmodified=dict(resolved=False, exit_code=1, duration_seconds=1, error=None),
            reference=dict(resolved=True, exit_code=0, duration_seconds=1, error=None),
        )
        for task in manifest["task_ids"]
    ]
    (path / "controls.json").write_text(json.dumps(controls))
    result = load_run(path)
    assert result.complete and result.completed_count == 2 and result.resolution_rate is None
    with pytest.raises(ValueError, match="not an agent"):
        comparison_differences(result, result)
    (path / "controls.json").write_text(json.dumps(controls + controls[:1]))
    assert load_run(path).problems


@pytest.mark.parametrize("field", ["verification_timeout_seconds", "vllm_config"])
def test_comparison_checks_verification_time_and_applied_server_defaults(
    make_run: Callable[..., Path], field: str
) -> None:
    left = load_run(make_run())
    path = make_run("run-b")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest[field] = 600 if field == "verification_timeout_seconds" else {"max_model_len": 16384}
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert comparison_differences(left, load_run(path)) == [field]


@pytest.mark.parametrize(
    "member", ["agent.yaml", "prompts/system.md", "adapters/main/adapter_config.json"]
)
def test_changed_agent_or_adapter_is_a_different_comparison_condition(
    make_run: Callable[..., Path], member: str
) -> None:
    left = load_run(make_run())
    directory = make_run("run-b")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["submission_files"][member] = "changed"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert comparison_differences(left, load_run(directory)) == ["submission_files"]


def test_missing_submission_inventory_cannot_establish_equivalence(
    make_run: Callable[..., Path],
) -> None:
    directory = make_run()
    manifest = json.loads((directory / "manifest.json").read_text())
    del manifest["submission_files"]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    run = load_run(directory)
    assert comparison_differences(run, run) == ["submission_files"]


def test_invalid_submission_inventory_is_a_read_error(make_run: Callable[..., Path]) -> None:
    directory = make_run()
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["submission_files"] = []
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="submission_files"):
        load_run(directory)
