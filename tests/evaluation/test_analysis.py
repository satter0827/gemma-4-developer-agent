import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation import analysis
from gemma_agent_dev.evaluation.results import load_run

ROOT = Path(__file__).resolve().parents[2]


def test_analyze_separates_official_result_from_trace_claims_and_keeps_unknowns(
    make_run: Any, tmp_path: Path
) -> None:
    path = make_run()
    result_file = path / "task_results.jsonl"
    rows = [json.loads(line) for line in result_file.read_text().splitlines()]
    rows[0]["tool_calls"] = 4
    result_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    folder = path / "traces"
    folder.mkdir()
    trace = {
        "steps": [
            {"tool_calls": [{"function_name": "read_file", "arguments": {}}]},
            {
                "tool_calls": [{"function_name": "edit_file", "arguments": {}}],
                "observation": {"content": '{"status":"error"}'},
            },
            {
                "tool_calls": [
                    {
                        "function_name": "run_command",
                        "arguments": {"command": 'python -c "print(1)"'},
                    },
                    {
                        "function_name": "run_command",
                        "arguments": {"command": "pytest tests/test_a.py"},
                    },
                ],
                "observation": {"content": '{"stdout":"0 tests collected","exit_code":0}'},
            },
            {"tool_calls": [{"function_name": "submit_patch", "arguments": {}}]},
        ]
    }
    write_json(folder / "trace_task_a.json", trace)
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 100000)
    assert inventory(path) == before
    report = json.loads((output / "report.json").read_text())
    a = report["tasks"]["task_a"]
    b = report["tasks"]["task_b"]
    assert a["official_result"]["resolved"] is True and a["individual_test_counts"] is None
    assert a["first_edit_attempt_trace_call"] == 2 and len(a["edit_failures"]) == 1
    assert a["official_tool_calls"] == 4 and a["trace_tool_calls"] == 5
    assert b["official_tool_calls"] is None and b["trace_tool_calls"] is None
    assert (
        len(a["verification_commands"]) == 1
        and not a["verification_commands"][0]["establishes_resolution"]
    )
    assert b["time_limit_reached"] is None and b["trace_error"]


def test_intermediate_submission_never_replaces_the_final_official_patch(
    make_run: Any, tmp_path: Path
) -> None:
    path = make_run()
    result_file = path / "task_results.jsonl"
    rows = [json.loads(line) for line in result_file.read_text().splitlines()]
    rows[1]["agent_patch_size"] = 0
    result_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "traces").mkdir()
    write_json(
        path / "traces/trace_task_b.json",
        {
            "steps": [
                {
                    "tool_calls": [{"function_name": "submit_patch", "arguments": {}}],
                    "observation": {"content": json.dumps({"status": "ok", "patch_size": size})},
                }
                for size in (404, 0)
            ]
        },
    )
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 10000)
    evidence = json.loads((output / "report.json").read_text())["tasks"]["task_b"]
    assert [s["observation"]["patch_size"] for s in evidence["reported_patch_submissions"]] == [
        404,
        0,
    ]
    assert evidence["official_result"]["agent_patch_size"] == 0
    assert evidence["official_result"]["resolved"] is False and inventory(path) == before


@pytest.mark.parametrize("outcome", ["improved", "unchanged", "regressed", "invalid"])
def test_analyze_compares_only_main_instruction_under_identical_conditions(
    make_run: Any, tmp_path: Path, outcome: str
) -> None:
    a = make_run("a")
    b = make_run("b")
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["submission_files"]["prompts/system.md"] = "new"
    if outcome == "invalid":
        manifest["task_ids"].reverse()
    write_json(b / "manifest.json", manifest)
    if outcome in {"improved", "regressed"}:
        rows = [json.loads(line) for line in (b / "task_results.jsonl").read_text().splitlines()]
        for row in rows:
            row["resolved"] = outcome == "improved"
            row["test_exit_code"] = 0 if row["resolved"] else 1
        (b / "task_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(
            b / "summary.json", {"total_tasks": 2, "resolved": sum(r["resolved"] for r in rows)}
        )
    output = analysis.analyze(load_run(a), tmp_path / "analysis", 10000, load_run(b))
    assert json.loads((output / "report.json").read_text())["comparison"]["outcome"] == outcome


@pytest.mark.parametrize(
    "summary,expected",
    [
        ("6 failed, 18 passed in 1.24s", {"failed": 6, "passed": 18, "executed": 24}),
        ("1 skipped in 0.01s", {"skipped": 1, "executed": 0}),
        ("1 passed, 1 error in 0.01s", {"passed": 1, "errors": 1, "executed": None}),
        ("no tests ran in 0.01s", {"reported_total": 0, "executed": 0}),
        (
            "ERROR collecting tests/test_a.py\n1 error in 0.03s",
            {"collection_errors": 1, "errors": 0, "executed": 0},
        ),
        ("print-only: 24 passed", None),
    ],
)
def test_analysis_reads_reported_counts_without_replacing_official_grade(
    make_run: Any, tmp_path: Path, summary: str, expected: dict[str, int | None] | None
) -> None:
    path = make_run()
    folder = path / "test_outputs"
    folder.mkdir()
    (folder / "task_b.log").write_text(summary)
    (path / "logs").mkdir()
    (path / "logs/task_b.log").write_text("Agent execution evidence")
    (path / "patches").mkdir()
    (path / "patches/task_b.patch").write_text("--- a/example.py\n+++ b/example.py\n")
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 10000)
    report = json.loads((output / "report.json").read_text())
    evidence = report["tasks"]["task_b"]
    assert evidence["official_result"]["resolved"] is False
    counts = evidence["individual_test_counts"]
    if expected is None:
        assert counts is None
    else:
        assert {key: counts[key] for key in expected} == expected
    assert inventory(path) == before and str(folder / "task_b.log") in report["source_hashes"]
    for name in ("logs/task_b.log", "patches/task_b.patch"):
        assert report["source_hashes"][str(path / name)] == file_hash(path / name)


@pytest.mark.parametrize(
    "error,time_limit,tool_limit",
    [
        (None, None, None),
        ("Agent exceeded session timeout (1 min)", True, None),
        ("Agent exceeded tool call budget (10 calls)", None, True),
        ("Verification timed out after 300 seconds", None, None),
        ("Sandbox execution error: litellm.ContextWindowExceededError: input too long", None, None),
    ],
)
def test_analysis_does_not_misread_configured_limits_as_exhausted_budgets(
    make_run: Any, error: str | None, time_limit: bool | None, tool_limit: bool | None
) -> None:
    path = make_run()
    rows = [json.loads(line) for line in (path / "task_results.jsonl").read_text().splitlines()]
    rows[1]["error"] = error
    (path / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "traces").mkdir()
    write_json(
        path / "traces/trace_task_b.json",
        {
            "steps": [
                {
                    "tool_calls": [{"function_name": "get_status", "arguments": {}}],
                    "observation": {
                        "content": '{"timeout":60,"tool_call_limit":30,"max_tokens":16384}'
                    },
                }
            ]
        },
    )
    evidence = analysis.task_evidence(load_run(path), "task_b", 10000)
    assert (
        evidence["time_limit_reached"] is time_limit
        and evidence["tool_limit_reached"] is tool_limit
    )
    assert evidence["verification_timeout"] is (
        True if error and error.startswith("Verification") else None
    )
    assert evidence["context_window_exceeded"] is (
        True if error and "ContextWindowExceededError" in error else None
    )


def test_analysis_preserves_unknown_for_truncation_and_provider_timeouts(make_run: Any) -> None:
    assert analysis.reported_test_counts("1 passed in 0.01s\n[Display truncated]") is None
    path = make_run()
    rows = [json.loads(line) for line in (path / "task_results.jsonl").read_text().splitlines()]
    rows[1]["error"] = "Agent failed due to provider timeout"
    (path / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert analysis.task_evidence(load_run(path), "task_b", 10000)["time_limit_reached"] is None


def test_compare_displays_explicit_task_order(
    diagnostic_project: DiagnosticSettings,
    make_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gemma_agent_dev.cli import main

    root = diagnostic_project.root
    folder = root / "artifacts/results"
    folder.mkdir()
    for name in ["left", "right"]:
        shutil.copytree(make_run(name, ids=("task_z", "task_a")), folder / name)
    monkeypatch.chdir(root)
    assert main(["compare", "left", "right"]) == 0
    rows = [
        line.split("\t")[0]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("task_z\t", "task_a\t"))
    ]
    assert rows == ["task_z", "task_a"]


@pytest.mark.parametrize("damage", ["permuted", "missing", "duplicate"])
def test_analysis_compares_actual_sdk_task_order_not_only_selected_ids(
    make_run: Any, tmp_path: Path, damage: str
) -> None:
    left = make_run("left")
    right = make_run("right")
    manifest = json.loads((right / "manifest.json").read_text())
    manifest["submission_files"]["prompts/system.md"] = "changed"
    write_json(right / "manifest.json", manifest)
    rows = [json.loads(line) for line in (right / "task_results.jsonl").read_text().splitlines()]
    if damage == "permuted":
        rows[0]["task_index"], rows[1]["task_index"] = 2, 1
    elif damage == "missing":
        rows[0].pop("task_index")
    else:
        rows[1]["task_index"] = 1
    (right / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = analysis.analyze(load_run(left), tmp_path / "analysis", 10000, load_run(right))
    report = json.loads((output / "report.json").read_text())
    assert (
        report["comparison"]["outcome"] == "invalid"
        and report["comparison"]["task_order_verified"] is False
    )
    assert "actual_task_order" in report["comparison"]["differences"]


def test_scratch_files_and_argument_failures_are_not_source_repairs(make_run: Any) -> None:
    path = make_run()
    (path / "traces").mkdir()
    write_json(
        path / "traces/trace_task_a.json",
        {
            "steps": [
                {
                    "extra": {"author": "repair_agent"},
                    "tool_calls": [
                        {
                            "function_name": "write_file",
                            "arguments": {"file_path": "reproduce_bug.py"},
                        }
                    ],
                    "observation": {"content": '{"error":"missing required content argument"}'},
                },
                {
                    "tool_calls": [
                        {"function_name": "edit_file", "arguments": {"path": "rich/prompt.py"}}
                    ],
                    "observation": {"content": '{"status":"ok"}'},
                },
                {
                    "extra": {"author": "verify_agent"},
                    "tool_calls": [
                        {
                            "function_name": "run_command",
                            "arguments": {"command": "python repro.py"},
                        }
                    ],
                    "observation": {"content": '{"exit_code":0}'},
                },
            ]
        },
    )
    evidence = analysis.task_evidence(load_run(path), "task_a", 10000)
    assert evidence["first_edit_attempt_trace_call"] == 1
    assert evidence["first_implementation_edit_trace_call"] == 2
    assert evidence["scratch_edit_attempts"] == [1] and len(evidence["tool_argument_errors"]) == 1
    assert evidence["stage_authors"] == ["repair_agent", "verify_agent"]
    assert len(evidence["edit_failures"]) == 1 and len(evidence["verification_commands"]) == 1
