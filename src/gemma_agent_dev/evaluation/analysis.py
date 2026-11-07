"""Derive trace evidence without rewriting official evaluation results."""

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from gemma_agent_dev.common.artifacts import file_hash, new_run_id, write_json
from gemma_agent_dev.common.events import LABEL, context, event
from gemma_agent_dev.evaluation.results import Run, comparison_differences, read_artifact

EDIT_TOOLS = frozenset({"edit_file", "write_file"})
SCORABLE_TERMINATIONS = frozenset(
    {
        "completed",
        "time_limit_reached",
        "tool_limit_reached",
        "turn_limit_reached",
        "context_window_exceeded",
    }
)
LIMIT_PATTERNS = {
    "time_limit_reached": (
        r"\bagent (?:execution )?timed out\b|"
        r"\bagent\b.*\bexceeded (?:session|generation) (?:timeout|time budget)\b|"
        r"\b(?:session|generation) (?:budget|time limit) (?:exceeded|exhausted|reached)\b"
    ),
    "tool_limit_reached": (
        r"(?:exceeded|exhausted|reached).*tool.?call.*(?:budget|limit)|"
        r"tool.?call.*(?:budget|limit).*(?:exceeded|exhausted|reached)"
    ),
    "verification_timeout": r"\bverification\b.*(?:timeout|timed out)",
    "context_window_exceeded": r"\bContextWindowExceededError\b",
    "turn_limit_reached": (
        r"\bLlmCallsLimitExceededError\b|\bMax number of llm calls limit\b.*\bexceeded\b"
    ),
}


def termination_reason(run: Run, task: str, evidence: dict[str, Any]) -> str:
    """Separate generation limits from grading and execution failures."""
    return next(
        (
            key
            for key in (
                "context_window_exceeded",
                "time_limit_reached",
                "tool_limit_reached",
                "turn_limit_reached",
                "verification_timeout",
            )
            if evidence.get(key)
        ),
        "execution_error"
        if run.rows.get(task, {}).get("error")
        else "completed"
        if run.complete
        else "unknown",
    )


def emit_tool_events(run: Run, task_id: str, max_bytes: int) -> None:
    try:
        trace = json.loads(read_artifact(run, f"traces/trace_{task_id}.json", max_bytes))
        for step in trace["steps"]:
            calls = step.get("tool_calls", [])
            observation = step.get("observation", {}).get("content", "")
            try:
                observed = json.loads(observation) if len(calls) == 1 else {}
            except (ValueError, TypeError):
                observed = {}
            observed = observed if isinstance(observed, dict) else {}
            author = step.get("extra", {}).get("author")
            for call in calls:
                name = call.get("function_name")
                if not isinstance(name, str) or not LABEL.fullmatch(name):
                    continue
                error = observed.get("error") or observed.get("error_message")
                invalid = bool(
                    error and re.search(r"argument|parameter|required|validation", str(error), re.I)
                )
                outcome = (
                    "error"
                    if error or observed.get("status") == "error"
                    else "completed"
                    if observed.get("status") in {"success", "ok"} or observed.get("exit_code") == 0
                    else "unknown"
                )
                with context(task_id=task_id):
                    event(
                        "tool.observed",
                        tool_name=name,
                        stage=author
                        if isinstance(author, str) and LABEL.fullmatch(author)
                        else None,
                        duration_seconds=None,
                        result=outcome,
                        reason_code="argument_validation_error" if invalid else None,
                    )
    except (OSError, ValueError, KeyError, TypeError):
        event(
            "tool.evidence_unavailable",
            level="WARNING",
            task_id=task_id,
            reason_code="trace_unavailable",
        )


TEST_STATUSES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NON_IMPLEMENTATION_PATH = re.compile(
    r"(?:^|/)(?:tests?(?:/|_)|conftest\.py$|(?:repro|reproduce|scratch|tmp)[^/]*(?:/|$))"
    r"|(?:^|/)[^/]+_test\.py$|(?:^|/)(?:__pycache__|\.pytest_cache)(?:/|$)|\.pyc$"
)


def implementation_path(name: str) -> bool:
    return not NON_IMPLEMENTATION_PATH.search(name)


def reported_test_counts(text: str) -> dict[str, int | None] | None:
    """Read the last pytest summary as reported evidence, never as an official grade."""
    if "[Display truncated]" in text:
        return None
    text = ANSI_ESCAPE.sub("", text)
    for line in reversed(text.splitlines()):
        if not re.search(r"\bin \d+(?:\.\d+)?s\b", line):
            continue
        values = re.findall(r"\b(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b", line)
        if not values and "no tests ran" not in line:
            continue
        counts = {status: 0 for status in TEST_STATUSES}
        for number, label in values:
            counts["errors" if label == "error" else label] = int(number)
        collection = bool(re.search(r"ERROR collecting|errors? during collection", text))
        counts["collection_errors"] = counts["errors"] if collection else 0
        if collection:
            counts["errors"] = 0
        counts["reported_total"] = sum(counts[status] for status in TEST_STATUSES)
        # A passed or failed case can also have a teardown error in this summary.
        # Without case-level evidence those outcomes cannot be deduplicated.
        executed = None if counts["errors"] else counts["reported_total"] - counts["skipped"]
        return {**counts, "executed": executed}
    return None


def source_hashes(run: Run) -> dict[str, str]:
    paths = [run.path / name for name in ("manifest.json", "task_results.jsonl", "summary.json")]
    paths.extend(
        run.path / folder / name
        for task in run.task_ids
        for folder, name in (
            ("traces", f"trace_{task}.json"),
            ("test_outputs", f"{task}.log"),
            ("logs", f"{task}.log"),
            ("patches", f"{task}.patch"),
        )
    )
    hashes = {}
    for path in paths:
        if not path.resolve().is_relative_to(run.path.resolve()):
            raise ValueError("Analysis input escapes the run directory")
        if path.is_file():
            hashes[str(path)] = file_hash(path)
    return hashes


def task_evidence(run: Run, task_id: str, max_bytes: int) -> dict[str, Any]:
    row = run.rows.get(task_id)
    result: dict[str, Any] = {
        "official_result": row,
        "individual_test_counts": None,
        "official_tool_calls": row.get("tool_calls") if row else None,
        "first_edit_attempt_trace_call": None,
        "first_implementation_edit_seconds": None,
        "status_checks": [],
        "skill_verifications": [],
        "first_implementation_edit_trace_call": None,
        "scratch_edit_attempts": [],
        "unknown_edit_targets": [],
        "tool_argument_errors": [],
        "stage_authors": [],
        "reported_patch_submissions": [],
        "edit_failures": [],
        "verification_commands": [],
        "trace_error": None,
        "test_log_error": None,
        "ambiguous_tool_observations": False,
    }
    error_text = str(row.get("error") or "") if row else ""
    try:
        relative = f"test_outputs/{task_id}.log"
        counts = reported_test_counts(read_artifact(run, relative, max_bytes))
        result["individual_test_counts"] = counts
        result["test_counts_source"] = relative if counts is not None else None
    except OSError as error:
        result["test_log_error"] = str(error)
    try:
        trace = json.loads(read_artifact(run, f"traces/trace_{task_id}.json", max_bytes))
        calls = 0
        for step in trace["steps"]:
            observation = step.get("observation", {}).get("content", "")
            try:
                observed = json.loads(observation)
            except (ValueError, TypeError):
                observed = {}
            if not isinstance(observed, dict):
                observed = {}
            step_calls = step.get("tool_calls", [])
            if len(step_calls) > 1:
                result["ambiguous_tool_observations"] = True
                observed = {}
            author = step.get("extra", {}).get("author")
            if author and author not in result["stage_authors"]:
                result["stage_authors"].append(author)
            for call in step_calls:
                calls += 1
                name = call["function_name"]
                if name == "get_status":
                    result["status_checks"].append(
                        {
                            "trace_call": calls,
                            "elapsed_seconds": step.get("extra", {}).get("elapsed_s"),
                            "observation": observed,
                        }
                    )
                if name == "run_skill_script":
                    result["skill_verifications"].append(
                        {
                            "trace_call": calls,
                            "elapsed_seconds": step.get("extra", {}).get("elapsed_s"),
                            "observation": observed,
                        }
                    )
                if name == "submit_patch":
                    result["reported_patch_submissions"].append(
                        {"trace_call": calls, "observation": observed}
                    )
                if name in EDIT_TOOLS:
                    if result["first_edit_attempt_trace_call"] is None:
                        result["first_edit_attempt_trace_call"] = calls
                    arguments = call.get("arguments", {})
                    arguments = arguments if isinstance(arguments, dict) else {}
                    target = arguments.get(
                        "filepath", arguments.get("file_path", arguments.get("path"))
                    )
                    if not isinstance(target, str):
                        result["unknown_edit_targets"].append(calls)
                    elif re.search(r"(?:^|/)(?:repro|reproduce|scratch|tmp)[^/]*\.py$", target):
                        result["scratch_edit_attempts"].append(calls)
                    elif implementation_path(target):
                        if result["first_implementation_edit_trace_call"] is None:
                            result["first_implementation_edit_trace_call"] = calls
                            result["first_implementation_edit_seconds"] = step.get("extra", {}).get(
                                "elapsed_s"
                            )
                    else:
                        result["unknown_edit_targets"].append(calls)
                    if observed.get("status") == "error" or observed.get("error"):
                        result["edit_failures"].append({"call": calls, "observation": observed})
                tool_error = observed.get("error") or observed.get("error_message")
                if tool_error and re.search(
                    r"argument|parameter|required|validation", str(tool_error), re.I
                ):
                    result["tool_argument_errors"].append(
                        {"call": calls, "tool": name, "observation": observed}
                    )
                arguments = call.get("arguments", {})
                command = arguments.get("command") if isinstance(arguments, dict) else None
                if (
                    name == "run_command"
                    and isinstance(command, str)
                    and re.search(
                        r"\bpytest\b|\bunittest\b|\bassert\b|\bpython[\d.]*\s+\S+\.py\b", command
                    )
                ):
                    result["verification_commands"].append(
                        {
                            "call": calls,
                            "command": command,
                            "observation": observed,
                            "establishes_resolution": False,
                        }
                    )
        result["trace_tool_calls"] = calls
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["trace_error"] = str(error)
        result["trace_tool_calls"] = None
    for key, pattern in LIMIT_PATTERNS.items():
        result[key] = True if re.search(pattern, error_text, re.I) else None
    return result


class Behavior(TypedDict):
    submission_count: int | None
    nonempty_submission_then_empty_final: bool | None
    first_implementation_edit_trace_call: int | None
    skill_script_calls: int | None


def unknown_behavior() -> Behavior:
    return {
        "submission_count": None,
        "nonempty_submission_then_empty_final": None,
        "first_implementation_edit_trace_call": None,
        "skill_script_calls": None,
    }


BEHAVIOR_FIELDS = frozenset(
    {
        "submission_count",
        "nonempty_submission_then_empty_final",
        "first_implementation_edit_trace_call",
        "skill_script_calls",
    }
)


def behavior(run: Run, task: str, max_bytes: int) -> Behavior:
    """Export only observable, bounded behavior; absent traces remain unknown."""
    evidence = task_evidence(run, task, max_bytes)
    if evidence["trace_error"] is not None or evidence["ambiguous_tool_observations"]:
        return unknown_behavior()
    submissions = evidence["reported_patch_submissions"]
    row = run.rows.get(task, {})
    final_size = row.get("agent_patch_size")
    nonempty = any(
        type(item["observation"].get("patch_size")) is int and item["observation"]["patch_size"] > 0
        for item in submissions
    )
    unknown_submission = any(
        type(item["observation"].get("patch_size")) is not int for item in submissions
    )
    trace = json.loads(read_artifact(run, f"traces/trace_{task}.json", max_bytes))
    return {
        "submission_count": len(submissions),
        "nonempty_submission_then_empty_final": (nonempty and final_size == 0)
        if type(final_size) is int and (final_size != 0 or nonempty or not unknown_submission)
        else None,
        "first_implementation_edit_trace_call": evidence["first_implementation_edit_trace_call"],
        "skill_script_calls": sum(
            call["function_name"] == "run_skill_script"
            for step in trace["steps"]
            for call in step.get("tool_calls", [])
        ),
    }


def actual_task_order(run: Run) -> list[str] | None:
    """Use the official task indices; selected IDs do not define SDK execution order."""
    if run.kind != "evaluation" or not run.complete:
        return None
    rows = list(run.rows.values())
    if any(type(row.get("task_index")) is not int for row in rows):
        return None
    ordered = sorted(rows, key=lambda row: row["task_index"])
    if [row["task_index"] for row in ordered] != list(range(1, len(run.task_ids) + 1)):
        return None
    return [row["instance_id"] for row in ordered]


def implementation_changed(run: Run, task: str, max_bytes: int) -> bool | None:
    """Inspect retained patch bytes rather than treating edit attempts as source changes."""
    relative = f"patches/{task}.patch"
    path = run.path / relative
    if not path.is_file() or path.stat().st_size > max_bytes:
        return None
    text = read_artifact(run, relative, max_bytes)
    names = re.findall(r"^(?:\+\+\+ b/|--- a/)(.+)$", text, re.M)
    if not names:
        return None
    return any(implementation_path(name) for name in names)


def analyze(run: Run, output_root: Path, max_bytes: int, other: Run | None = None) -> Path:
    if run.kind != "evaluation":
        raise ValueError("Diagnostics and controls are not agent evaluations")
    destination = output_root / new_run_id()
    destination.mkdir(parents=True)
    report: dict[str, Any] = {
        "kind": "analysis",
        "run_id": run.manifest["run_id"],
        "run_complete": run.complete,
        "actual_task_order": actual_task_order(run),
        "problems": list(run.problems),
        "source_hashes": source_hashes(run),
        "phase": run.manifest.get("phase"),
        "run_error": run.manifest.get("error"),
        "cleanup_error": run.manifest.get("cleanup_error"),
        "tasks": {task: task_evidence(run, task, max_bytes) for task in run.task_ids},
        "formal_score": None,
    }
    if other is not None:
        differences = comparison_differences(run, other)
        changed = sorted(
            k
            for k in run.manifest.get("submission_files", {}).keys()
            | other.manifest.get("submission_files", {}).keys()
            if run.manifest.get("submission_files", {}).get(k)
            != other.manifest.get("submission_files", {}).get(k)
        )
        left_order, right_order = actual_task_order(run), actual_task_order(other)
        order_verified = left_order is not None and left_order == right_order
        if not order_verified:
            differences.append("actual_task_order")
        comparable = not [d for d in differences if d != "submission_files"] and changed == [
            "prompts/system.md"
        ]
        report["comparison"] = {
            "other_run_id": other.manifest["run_id"],
            "other_task_order": right_order,
            "task_order_verified": order_verified,
            "other_source_hashes": source_hashes(other),
            "differences": differences,
            "changed_files": changed,
            "comparable_workflow_trial": comparable,
            "outcome": "invalid"
            if not comparable
            else (
                "improved"
                if other.resolved > run.resolved
                else "regressed"
                if other.resolved < run.resolved
                else "unchanged"
            ),
            "resolved": [run.resolved, other.resolved],
            "scope": "development tasks; no formal-score prediction",
        }
    write_json(destination / "report.json", report)
    return destination
