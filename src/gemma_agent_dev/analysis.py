"""Derive trace evidence without rewriting official evaluation results."""

import json
import re
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash, new_run_id, write_json
from gemma_agent_dev.results import Run, comparison_differences, read_artifact

EDIT_TOOLS = frozenset({"edit_file", "write_file"})
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
}
TEST_STATUSES = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def reported_test_counts(text: str) -> dict[str, int] | None:
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
        counts["executed"] = counts["reported_total"] - counts["skipped"]
        return counts
    return None


def source_hashes(run: Run) -> dict[str, str]:
    paths = [run.path / name for name in ("manifest.json", "task_results.jsonl", "summary.json")]
    paths.extend(
        run.path / folder / name
        for task in run.task_ids
        for folder, name in (("traces", f"trace_{task}.json"), ("test_outputs", f"{task}.log"))
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
        "first_edit_call": None,
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
            for call in step_calls:
                calls += 1
                name = call["function_name"]
                if name in EDIT_TOOLS:
                    if result["first_edit_call"] is None:
                        result["first_edit_call"] = calls
                    if observed.get("status") == "error":
                        result["edit_failures"].append({"call": calls, "observation": observed})
                command = call.get("arguments", {}).get("command")
                if (
                    name == "run_command"
                    and isinstance(command, str)
                    and re.search(r"\bpytest\b|\bunittest\b|\bassert\b", command)
                ):
                    result["verification_commands"].append(
                        {
                            "call": calls,
                            "command": command,
                            "observation": observed,
                            "establishes_resolution": False,
                        }
                    )
        result["tool_calls"] = calls
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["trace_error"] = str(error)
        result["tool_calls"] = None
    for key, pattern in LIMIT_PATTERNS.items():
        result[key] = True if re.search(pattern, error_text, re.I) else None
    return result


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
