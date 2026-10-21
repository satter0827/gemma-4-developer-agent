"""Derive trace evidence without rewriting official evaluation results."""

import json
import re
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash, new_run_id, write_json
from gemma_agent_dev.results import Run, comparison_differences, read_artifact

EDIT_TOOLS = frozenset({"edit_file", "write_file"})
LIMIT_PATTERNS = {
    "time_limit_reached": r"timed? out|time.?budget.*(?:exceed|exhaust)|timeout",
    "tool_limit_reached": r"tool.?call.*(?:limit|exceed|exhaust)",
}


def task_evidence(run: Run, task_id: str, max_bytes: int) -> dict[str, Any]:
    row = run.rows.get(task_id)
    result: dict[str, Any] = {
        "official_result": row,
        "individual_test_counts": None,
        "first_edit_call": None,
        "edit_failures": [],
        "verification_commands": [],
        "trace_error": None,
    }
    text = str(row.get("error") or "") if row else ""
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
            for call in step.get("tool_calls", []):
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
            text += "\n" + str(observation)
        result["tool_calls"] = calls
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["trace_error"] = str(error)
        result["tool_calls"] = None
    for key, pattern in LIMIT_PATTERNS.items():
        result[key] = True if re.search(pattern, text, re.I) else None
    return result


def analyze(run: Run, output_root: Path, max_bytes: int, other: Run | None = None) -> Path:
    if run.kind != "evaluation":
        raise ValueError("Diagnostics and controls are not agent evaluations")
    destination = output_root / new_run_id()
    destination.mkdir(parents=True)
    sources = {
        str(path): file_hash(path)
        for path in run.path.rglob("*")
        if path.is_file() and path.name in {"manifest.json", "task_results.jsonl", "summary.json"}
    }
    report: dict[str, Any] = {
        "kind": "analysis",
        "run_id": run.manifest["run_id"],
        "run_complete": run.complete,
        "problems": list(run.problems),
        "source_hashes": sources,
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
        comparable = not [d for d in differences if d != "submission_files"] and changed == [
            "prompts/system.md"
        ]
        report["comparison"] = {
            "other_run_id": other.manifest["run_id"],
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
