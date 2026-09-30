"""Read official artifacts without treating missing or corrupt results as success."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    problems: tuple[str, ...]

    @property
    def task_ids(self) -> list[str]:
        return list(self.manifest["task_ids"])

    @property
    def resolved(self) -> int:
        return sum(row["resolved"] for row in self.rows.values())

    @property
    def complete(self) -> bool:
        return (
            self.manifest["status"] == "completed"
            and set(self.rows) == set(self.task_ids)
            and not self.problems
        )

    @property
    def resolution_rate(self) -> float | None:
        return self.resolved / len(self.task_ids) if self.complete else None


def validate_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
        raise ValueError("Missing task ID")
    if type(row.get("resolved")) is not bool:
        raise ValueError("Resolved must be a boolean")
    for name in ("duration_seconds", "agent_patch_size"):
        value = row.get(name)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid {name}")
    if type(row.get("test_exit_code")) is not int:
        raise ValueError("Invalid test exit code")
    if row["resolved"] and (row["test_exit_code"] != 0 or row.get("error")):
        raise ValueError("Resolved result conflicts with verification error")
    return row


def load_run(path: Path) -> Run:
    manifest = read_object(path / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    ids = manifest.get("task_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid or duplicate planned task IDs")
    if manifest.get("status") not in {"running", "completed", "failed"}:
        raise ValueError("Invalid run status")
    for field in ("evaluation", "environment", "runtime_config", "timings"):
        if field in manifest and not isinstance(manifest[field], dict):
            raise ValueError(f"Invalid manifest mapping: {field}")
    rows: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    problems: list[str] = []
    result_file = path / "task_results.jsonl"
    if not result_file.is_file():
        problems.append("Missing task_results.jsonl")
    else:
        for line_number, line in enumerate(result_file.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = validate_row(json.loads(line))
                task_id = row["instance_id"]
                if task_id not in ids:
                    raise ValueError(f"Unplanned task: {task_id}")
                if task_id in seen:
                    rows.pop(task_id, None)
                    raise ValueError(f"Duplicate task: {task_id}")
                seen.add(task_id)
                rows[task_id] = row
            except (ValueError, TypeError) as error:
                problems.append(f"Line {line_number}: {error}")
    try:
        summary = read_object(path / "summary.json")
        if summary.get("total_tasks") != len(rows) or summary.get("resolved") != sum(
            row["resolved"] for row in rows.values()
        ):
            problems.append("Summary disagrees with task results")
    except (OSError, ValueError) as error:
        problems.append(f"Cannot read summary: {error}")
    if manifest["status"] == "completed" and set(rows) != set(ids):
        problems.append("Completed run has missing task results")
    return Run(path, manifest, rows, tuple(problems))


def comparison_differences(left: Run, right: Run) -> list[str]:
    differences = []
    if set(left.task_ids) != set(right.task_ids):
        differences.append("task_ids")
    for field in ("evaluation", "environment", "tasks_sha256"):
        if field not in left.manifest or field not in right.manifest:
            differences.append(field)
        elif left.manifest[field] != right.manifest[field]:
            differences.append(field)
    for field in ("model_source", "docker_image", "vllm", "context_cache", "compaction"):
        a = left.manifest.get("runtime_config", {})
        b = right.manifest.get("runtime_config", {})
        if field not in a or field not in b or a[field] != b[field]:
            differences.append(field)
    if not left.complete or not right.complete:
        differences.append("incomplete")
    return differences


def read_artifact(run: Run, relative: str, max_bytes: int = 2_000_000) -> str:
    path = (run.path / relative).resolve()
    if not path.is_relative_to(run.path.resolve()):
        raise ValueError("Artifact path escapes the run directory")
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    text = content[:max_bytes].decode("utf-8", errors="replace")
    return text + ("\n[Display truncated]" if len(content) > max_bytes else "")
