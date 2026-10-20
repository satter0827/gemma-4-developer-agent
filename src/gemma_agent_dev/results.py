"""Read official artifacts without treating missing or corrupt results as success."""

import json
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash


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
    controls: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)

    @property
    def kind(self) -> str:
        return str(self.manifest.get("kind", "evaluation"))

    @property
    def completed_count(self) -> int:
        return len(self.controls) if self.kind == "verification" else len(self.rows)

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
            and set(self.controls if self.kind == "verification" else self.rows)
            == set(self.task_ids)
            and not self.problems
        )

    @property
    def resolution_rate(self) -> float | None:
        return (
            self.resolved / len(self.task_ids)
            if self.kind == "evaluation" and self.complete
            else None
        )


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


def read_controls(path: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected a control result list")
    controls: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid control record")
        task_id = record.get("instance_id")
        if not isinstance(task_id, str) or task_id not in ids or task_id in controls:
            raise ValueError("Unknown or duplicate control task")
        for label in ("unmodified", "reference"):
            outcome = record.get(label)
            if not isinstance(outcome, dict) or type(outcome.get("resolved")) is not bool:
                raise ValueError("Invalid control outcome")
            duration = outcome.get("duration_seconds")
            if type(outcome.get("exit_code")) is not int or (
                not isinstance(duration, int | float)
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError("Invalid control timing or exit code")
            if outcome["resolved"] and (outcome["exit_code"] != 0 or outcome.get("error")):
                raise ValueError("Control resolution conflicts with an error")
        controls[task_id] = record
    return controls


def load_run(path: Path, *, require_evaluation: bool = False) -> Run:
    manifest = read_object(path / "manifest.json")
    if manifest.get("kind", "evaluation") not in {"evaluation", "verification"}:
        raise ValueError("Invalid run kind")
    if require_evaluation and manifest.get("kind", "evaluation") != "evaluation":
        raise ValueError("Expected an evaluation run")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("Invalid run ID")
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
    for field in ("evaluation", "environment", "runtime_config", "timings", "submission_files"):
        if field in manifest and not isinstance(manifest[field], dict):
            raise ValueError(f"Invalid manifest mapping: {field}")
    rows: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    problems: list[str] = []
    if manifest.get("kind") == "verification":
        controls: dict[str, dict[str, Any]] = {}
        control_file = path / "controls.json"
        try:
            if control_file.exists() or manifest["status"] == "completed":
                controls = read_controls(control_file, ids)
            if manifest["status"] == "completed" and (
                set(controls) != set(ids)
                or manifest.get("controls_passed") is not True
                or any(
                    row["unmodified"]["resolved"] or not row["reference"]["resolved"]
                    for row in controls.values()
                )
            ):
                problems.append("Completed verification has missing or failed controls")
        except (OSError, ValueError, TypeError) as error:
            problems.append(f"Cannot read controls: {error}")
        return Run(path, manifest, rows, tuple(problems), controls)
    result_file = path / "task_results.jsonl"
    summary_file = path / "summary.json"
    required = require_evaluation or manifest["status"] == "completed"
    if not required and not result_file.exists() and not summary_file.exists():
        return Run(path, manifest, rows, ())
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
        summary = read_object(summary_file)
        if summary.get("total_tasks") != len(rows) or summary.get("resolved") != sum(
            row["resolved"] for row in rows.values()
        ):
            problems.append("Summary disagrees with task results")
    except (OSError, ValueError) as error:
        problems.append(f"Cannot read summary: {error}")
    if required and set(rows) != set(ids):
        problems.append("Completed run has missing task results")
    if "evaluation_artifacts" in manifest:
        try:
            for name, digest in manifest["evaluation_artifacts"].items():
                artifact = (path / name).resolve()
                if not artifact.is_relative_to(path.resolve()) or file_hash(artifact) != digest:
                    raise ValueError("Evaluation artifact changed")
        except (OSError, ValueError, TypeError, AttributeError) as error:
            problems.append(f"Invalid evaluation artifacts: {error}")
    return Run(path, manifest, rows, tuple(problems))


def comparison_differences(left: Run, right: Run) -> list[str]:
    if left.kind != "evaluation" or right.kind != "evaluation":
        raise ValueError("Reference verification is not an agent evaluation")
    differences = []
    if set(left.task_ids) != set(right.task_ids):
        differences.append("task_ids")
    for field in (
        "evaluation",
        "environment",
        "tasks_sha256",
        "verification_timeout_seconds",
        "vllm_config",
        "submission_files",
    ):
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


def read_artifact(run: Run, relative: str, max_bytes: int) -> str:
    path = (run.path / relative).resolve()
    if not path.is_relative_to(run.path.resolve()):
        raise ValueError("Artifact path escapes the run directory")
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    text = content[:max_bytes].decode("utf-8", errors="replace")
    return text + ("\n[Display truncated]" if len(content) > max_bytes else "")
