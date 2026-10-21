"""Shared task inputs without saved agent patches or model initialization."""

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.config import mapping, project_path


def checksum(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Expected SHA-256")
    return value


@dataclass(frozen=True)
class TaskSpec:
    instance_id: str
    snapshot_sha256: str
    test_paths: list[str]
    source_roots: list[str]
    import_modules: list[str]
    dependency_wheels: dict[str, str]

    @classmethod
    def parse(cls, raw: Any) -> "TaskSpec":
        data = mapping(raw, {f.name for f in fields(cls)}, "task specification")
        if not isinstance(data["instance_id"], str) or not re.fullmatch(
            r"[\w-]+", data["instance_id"]
        ):
            raise ValueError("Invalid task ID")
        checksum(data["snapshot_sha256"])
        if not isinstance(data["dependency_wheels"], dict):
            raise ValueError("Expected task dependency wheel hashes")
        for name, digest in data["dependency_wheels"].items():
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".whl"):
                raise ValueError("Invalid dependency wheel name")
            checksum(digest)
        for key in ("test_paths", "source_roots", "import_modules"):
            items = data[key]
            if (
                not isinstance(items, list)
                or not items
                or any(not isinstance(p, str) or not p for p in items)
                or len(items) != len(set(items))
            ):
                raise ValueError(f"Invalid task list: {key}")
        for path in data["test_paths"] + data["source_roots"]:
            if Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path:
                raise ValueError("Task paths must stay inside the workspace")
        if any(not p.endswith(".py") for p in data["test_paths"]) or any(
            not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", p) for p in data["import_modules"]
        ):
            raise ValueError("Invalid test paths or import modules")
        return cls(**data)


@dataclass(frozen=True)
class TaskCatalog:
    path: Path
    tasks_file: Path
    tasks_sha256: str
    tasks: dict[str, TaskSpec]
    wheel_cache: Path

    @classmethod
    def load(cls, root: Path, path: Path) -> "TaskCatalog":
        path = project_path(root, str(path))
        data = mapping(
            yaml.safe_load(path.read_text()),
            {"tasks_file", "tasks_sha256", "tasks", "wheel_cache"},
            "task catalog",
        )
        if not isinstance(data["tasks"], list) or not data["tasks"]:
            raise ValueError("Expected catalog tasks")
        items = [TaskSpec.parse(t) for t in data["tasks"]]
        tasks = {t.instance_id: t for t in items}
        if len(tasks) != len(items):
            raise ValueError("Duplicate catalog task IDs")
        wheel_hashes: dict[str, str] = {}
        for task in items:
            for name, digest in task.dependency_wheels.items():
                if name in wheel_hashes and wheel_hashes[name] != digest:
                    raise ValueError("Conflicting task dependency hashes")
                wheel_hashes[name] = digest
        result = cls(
            path,
            project_path(root, data["tasks_file"]),
            checksum(data["tasks_sha256"]),
            tasks,
            project_path(root, data["wheel_cache"]),
        )
        if not result.wheel_cache.is_relative_to(root / "artifacts"):
            raise ValueError("Task wheel cache must stay in artifacts")
        result.read_tasks()
        return result

    def wheels(self, ids: list[str]) -> dict[Path, str]:
        expected = {
            name: digest
            for task_id in ids
            for name, digest in self.tasks[task_id].dependency_wheels.items()
        }
        for name, digest in expected.items():
            if file_hash(self.wheel_cache / name) != digest:
                raise ValueError("Task dependency wheel checksum mismatch")
        return {self.wheel_cache / name: digest for name, digest in expected.items()}

    def read_tasks(self) -> dict[str, dict[str, Any]]:
        if file_hash(self.tasks_file) != self.tasks_sha256:
            raise ValueError("Task data checksum mismatch")
        records: dict[str, dict[str, Any]] = {}
        for line in self.tasks_file.read_text().splitlines():
            task = json.loads(line)
            if not isinstance(task, dict) or not isinstance(task.get("instance_id"), str):
                raise ValueError("Invalid task record")
            task_id = task["instance_id"]
            if task_id not in self.tasks:
                continue
            if task_id in records or any(
                not isinstance(task.get(k), str) or not task[k].strip()
                for k in ("repo", "base_commit", "patch", "test_patch", "problem_statement")
            ):
                raise ValueError("Invalid or duplicate task data")
            expected = re.findall(r"^\+\+\+ b/(.+)$", task["test_patch"], re.M)
            if set(expected) != set(self.tasks[task_id].test_paths):
                raise ValueError("Test paths must match the official test patch")
            records[task_id] = task
        if set(records) != set(self.tasks):
            raise ValueError("Task data is incomplete")
        return records
