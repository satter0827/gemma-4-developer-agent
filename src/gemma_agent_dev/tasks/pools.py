"""Freeze task pools and explicit use history without selecting on model outcomes."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.config import mapping, project_path
from gemma_agent_dev.tasks.catalog import checksum

IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*")


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Expected a stable lowercase identifier")
    return value


def task_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) != len(set(map(str, value))):
        raise ValueError("Expected ordered, unique task IDs")
    return [identifier(item) for item in value]


@dataclass(frozen=True)
class TaskPool:
    pool_id: str
    difficulty: str
    tasks_sha256: str
    sdk_version: str
    tuning: list[str]
    confirmation: list[str]
    uses: list[dict[str, Any]]

    @classmethod
    def load(cls, root: Path, config: Path) -> "TaskPool":
        data = mapping(
            yaml.safe_load(project_path(root, str(config)).read_text()),
            set(cls.__dataclass_fields__),
            "task pool",
        )
        identifier(data["pool_id"])
        checksum(data["tasks_sha256"])
        if data["difficulty"] != "Easy" or not isinstance(data["sdk_version"], str):
            raise ValueError("New experiments require the fixed Easy pool")
        tuning, confirmation = task_ids(data["tuning"]), task_ids(data["confirmation"])
        if set(tuning) & set(confirmation) or not isinstance(data["uses"], list):
            raise ValueError("Task roles overlap or use history is missing")
        seen = set()
        for raw in data["uses"]:
            use = mapping(raw, {"experiment_id", "purpose", "task_ids"}, "task use")
            experiment = identifier(use["experiment_id"])
            if experiment in seen or use["purpose"] not in {"tuning", "confirmation"}:
                raise ValueError("Duplicate experiment or invalid use purpose")
            seen.add(experiment)
            ids = task_ids(use["task_ids"])
            if not set(ids) <= set(data[use["purpose"]]):
                raise ValueError("Task use violates its frozen role")
        return cls(**data)

    def selection(self, experiment_id: str) -> list[str]:
        matches = [use for use in self.uses if use["experiment_id"] == experiment_id]
        if len(matches) != 1:
            raise ValueError("Experiment needs an explicit task-use record")
        return list(matches[0]["task_ids"])

    def identity(self, experiment_id: str) -> dict[str, str]:
        self.selection(experiment_id)
        use = next(use for use in self.uses if use["experiment_id"] == experiment_id)
        return {
            "pool_id": self.pool_id,
            "difficulty": self.difficulty,
            "tasks_sha256": self.tasks_sha256,
            "sdk_version": self.sdk_version,
            "purpose": str(use["purpose"]),
        }
