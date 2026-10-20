"""Load the evaluation contract without implicit CLI defaults."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvaluationSettings:
    timeout_seconds: int
    max_tool_calls: int
    max_time_minutes: float
    max_turns: int

    @classmethod
    def load(cls, path: Path) -> "EvaluationSettings":
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid evaluation YAML: {path.name}") from error
        if not isinstance(document, dict) or not isinstance(document.get("evaluation"), dict):
            raise ValueError("Expected an evaluation mapping")
        values = document["evaluation"]
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            raise ValueError(f"Expected evaluation keys: {sorted(expected)}")
        for name, value in values.items():
            valid_type = (
                type(value) in (int, float) if name == "max_time_minutes" else type(value) is int
            )
            if not valid_type or not 0 < value < float("inf"):
                raise ValueError(f"Invalid positive evaluation value: {name}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
