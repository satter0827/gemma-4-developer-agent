"""Validate explicit configuration values and project-relative paths."""

import math
from pathlib import Path
from typing import Any


def mapping(value: Any, names: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"Invalid {label} keys; expected {sorted(names)}")
    return value


def positive(value: Any, label: str, *, integer: bool = False) -> None:
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Invalid positive value: {label}")


def project_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty project path")
    path = root / value
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay inside the project: {value}")
    return path.resolve()
