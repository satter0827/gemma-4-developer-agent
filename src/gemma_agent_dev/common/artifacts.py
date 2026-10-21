"""Create immutable source fingerprints and atomically write run metadata."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_RANDOM_HEX_LENGTH = 8


def new_run_id() -> str:
    return (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:RUN_ID_RANDOM_HEX_LENGTH]
    )


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def python_inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in Python sources: {path}")
        files[path.relative_to(root).as_posix()] = file_hash(path)
    return files
