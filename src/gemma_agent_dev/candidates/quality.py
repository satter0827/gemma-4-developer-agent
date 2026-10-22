"""Bind reusable quality evidence to the complete inspected source and its logs."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.processes import process_group

COMMANDS = ("check", "setup-submission", "verify")


def inputs(root: Path) -> dict[str, str]:
    paths = [root / name for name in ("Makefile", "pyproject.toml", "uv.lock")]
    for name in ("src", "tests", "scripts", "agent", "configs", "docs"):
        paths.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and not path.is_relative_to(root / "docs/records")
        )
    return {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}


def reuse(root: Path, path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_relative_to(root / "artifacts"):
        raise ValueError("Quality evidence must be a saved local artifact")
    report = json.loads(path.read_text())
    if (
        report.get("kind") != "quality"
        or report.get("status") != "completed"
        or report.get("inputs") != inputs(root)
        or [step["command"] for step in report["steps"]] != [["make", c] for c in COMMANDS]
    ):
        raise ValueError("Quality evidence is incomplete or source inputs changed")
    for step in report["steps"]:
        log = (path.parent / step["log"]).resolve(strict=True)
        if (
            not log.is_relative_to(path.parent)
            or step["exit_code"] != 0
            or file_hash(log) != step["sha256"]
        ):
            raise ValueError("Quality command or log changed")
    return {"path": str(path), "sha256": file_hash(path)}


def run(root: Path, output: Path, timeout_seconds: int) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    path = output / "quality.json"
    report: dict[str, Any] = {
        "kind": "quality",
        "status": "running",
        "inputs": inputs(root),
        "steps": [],
    }
    write_json(path, report)
    try:
        for name in COMMANDS:
            log_path = output / (name + ".log")
            with (
                log_path.open("w") as log,
                process_group(
                    ["make", name], cwd=root, env=dict(os.environ), stdout=log
                ) as process,
            ):
                code = process.wait(timeout=timeout_seconds)
            report["steps"].append(
                {
                    "command": ["make", name],
                    "exit_code": code,
                    "log": log_path.name,
                    "sha256": file_hash(log_path),
                }
            )
            write_json(path, report)
            if code:
                raise subprocess.CalledProcessError(code, ["make", name])
        if inputs(root) != report["inputs"]:
            raise ValueError("Source inputs changed during quality checks")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(path, report)
    return path
