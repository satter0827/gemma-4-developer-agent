"""Persist references to authoritative evidence and resume only verified stages."""

import fcntl
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import (
    content_hash,
    file_hash,
    inventory,
    new_run_id,
    write_json,
)


@contextmanager
def exclusive(root: Path) -> Iterator[None]:
    directory = root / "artifacts/workflows"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another workflow is updating this checkout") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def checked_path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Workflow paths must stay inside the project")
    return path


def validate_files(root: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if file_hash(checked_path(root, name)) != digest:
            raise ValueError("Workflow evidence changed")


def inspect(root: Path, path: Path) -> dict[str, Any]:
    path = checked_path(root, path)
    report = json.loads(path.read_text())
    if report.get("kind") != "development-workflow":
        raise ValueError("Expected a workflow record")
    validate_files(root, report["input_files"])
    for stage in report["stages"].values():
        if stage["status"] == "completed":
            validate_files(root, stage["files"])
    return dict(report)


class Flow:
    def __init__(
        self,
        root: Path,
        name: str,
        inputs: dict[str, Any],
        files: dict[str, str],
        resume: Path | None,
    ):
        self.root = root
        if resume is not None:
            self.path = checked_path(root, resume)
            self.report = inspect(root, self.path)
            if (
                self.report["flow"] != name
                or self.report["inputs_sha256"] != content_hash(inputs)
                or self.report["input_files"] != files
            ):
                raise ValueError("Workflow inputs changed; start a new workflow")
        else:
            self.path = root / "artifacts/workflows" / new_run_id() / "workflow.json"
            self.path.parent.mkdir(parents=True)
            self.report = {
                "kind": "development-workflow",
                "workflow_id": self.path.parent.name,
                "flow": name,
                "inputs_sha256": content_hash(inputs),
                "input_files": files,
                "status": "running",
                "stage": "inputs",
                "stages": {},
                "artifacts": {},
                "reason_codes": [],
                "next_actions": [],
                "completion": {
                    name: "not_run"
                    for name in (
                        "quality",
                        "cpu_verification",
                        "gpu_evaluation",
                        "comparison",
                        "formal_submission",
                    )
                },
            }
        self.report.update(status="running", reason_codes=[])
        self.save()

    def __enter__(self) -> "Flow":
        return self

    def __exit__(self, kind: object, error: BaseException | None, traceback: object) -> None:
        if error is not None:
            self.report.update(status="failed", reason_codes=[type(error).__name__])
            error.add_note("workflow_record=" + str(self.path))
            try:
                self.save()
            except OSError as save_error:
                error.add_note("Cannot save workflow failure: " + str(save_error))

    def save(self) -> None:
        write_json(self.path, self.report)

    def step(self, name: str, action: Callable[[], Path]) -> Path:
        stages = self.report["stages"]
        self.report["stage"] = name
        if name in stages and stages[name]["status"] == "completed":
            validate_files(self.root, stages[name]["files"])
            return checked_path(self.root, stages[name]["artifact"])
        stages[name] = {"status": "running"}
        self.save()
        try:
            path = checked_path(self.root, action())
            # Environment receipts are verified by their owning service, never by hashing a venv.
            files = (
                {path.name: file_hash(path)}
                if path.name == "prepared.json" or not path.is_relative_to(self.root / "artifacts")
                else inventory(path.parent)
            )
            references = {
                str((path.parent / key).relative_to(self.root)): value
                for key, value in files.items()
            }
            stages[name] = {
                "status": "completed",
                "artifact": str(path.relative_to(self.root)),
                "files": references,
            }
            self.report["artifacts"][name] = str(path.resolve())
            self.save()
            return path
        except BaseException as error:
            stages[name] = {"status": "failed"}
            self.report.update(status="failed", reason_codes=[type(error).__name__])
            error.add_note("workflow_record=" + str(self.path))
            try:
                self.save()
            except OSError as save_error:
                error.add_note("Cannot save workflow failure: " + str(save_error))
            raise

    def finish(self, stage: str, next_actions: list[dict[str, Any]]) -> dict[str, Any]:
        self.report.update(status="completed", stage=stage, next_actions=next_actions)
        self.report["artifacts"]["workflow"] = str(self.path)
        self.save()
        return dict(self.report)
