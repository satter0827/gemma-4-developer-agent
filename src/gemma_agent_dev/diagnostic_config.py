"""Validate reproducible patch-diagnostic inputs independently of model execution."""

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.artifacts import file_hash
from gemma_agent_dev.config import RuntimeSettings, mapping, positive, project_path


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    count: int


@dataclass(frozen=True)
class DiagnosticTask:
    instance_id: str
    snapshot_sha256: str
    test_paths: list[str]
    generated_patch: Path
    generated_sha256: str
    probe_replacements: list[Replacement]
    probe_expected_pass: bool
    generated_expected_pass: bool
    collection_error_cases: list[str]

    def expected_outcome(self, name: str) -> str:
        if name in self.collection_error_cases:
            return "collection_error"
        passed = (
            name == "reference"
            or (name == "probe" and self.probe_expected_pass)
            or (name == "generated" and self.generated_expected_pass)
        )
        return "passed" if passed else "failed"

    def patches(self, task: dict[str, Any]) -> dict[str, str]:
        if file_hash(self.generated_patch) != self.generated_sha256:
            raise ValueError(f"Generated patch checksum mismatch: {self.instance_id}")
        generated = self.generated_patch.read_text()
        probe = generated
        for change in self.probe_replacements:
            if probe.count(change.old) != change.count:
                raise ValueError(f"Diagnostic replacement count mismatch: {self.instance_id}")
            probe = probe.replace(change.old, change.new)
        patches = {
            "reference": task["patch"],
            "unmodified": "",
            "generated": generated,
        }
        if self.probe_replacements:
            patches["probe"] = probe
        return patches


@dataclass(frozen=True)
class DiagnosticSettings:
    root: Path
    config_path: Path
    runtime_path: Path
    runtime: RuntimeSettings
    tasks_file: Path
    tasks_sha256: str
    cache_dir: Path
    environments_dir: Path
    output_dir: Path
    requirements: Path
    lock_file: Path
    python_version: str
    kernel_id: str
    timeout_seconds: int
    pytest_args: list[str]
    tasks: list[DiagnosticTask]

    @classmethod
    def load(cls, root: Path, config: Path) -> "DiagnosticSettings":
        root = root.resolve()
        config_path = project_path(root, str(config))
        names = {f.name for f in fields(cls)} - {"root", "config_path", "runtime"}
        try:
            document = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid diagnostic YAML: {config_path.name}") from error
        value = mapping(document, names, "diagnostics")
        paths = {
            name: project_path(root, value[name])
            for name in (
                "runtime_path",
                "tasks_file",
                "cache_dir",
                "environments_dir",
                "output_dir",
                "requirements",
                "lock_file",
            )
        }
        for name in ("cache_dir", "environments_dir", "output_dir"):
            if not paths[name].is_relative_to(root / "artifacts"):
                raise ValueError(f"Diagnostic storage must stay in artifacts: {name}")
            if paths[name].exists() and not paths[name].is_dir():
                raise ValueError(f"Diagnostic storage must be a directory: {name}")
        storage = [paths[name] for name in ("cache_dir", "environments_dir", "output_dir")]
        protected = [root / "artifacts" / name for name in ("results", "venv", "cache", "kaggle")]
        for index, directory in enumerate(storage):
            for other in storage[index + 1 :] + protected:
                if directory.is_relative_to(other) or other.is_relative_to(directory):
                    raise ValueError("Diagnostic storage must not overlap other storage")
        for name in ("runtime_path", "tasks_file", "requirements", "lock_file"):
            if not paths[name].is_file():
                raise ValueError(f"Diagnostic input must be a file: {name}")
        if not isinstance(value["tasks_sha256"], str) or not re.fullmatch(
            r"[a-f0-9]{64}", value["tasks_sha256"]
        ):
            raise ValueError("Expected tasks SHA-256")
        if value["python_version"] != "3.12":
            raise ValueError("Diagnostics require Python 3.12")
        if not isinstance(value["kernel_id"], str) or not re.fullmatch(
            r"[\w-]+/[\w-]+", value["kernel_id"]
        ):
            raise ValueError("Invalid diagnostic kernel ID")
        positive(value["timeout_seconds"], "diagnostic timeout", integer=True)
        if not isinstance(value["pytest_args"], list) or any(
            not isinstance(arg, str) or not arg for arg in value["pytest_args"]
        ):
            raise ValueError("Expected pytest arguments")
        runtime = RuntimeSettings.load(paths["runtime_path"])
        if value["kernel_id"] in {runtime.kernel_id, runtime.verification_kernel_id}:
            raise ValueError("Diagnostics require a separate private notebook")
        if not isinstance(value["tasks"], list):
            raise ValueError("Expected diagnostic tasks")
        tasks = []
        for raw in value["tasks"]:
            data = mapping(raw, {f.name for f in fields(DiagnosticTask)}, "diagnostic task")
            if not isinstance(data["instance_id"], str):
                raise ValueError("Invalid diagnostic task ID")
            test_paths = data["test_paths"]
            if (
                not isinstance(test_paths, list)
                or not test_paths
                or any(
                    not isinstance(p, str)
                    or not p.endswith(".py")
                    or Path(p).is_absolute()
                    or ".." in Path(p).parts
                    for p in test_paths
                )
                or len(test_paths) != len(set(test_paths))
            ):
                raise ValueError("Expected relative Python test paths")
            if any(
                type(data[k]) is not bool
                for k in ("probe_expected_pass", "generated_expected_pass")
            ):
                raise ValueError("Expected probe outcome boolean")
            for key in ("generated_sha256", "snapshot_sha256"):
                if not isinstance(data[key], str) or not re.fullmatch(r"[a-f0-9]{64}", data[key]):
                    raise ValueError(f"Expected SHA-256: {key}")
            changes = []
            if not isinstance(data["probe_replacements"], list):
                raise ValueError("Expected diagnostic replacements")
            for raw_change in data["probe_replacements"]:
                change = mapping(raw_change, {f.name for f in fields(Replacement)}, "replacement")
                if any(not isinstance(change[k], str) or not change[k] for k in ("old", "new")):
                    raise ValueError("Expected nonempty replacement strings")
                positive(change["count"], "replacement count", integer=True)
                changes.append(Replacement(**change))
            tasks.append(
                DiagnosticTask(
                    **{
                        **data,
                        "generated_patch": project_path(root, data["generated_patch"]),
                        "probe_replacements": changes,
                    }
                )
            )
        if [t.instance_id for t in tasks] != runtime.task_ids:
            raise ValueError("Diagnostic task order must match the shared runtime task set")
        for task in tasks:
            collection_cases = task.collection_error_cases
            allowed = {"unmodified", "generated"} | (
                {"probe"} if task.probe_replacements else set()
            )
            if (
                not isinstance(collection_cases, list)
                or any(not isinstance(n, str) or n not in allowed for n in collection_cases)
                or len(collection_cases) != len(set(collection_cases))
            ):
                raise ValueError("Invalid expected collection-error cases")
            if ("generated" in collection_cases and task.generated_expected_pass) or (
                "probe" in collection_cases and task.probe_expected_pass
            ):
                raise ValueError("Collection failure cannot also be expected to pass")
        settings = cls(
            **{
                **value,
                **paths,
                "root": root,
                "config_path": config_path,
                "runtime": runtime,
                "tasks": tasks,
            }
        )
        inputs = [
            config_path,
            *(paths[n] for n in ("runtime_path", "tasks_file", "requirements", "lock_file")),
        ]
        inputs.extend(task.generated_patch for task in tasks)
        if any(path.is_relative_to(directory) for path in inputs for directory in storage):
            raise ValueError("Diagnostic inputs must stay outside managed storage")
        settings.read_tasks()
        return settings

    def read_tasks(self) -> dict[str, dict[str, Any]]:
        if file_hash(self.tasks_file) != self.tasks_sha256:
            raise ValueError("Task data checksum mismatch")
        records: dict[str, dict[str, Any]] = {}
        for line in self.tasks_file.read_text().splitlines():
            task = json.loads(line)
            if not isinstance(task, dict) or not isinstance(task.get("instance_id"), str):
                raise ValueError("Invalid diagnostic task record")
            task_id = task["instance_id"]
            if task_id in self.runtime.task_ids:
                if task_id in records or any(
                    not isinstance(task.get(k), str) or not task[k].strip()
                    for k in ("repo", "base_commit", "patch", "test_patch", "problem_statement")
                ):
                    raise ValueError("Invalid or duplicate diagnostic task data")
                records[task_id] = task
        if set(records) != set(self.runtime.task_ids):
            raise ValueError("Diagnostic task data is incomplete")
        for task in self.tasks:
            test_paths = re.findall(
                r"^\+\+\+ b/(.+)$", records[task.instance_id]["test_patch"], re.M
            )
            if set(task.test_paths) != set(test_paths):
                raise ValueError("Diagnostic test paths must match the official test patch")
            task.patches(records[task.instance_id])
        return records

    def fingerprint(self) -> str:
        paths = [
            self.config_path,
            self.runtime_path,
            self.tasks_file,
            self.requirements,
            self.lock_file,
        ]
        paths.extend(
            self.root / "src/gemma_agent_dev" / name
            for name in (
                "diagnostics.py",
                "diagnostic_config.py",
                "diagnostic_results.py",
                "config.py",
                "artifacts.py",
                "processes.py",
            )
        )
        hashes = {p.relative_to(self.root).as_posix(): file_hash(p) for p in paths}
        return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
