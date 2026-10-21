"""Validate reproducible patch-diagnostic inputs independently of model execution."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.tasks.catalog import TaskCatalog, TaskSpec


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    count: int


@dataclass(frozen=True)
class DiagnosticTask(TaskSpec):
    generated_patch: Path | None
    generated_sha256: str | None
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
        if self.generated_patch is None:
            return {"reference": task["patch"], "unmodified": ""}
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
    catalog_path: Path
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
        names = {f.name for f in fields(cls)} - {
            "root",
            "config_path",
            "runtime",
            "tasks_file",
            "tasks_sha256",
        }
        try:
            document = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid diagnostic YAML: {config_path.name}") from error
        value = mapping(document, names, "diagnostics")
        paths = {
            name: project_path(root, value[name])
            for name in (
                "runtime_path",
                "catalog_path",
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
        for name in ("runtime_path", "catalog_path", "requirements", "lock_file"):
            if not paths[name].is_file():
                raise ValueError(f"Diagnostic input must be a file: {name}")
        catalog = TaskCatalog.load(root, paths["catalog_path"])
        paths["tasks_file"] = catalog.tasks_file
        value["tasks_sha256"] = catalog.tasks_sha256
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
            data = mapping(
                raw,
                ({f.name for f in fields(DiagnosticTask)} - {f.name for f in fields(TaskSpec)})
                | {"instance_id"},
                "diagnostic task",
            )
            if data["instance_id"] not in catalog.tasks:
                raise ValueError("Unknown diagnostic task ID")
            data = {**asdict(catalog.tasks[data["instance_id"]]), **data}
            if not isinstance(data["instance_id"], str):
                raise ValueError("Invalid diagnostic task ID")
            if any(
                type(data[k]) is not bool
                for k in ("probe_expected_pass", "generated_expected_pass")
            ):
                raise ValueError("Expected probe outcome boolean")
            for key in ("generated_sha256", "snapshot_sha256"):
                if (
                    key == "generated_sha256"
                    and data["generated_patch"] is None
                    and data[key] is None
                ):
                    continue
                if not isinstance(data[key], str) or not re.fullmatch(r"[a-f0-9]{64}", data[key]):
                    raise ValueError(f"Expected SHA-256: {key}")
            if (data["generated_patch"] is None) != (data["generated_sha256"] is None):
                raise ValueError("Generated patch and hash must be present together")
            if data["generated_patch"] is None and (
                data["probe_replacements"]
                or data["generated_expected_pass"]
                or data["probe_expected_pass"]
            ):
                raise ValueError("Control-only diagnostics cannot declare generated outcomes")
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
                        "generated_patch": project_path(root, data["generated_patch"])
                        if data["generated_patch"] is not None
                        else None,
                        "probe_replacements": changes,
                    }
                )
            )
        if not tasks or len({t.instance_id for t in tasks}) != len(tasks):
            raise ValueError("Empty or duplicate diagnostic tasks")
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
            *(paths[n] for n in ("runtime_path", "catalog_path", "requirements", "lock_file")),
        ]
        inputs.extend(task.generated_patch for task in tasks if task.generated_patch is not None)
        if any(path.is_relative_to(directory) for path in inputs for directory in storage):
            raise ValueError("Diagnostic inputs must stay outside managed storage")
        settings.read_tasks()
        return settings

    def read_tasks(self) -> dict[str, dict[str, Any]]:
        if file_hash(self.tasks_file) != self.tasks_sha256:
            raise ValueError("Task data checksum mismatch")
        records = TaskCatalog.load(self.root, self.catalog_path).read_tasks()
        records = {t.instance_id: records[t.instance_id] for t in self.tasks}
        for task in self.tasks:
            test_paths = re.findall(
                r"^\+\+\+ b/(.+)$", records[task.instance_id]["test_patch"], re.M
            )
            if set(task.test_paths) != set(test_paths):
                raise ValueError("Diagnostic test paths must match the official test patch")
            task.patches(records[task.instance_id])
        return records

    def preparation_fingerprint(self) -> str:
        value = {
            "python": self.python_version,
            "tasks_sha256": self.tasks_sha256,
            "requirements": file_hash(self.requirements),
            "lock": file_hash(self.lock_file),
            "snapshots": {t.instance_id: t.snapshot_sha256 for t in self.tasks},
            "wheels": {t.instance_id: t.dependency_wheels for t in self.tasks},
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def fingerprint(self) -> str:
        from gemma_agent_dev.common.sources import LOCAL_DIAGNOSTIC_MODULES, source_paths

        value = {
            "preparation": self.preparation_fingerprint(),
            "settings": asdict(self),
            "runtime": self.runtime.as_dict(),
            "sources": {
                p.relative_to(self.root).as_posix(): file_hash(p)
                for p in source_paths(self.root, LOCAL_DIAGNOSTIC_MODULES)
            },
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
