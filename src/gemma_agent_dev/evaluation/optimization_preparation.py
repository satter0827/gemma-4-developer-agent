"""Prepare GEPA independently and freeze CPU evidence for one optimization."""

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.candidates.preflight import (
    check_installation,
    freeze_candidate,
    verify_quality,
)
from gemma_agent_dev.common.artifacts import (
    content_hash,
    file_hash,
    inventory,
    new_run_id,
    write_json,
)
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.common.events import logged
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import check_prepared, diagnose
from gemma_agent_dev.evaluation.control_evidence import validate_control_set
from gemma_agent_dev.evaluation.optimization import Optimization
from gemma_agent_dev.tasks.catalog import checksum


@dataclass(frozen=True)
class GepaEnvironment:
    requirements: Path
    lock_file: Path
    environment: Path
    wheel_cache: Path
    wheel_name: str
    wheel_sha256: str
    python_version: str
    timeout_seconds: int

    @classmethod
    def load(cls, root: Path, config: Path) -> "GepaEnvironment":
        data = mapping(
            yaml.safe_load(project_path(root, str(config)).read_text()),
            set(cls.__dataclass_fields__),
            "GEPA environment",
        )
        for name in ("requirements", "lock_file", "environment", "wheel_cache"):
            data[name] = project_path(root, data[name])
        storage = [data["environment"], data["wheel_cache"]]
        protected = [
            root / "artifacts" / name
            for name in ("venv", "runtime-env", "tracking-env", "results", "kaggle")
        ]
        if (
            any(
                not p.is_relative_to(root / "artifacts")
                or any(p.is_relative_to(other) or other.is_relative_to(p) for other in protected)
                for p in storage
            )
            or storage[0].is_relative_to(storage[1])
            or storage[1].is_relative_to(storage[0])
        ):
            raise ValueError("GEPA storage must be isolated inside artifacts")
        if data["python_version"] != "3.12" or data["wheel_name"] != "gepa-0.1.4-py3-none-any.whl":
            raise ValueError("Unsupported GEPA Python or wheel version")
        checksum(data["wheel_sha256"])
        positive(data["timeout_seconds"], "GEPA preparation timeout", integer=True)
        if any(not data[n].is_file() for n in ("requirements", "lock_file")):
            raise ValueError("Missing GEPA preparation inputs")
        return cls(**data)

    def wheel(self) -> Path:
        path = self.wheel_cache / self.wheel_name
        if not path.is_file() or file_hash(path) != self.wheel_sha256:
            raise ValueError("Missing or changed GEPA wheel")
        return path

    def fingerprint(self) -> dict[str, str]:
        return {
            "requirements": file_hash(self.requirements),
            "lock_file": file_hash(self.lock_file),
            "wheel_sha256": self.wheel_sha256,
            "python_version": self.python_version,
        }

    def installed(self, root: Path) -> dict[str, Any]:
        return dict(
            json.loads(
                command(
                    [
                        str(self.environment / "bin/python"),
                        "-c",
                        "import importlib.metadata,json,platform;"
                        "print(json.dumps({'python':platform.python_version(),"
                        "'packages':{d.metadata['Name']:d.version "
                        "for d in importlib.metadata.distributions()}}))",
                    ],
                    cwd=root,
                    timeout=self.timeout_seconds,
                )
            )
        )

    def verify(self, root: Path) -> None:
        self.wheel()
        if json.loads((self.environment / "prepared.json").read_text()) != {
            "inputs": self.fingerprint(),
            "installed": self.installed(root),
        }:
            raise ValueError("GEPA environment changed; prepare again")


@logged("optimization.environment")
def prepare_environment(root: Path, config: Path) -> Path:
    settings = GepaEnvironment.load(root, config)
    path = settings.environment / "prepared.json"
    path.unlink(missing_ok=True)
    wheel = settings.wheel_cache / settings.wheel_name
    if not wheel.exists():
        with urllib.request.urlopen(
            "https://pypi.org/pypi/gepa/0.1.4/json", timeout=settings.timeout_seconds
        ) as response:
            metadata = json.load(response)
        item = next(
            (
                item
                for item in metadata["urls"]
                if item["filename"] == settings.wheel_name
                and item["digests"]["sha256"] == settings.wheel_sha256
            ),
            None,
        )
        if item is None:
            raise ValueError("Pinned GEPA wheel is absent from the official package index")
        with urllib.request.urlopen(item["url"], timeout=settings.timeout_seconds) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != settings.wheel_sha256:
            raise ValueError("Downloaded GEPA wheel changed")
        settings.wheel_cache.mkdir(parents=True, exist_ok=True)
        wheel.write_bytes(content)
    settings.wheel()
    env = {**os.environ, "UV_CACHE_DIR": str(root / "artifacts/cache/uv")}
    python = settings.environment / "bin/python"
    if not python.exists():
        command(
            ["uv", "venv", "--python", settings.python_version, str(settings.environment)],
            cwd=root,
            env=env,
            timeout=settings.timeout_seconds,
        )
    command(
        ["uv", "pip", "sync", "--python", str(python), "--require-hashes", str(settings.lock_file)],
        cwd=root,
        env=env,
        timeout=settings.timeout_seconds,
    )
    state = settings.installed(root)
    if state["packages"].get("gepa") != "0.1.4" or not state["python"].startswith(
        settings.python_version + "."
    ):
        raise ValueError("GEPA installation changed")
    write_json(path, {"inputs": settings.fingerprint(), "installed": state})
    return path


@logged("optimization.prepare")
def prepare(root: Path, config: Path) -> Path:
    optimization = Optimization.load(root, config)
    settings = optimization.preflight(root)
    GepaEnvironment.load(root, optimization.environment).verify(root)
    check_installation(root)
    diagnostics = DiagnosticSettings.load(root, optimization.diagnostic_config)
    check_prepared(diagnostics)
    definition = optimization.definition(root)
    before = settings.inputs()
    output = root / "artifacts/optimizations" / new_run_id()
    output.mkdir(parents=True)
    path = output / "preparation.json"
    report: dict[str, Any] = {
        "kind": "optimization-preparation",
        "status": "running",
        "config": str(config.relative_to(root) if config.is_absolute() else config),
        "definition": definition,
        "definition_sha256": content_hash(definition),
        "inputs": before,
    }
    write_json(path, report)
    try:
        verify_quality(settings, output)
        controls = diagnose(diagnostics) / "report.json"
        report["controls"] = {
            "path": str(controls.relative_to(root)),
            "sha256": file_hash(controls),
        }
        seed = output / "seed"
        seed.mkdir()
        files = freeze_candidate(settings, seed)
        report["candidate"] = {
            "files": files,
            "sha256": file_hash(seed / "submission.zip"),
            "max_unpacked_bytes": settings.max_unpacked_bytes,
        }
        if definition != optimization.definition(root) or before != settings.inputs():
            raise ValueError("Optimization inputs changed during preparation")
        report.update(status="completed", artifacts=inventory(output))
        report["artifacts"].pop("preparation.json", None)
        write_json(path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, report)
        raise
    return path


def load_preparation(root: Path, path: Path) -> tuple[Optimization, dict[str, Any]]:
    path = project_path(root, str(path))
    if not path.is_relative_to(root / "artifacts/optimizations"):
        raise ValueError("Expected a local optimization preparation")
    report = json.loads(path.read_text())
    optimization = Optimization.load(root, Path(report["config"]))
    files = inventory(path.parent)
    files.pop("preparation.json", None)
    controls = project_path(root, report["controls"]["path"])
    if (
        report["kind"] != "optimization-preparation"
        or report["status"] != "completed"
        or report["definition"] != optimization.definition(root)
        or report["definition_sha256"] != content_hash(report["definition"])
        or report["inputs"] != optimization.preflight(root).inputs()
        or report["artifacts"] != files
        or file_hash(controls) != report["controls"]["sha256"]
    ):
        raise ValueError("Optimization preparation is incomplete or changed")
    evidence = json.loads(controls.read_text())
    validate_control_set(evidence, optimization.search_task_ids + optimization.selection_task_ids)
    if evidence["status"] != "completed":
        raise ValueError("Optimization controls did not complete")
    for case in evidence["cases"]:
        folder = controls.parent / case["task_id"] / case["case"]
        files = inventory(folder)
        files.pop("case.json", None)
        if (
            not case["matched"]
            or files != case["artifacts"]
            or json.loads((folder / "case.json").read_text()) != case
        ):
            raise ValueError("Optimization control evidence changed")
    return optimization, report
