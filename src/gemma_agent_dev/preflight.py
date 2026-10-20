"""Run offline local gates and freeze the exact candidate used by Kaggle."""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.candidates import extract_archive, validate_references
from gemma_agent_dev.config import (
    EvaluationSettings,
    RuntimeSettings,
    mapping,
    positive,
    project_path,
)
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.packaging import package_submission
from gemma_agent_dev.processes import process_group


@dataclass(frozen=True)
class PreflightSettings:
    root: Path
    config_path: Path
    agent_dir: Path
    evaluation_config: Path
    runtime_config: Path
    diagnostic_config: Path
    adapter_archive: Path
    adapter_sha256: str
    output_dir: Path
    max_unpacked_bytes: int
    quality_timeout_seconds: int

    @classmethod
    def load(cls, root: Path, path: Path) -> "PreflightSettings":
        root = root.resolve()
        path = project_path(root, str(path))
        data = mapping(
            yaml.safe_load(path.read_text()),
            {f.name for f in fields(cls)} - {"root", "config_path"},
            "preflight",
        )
        for key in (
            "agent_dir",
            "evaluation_config",
            "runtime_config",
            "diagnostic_config",
            "adapter_archive",
            "output_dir",
        ):
            data[key] = project_path(root, data[key])
        for key in ("max_unpacked_bytes", "quality_timeout_seconds"):
            positive(data[key], key, integer=True)
        output = data["output_dir"]
        if not output.is_relative_to(root / "artifacts/preflight"):
            raise ValueError("Preflight output must stay inside artifacts/preflight")
        for key in (
            "agent_dir",
            "evaluation_config",
            "runtime_config",
            "diagnostic_config",
            "adapter_archive",
        ):
            if (
                not data[key].exists()
                or data[key].is_relative_to(output)
                or output.is_relative_to(data[key])
            ):
                raise ValueError(f"Missing or overlapping preflight input: {key}")
        if file_hash(data["adapter_archive"]) != data["adapter_sha256"]:
            raise ValueError("Adapter archive checksum mismatch")
        EvaluationSettings.load(data["evaluation_config"])
        runtime = RuntimeSettings.load(data["runtime_config"])
        diagnostics = DiagnosticSettings.load(root, data["diagnostic_config"])
        if diagnostics.runtime.as_dict() != runtime.as_dict():
            raise ValueError("Preflight and diagnostic runtime settings differ")
        return cls(root=root, config_path=path, **data)

    def inputs(self) -> dict[str, str]:
        paths = [
            self.config_path,
            self.adapter_archive,
            self.evaluation_config,
            self.runtime_config,
            self.diagnostic_config,
        ]
        for directory in ("src", "kaggle", "scripts", "tests", "configs", "agent"):
            paths.extend(
                p
                for p in (self.root / directory).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            )
        paths.extend(self.root / name for name in ("Makefile", "pyproject.toml", "uv.lock"))
        diagnostics = DiagnosticSettings.load(self.root, self.diagnostic_config)
        paths.extend([diagnostics.tasks_file, diagnostics.lock_file, diagnostics.requirements])
        paths.extend(t.generated_patch for t in diagnostics.tasks)
        paths.extend(p for p in self.agent_dir.rglob("*") if p.is_file())
        return {p.relative_to(self.root).as_posix(): file_hash(p) for p in sorted(set(paths))}


def check_installation(root: Path) -> None:
    import gemma_agent_dev

    installed = Path(gemma_agent_dev.__file__).parent
    expected = {p.name: file_hash(p) for p in (root / "src/gemma_agent_dev").glob("*.py")}
    actual = {p.name: file_hash(p) for p in installed.glob("*.py")}
    if actual != expected:
        raise ValueError("Installed development package is stale; run make setup")


def run(root: Path, config: Path) -> Path:
    from gemma_agent_dev.diagnostic_bundle import verified_spec
    from gemma_agent_dev.diagnostics import diagnose

    settings = PreflightSettings.load(root, config)
    check_installation(settings.root)
    before = settings.inputs()
    output = settings.output_dir / new_run_id()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "kind": "preflight",
        "status": "running",
        "config": str(settings.config_path.relative_to(settings.root)),
        "inputs": before,
    }
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        with (output / "quality.log").open("w") as log:
            with process_group(
                ["make", "verify"],
                cwd=settings.root,
                env={**os.environ, "UV_OFFLINE": "1"},
                stdout=log,
            ) as process:
                code = process.wait(timeout=settings.quality_timeout_seconds)
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
        diagnostics = DiagnosticSettings.load(settings.root, settings.diagnostic_config)
        diagnostic_report = diagnose(diagnostics) / "report.json"
        spec = verified_spec(diagnostics)
        report["diagnostics"] = {
            "path": str(diagnostic_report),
            "sha256": file_hash(diagnostic_report),
            "fingerprint": spec["fingerprint"],
        }
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            staging = Path(temporary)
            extracted = staging / "archive"
            extract_archive(settings.adapter_archive, extracted, settings.max_unpacked_bytes)
            submission = staging / "candidate"
            inventory(settings.agent_dir)
            shutil.copytree(settings.agent_dir, submission)
            # Only adapters are reused; no historical agent settings enter the candidate.
            shutil.copytree(extracted / "adapters", submission / "adapters")
            shutil.copyfile(settings.evaluation_config, submission / "eval_config.yaml")
            report["files"] = package_submission(
                submission,
                output / "submission.zip",
                validate_references,
                validation_scope="structural",
            )
        if before != settings.inputs():
            raise ValueError("Preflight inputs changed during validation")
        report.update(status="completed", candidate_sha256=file_hash(output / "submission.zip"))
        report["artifacts"] = inventory(output)
        report["artifacts"].pop("report.json", None)
        write_json(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        raise
    return report_path


def load_report(root: Path, path: Path) -> tuple[PreflightSettings, dict[str, Any]]:
    path = project_path(root, str(path))
    if not path.is_relative_to(root.resolve() / "artifacts/preflight"):
        raise ValueError("Expected a local preflight report")
    report = json.loads(path.read_text())
    if report.get("kind") != "preflight" or report.get("status") != "completed":
        raise ValueError("Preflight did not complete")
    settings = PreflightSettings.load(root, Path(report["config"]))
    if report["inputs"] != settings.inputs():
        raise ValueError("Preflight inputs changed; run preflight again")
    artifacts = inventory(path.parent)
    artifacts.pop("report.json", None)
    if (
        artifacts != report["artifacts"]
        or file_hash(path.parent / "submission.zip") != report["candidate_sha256"]
    ):
        raise ValueError("Preflight artifacts changed")
    diagnostic = project_path(root, report["diagnostics"]["path"])
    diagnostic_data = json.loads(diagnostic.read_text())
    if (
        file_hash(diagnostic) != report["diagnostics"]["sha256"]
        or diagnostic_data["status"] != "completed"
    ):
        raise ValueError("Diagnostic evidence changed")
    for case in diagnostic_data["cases"]:
        folder = diagnostic.parent / case["task_id"] / case["case"]
        files = inventory(folder)
        files.pop("case.json", None)
        if (
            files != case["artifacts"]
            or json.loads((folder / "case.json").read_text()) != case
            or not case["matched"]
        ):
            raise ValueError("Diagnostic case artifacts changed")
    return settings, report


def artifact_limit(root: Path) -> int:
    value = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["gemma_agent_dev"][
        "results"
    ]["max_artifact_bytes"]
    positive(value, "artifact display bytes", integer=True)
    return int(value)
