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

from gemma_agent_dev.candidates.archive import (
    extract_archive,
    referenced_files,
    validate_references,
)
from gemma_agent_dev.candidates.packaging import package_submission
from gemma_agent_dev.common.artifacts import (
    file_hash,
    inventory,
    new_run_id,
    python_inventory,
    write_json,
)
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.common.processes import process_group
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.config import (
    EvaluationSettings,
    RuntimeSettings,
    SamplingSettings,
)


@dataclass(frozen=True)
class PreflightSettings:
    root: Path
    config_path: Path
    agent_dir: Path
    profiles: Path
    profile: str
    purpose: str
    evaluation_config: Path
    agent_config: Path
    task_selection: Path | None
    runtime_config: Path
    diagnostic_config: Path
    adapter_archive: Path
    adapter_sha256: str
    output_dir: Path
    max_unpacked_bytes: int
    quality_timeout_seconds: int

    @classmethod
    def load(cls, root: Path, path: Path, profile: str) -> "PreflightSettings":
        root = root.resolve()
        path = project_path(root, str(path))
        data = mapping(
            yaml.safe_load(path.read_text()),
            {f.name for f in fields(cls)}
            - {
                "root",
                "config_path",
                "profile",
                "purpose",
                "evaluation_config",
                "agent_config",
                "task_selection",
            },
            "preflight",
        )
        for key in (
            "agent_dir",
            "profiles",
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
            "profiles",
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
        profiles = yaml.safe_load(data["profiles"].read_text())
        if not isinstance(profiles, dict) or profile not in profiles:
            raise ValueError("Unknown execution profile")
        selected = mapping(
            profiles[profile],
            {"purpose", "evaluation_config", "agent_config", "task_selection"},
            "profile",
        )
        if selected["purpose"] not in {"smoke", "comparison"}:
            raise ValueError("Invalid profile purpose")
        for key in ("evaluation_config", "agent_config"):
            selected[key] = project_path(root, selected[key])
            if not selected[key].is_file():
                raise ValueError(f"Missing profile input: {key}")
        if selected["task_selection"] is not None:
            selected["task_selection"] = project_path(root, selected["task_selection"])
        if (selected["purpose"] == "comparison") != (selected["task_selection"] is not None):
            raise ValueError("Comparison profiles require qualified tasks")
        data.update(selected)
        if data["agent_config"].parent != data["agent_dir"]:
            raise ValueError("Agent root must be directly inside agent_dir")
        referenced_files(data["agent_dir"], data["agent_config"])
        EvaluationSettings.load(data["evaluation_config"])
        runtime = RuntimeSettings.load(data["runtime_config"])
        diagnostics = DiagnosticSettings.load(root, data["diagnostic_config"])
        if diagnostics.runtime.as_dict() != runtime.as_dict():
            raise ValueError("Preflight and diagnostic runtime settings differ")
        SamplingSettings.load(data["agent_dir"] / "configs/sampling.yaml", runtime)
        if data["task_selection"] is not None:
            from gemma_agent_dev.diagnostics.qualification import load_selection

            load_selection(root, data["task_selection"])
        return cls(root=root, config_path=path, profile=profile, **data)

    def inputs(self) -> dict[str, str]:
        paths = [self.adapter_archive, self.evaluation_config]
        paths.extend(
            self.agent_dir / name for name in referenced_files(self.agent_dir, self.agent_config)
        )
        for directory in ("src", "scripts", "tests"):
            paths.extend(
                p
                for p in (self.root / directory).rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            )
        paths.extend(self.root / name for name in ("Makefile", "pyproject.toml", "uv.lock"))
        diagnostics = DiagnosticSettings.load(self.root, self.diagnostic_config)
        paths.extend([diagnostics.tasks_file, diagnostics.requirements, diagnostics.lock_file])
        if self.task_selection is not None:
            paths.append(self.task_selection)
        paths.extend(t.generated_patch for t in diagnostics.tasks if t.generated_patch is not None)
        values = {p.relative_to(self.root).as_posix(): file_hash(p) for p in sorted(set(paths))}
        # Bind only the selected profile and the runtime values actually used.
        resolved = {
            f.name: str(getattr(self, f.name))
            for f in fields(self)
            if f.name not in {"root", "config_path", "profiles"}
        }
        import hashlib

        values["@preflight"] = hashlib.sha256(
            json.dumps(resolved, sort_keys=True).encode()
        ).hexdigest()
        values["@runtime"] = hashlib.sha256(
            json.dumps(RuntimeSettings.load(self.runtime_config).as_dict(), sort_keys=True).encode()
        ).hexdigest()
        values["@diagnostics"] = diagnostics.fingerprint()
        return values


def check_installation(root: Path) -> None:
    import gemma_agent_dev

    installed = Path(gemma_agent_dev.__file__).parent
    expected = python_inventory(root / "src/gemma_agent_dev")
    actual = python_inventory(installed)
    if actual != expected:
        raise ValueError("Installed development package is stale; run make setup")


def run(root: Path, config: Path, profile: str) -> Path:
    from gemma_agent_dev.diagnostics.local import diagnose
    from gemma_agent_dev.notebooks.diagnostics import verified_spec

    settings = PreflightSettings.load(root, config, profile)
    check_installation(settings.root)
    before = settings.inputs()
    output = settings.output_dir / new_run_id()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "kind": "preflight",
        "profile": profile,
        "purpose": settings.purpose,
        "execution": {
            "evaluation": EvaluationSettings.load(settings.evaluation_config).as_dict(),
            "sampling": SamplingSettings.load(
                settings.agent_dir / "configs/sampling.yaml",
                RuntimeSettings.load(settings.runtime_config),
            ).__dict__,
            "model_source": RuntimeSettings.load(settings.runtime_config).model_source,
        },
        "status": "running",
        "config": str(settings.config_path.relative_to(settings.root)),
        "inputs": before,
    }
    report_path = output / "report.json"
    write_json(report_path, report)
    try:
        verify_quality(settings, output)
        diagnostics = DiagnosticSettings.load(settings.root, settings.diagnostic_config)
        diagnostic_report = diagnose(diagnostics) / "report.json"
        spec = verified_spec(diagnostics)
        report["diagnostics"] = {
            "path": str(diagnostic_report),
            "sha256": file_hash(diagnostic_report),
            "fingerprint": spec["fingerprint"],
        }
        report["files"] = freeze_candidate(settings, output)
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


def verify_quality(settings: PreflightSettings, output: Path) -> None:
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


def freeze_candidate(settings: PreflightSettings, output: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        extracted = staging / "archive"
        extract_archive(settings.adapter_archive, extracted, settings.max_unpacked_bytes)
        submission = staging / "candidate"
        submission.mkdir()
        for name in referenced_files(settings.agent_dir, settings.agent_config):
            target = submission / (
                "agent.yaml" if settings.agent_dir / name == settings.agent_config else name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(settings.agent_dir / name, target)
        # Only adapters are reused; no historical agent settings enter the candidate.
        shutil.copytree(extracted / "adapters", submission / "adapters")
        shutil.copyfile(settings.evaluation_config, submission / "eval_config.yaml")
        return package_submission(
            submission,
            output / "submission.zip",
            validate_references,
            validation_scope="structural",
        )


def load_report(root: Path, path: Path) -> tuple[PreflightSettings, dict[str, Any]]:
    path = project_path(root, str(path))
    if not path.is_relative_to(root.resolve() / "artifacts/preflight"):
        raise ValueError("Expected a local preflight report")
    report = json.loads(path.read_text())
    if report.get("kind") != "preflight" or report.get("status") != "completed":
        raise ValueError("Preflight did not complete")
    settings = PreflightSettings.load(root, Path(report["config"]), report["profile"])
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
