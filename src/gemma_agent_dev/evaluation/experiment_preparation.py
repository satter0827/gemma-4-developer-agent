"""Freeze two candidates and CPU evidence before generating an experiment bundle."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import extract_archive
from gemma_agent_dev.candidates.changes import validate_change
from gemma_agent_dev.candidates.preflight import (
    PreflightSettings,
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
from gemma_agent_dev.common.config import project_path
from gemma_agent_dev.common.events import logged
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import check_prepared, diagnose
from gemma_agent_dev.evaluation.control_evidence import validate_control_set
from gemma_agent_dev.evaluation.experiments import Experiment
from gemma_agent_dev.tasks.pools import TaskPool


def selected_settings(root: Path, experiment: Experiment) -> dict[str, PreflightSettings]:
    base = PreflightSettings.load(root, experiment.preflight_config, "smoke")
    return {
        name: replace(
            base,
            profile=name,
            purpose="experiment",
            agent_config=entry,
            evaluation_config=experiment.evaluation_config,
            diagnostic_config=experiment.diagnostic_config,
            runtime_config=experiment.runtime_config,
            task_selection=None,
        )
        for name, entry in experiment.variants.items()
    }


@logged("experiment.prepare")
def prepare(root: Path, config: Path, *, quality_report: Path | None = None) -> Path:
    experiment = Experiment.load(root, config)
    check_installation(root)
    settings = selected_settings(root, experiment)
    diagnostic = DiagnosticSettings.load(root, experiment.diagnostic_config)
    pool = TaskPool.load(root, experiment.pool)
    if (
        pool.tasks_sha256 != diagnostic.tasks_sha256
        or pool.sdk_version != diagnostic.runtime.packages["swegemma"]
    ):
        raise ValueError("Task pool and CPU controls use different data or SDK versions")
    if [t.instance_id for t in diagnostic.tasks] != experiment.task_ids:
        raise ValueError("CPU controls must match the exact frozen experiment order")
    check_prepared(diagnostic)
    definition = experiment.definition(root)
    before = {name: value.inputs() for name, value in settings.items()}
    output = root / "artifacts/experiments" / new_run_id()
    output.mkdir(parents=True)
    path = output / "preparation.json"
    report: dict[str, Any] = {
        "kind": "experiment-preparation",
        "status": "running",
        "config": str(config.relative_to(root) if config.is_absolute() else config),
        "definition": definition,
        "definition_sha256": content_hash(definition),
        "inputs": before,
        "variants": {},
    }
    write_json(path, report)
    try:
        if quality_report is None:
            verify_quality(next(iter(settings.values())), output)
        else:
            from gemma_agent_dev.candidates.quality import reuse

            report["quality"] = reuse(root, quality_report)
        diagnostic_report = diagnose(diagnostic) / "report.json"
        report["controls"] = {
            "path": str(diagnostic_report.relative_to(root)),
            "sha256": file_hash(diagnostic_report),
        }
        for name, value in settings.items():
            folder = output / name
            folder.mkdir()
            files = freeze_candidate(value, folder)
            report["variants"][name] = {
                "files": files,
                "sha256": file_hash(folder / "submission.zip"),
                "max_unpacked_bytes": value.max_unpacked_bytes,
            }
        if experiment.change is not None:
            import tempfile

            with tempfile.TemporaryDirectory(dir=output) as temporary:
                folders = {name: Path(temporary) / name for name in settings}
                for name, folder in folders.items():
                    extract_archive(
                        output / name / "submission.zip", folder, settings[name].max_unpacked_bytes
                    )
                report["change_check"] = validate_change(
                    experiment.change,
                    folders,
                    {name: v["files"] for name, v in report["variants"].items()},
                )
        if definition != experiment.definition(root) or before != {
            name: value.inputs() for name, value in settings.items()
        }:
            raise ValueError("Experiment inputs changed during preparation")
        report["status"] = "completed"
        report["artifacts"] = inventory(output)
        report["artifacts"].pop("preparation.json", None)
        write_json(path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, report)
        raise
    return path


def load_preparation(root: Path, path: Path) -> tuple[Experiment, dict[str, Any]]:
    path = project_path(root, str(path))
    if not path.is_relative_to(root / "artifacts/experiments"):
        raise ValueError("Expected a local experiment preparation")
    report = json.loads(path.read_text())
    experiment = Experiment.load(root, Path(report["config"]))
    if (
        report["kind"] != "experiment-preparation"
        or report["status"] != "completed"
        or report["definition"] != experiment.definition(root)
        or report["definition_sha256"] != content_hash(report["definition"])
    ):
        raise ValueError("Experiment preparation changed or did not complete")
    if report["inputs"] != {
        name: value.inputs() for name, value in selected_settings(root, experiment).items()
    }:
        raise ValueError("Experiment source inputs changed")
    artifacts = inventory(path.parent)
    artifacts.pop("preparation.json", None)
    if artifacts != report["artifacts"]:
        raise ValueError("Prepared candidates changed")
    controls = project_path(root, report["controls"]["path"])
    if file_hash(controls) != report["controls"]["sha256"]:
        raise ValueError("CPU controls changed")
    data = json.loads(controls.read_text())
    validate_control_set(data, experiment.task_ids)
    if data["status"] != "completed":
        raise ValueError("CPU controls did not complete")
    for case in data["cases"]:
        folder = controls.parent / case["task_id"] / case["case"]
        files = inventory(folder)
        files.pop("case.json", None)
        if (
            not case["matched"]
            or files != case["artifacts"]
            or json.loads((folder / "case.json").read_text()) != case
        ):
            raise ValueError("CPU control artifacts changed")
    return experiment, report
