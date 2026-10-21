"""Build the same experiment source bundle for CPU proof and later GPU use."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.common.sources import EXPERIMENT_MODULES
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.evaluation.experiment_preparation import load_preparation
from gemma_agent_dev.evaluation.sdk_checks import (
    fingerprint,
    installed_state,
)
from gemma_agent_dev.evaluation.sdk_checks import (
    settings as sdk_settings,
)
from gemma_agent_dev.notebooks.bundle import payload as build_payload
from gemma_agent_dev.notebooks.bundle import write_notebook
from gemma_agent_dev.tasks.catalog import TaskCatalog


def load_cpu(
    path: Path, digest: str, definition_hash: str, task_ids: list[str], variants: dict[str, str]
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("kind") != "experiment-verification"
        or report.get("status") != "completed"
        or report.get("source_sha256") != digest
        or report.get("definition_sha256") != definition_hash
        or report.get("task_ids") != task_ids
        or report.get("variants") != variants
        or set(report.get("controls", {})) != set(variants)
    ):
        raise ValueError("CPU proof does not match the complete experiment bundle")
    for variant, files in report["controls"].items():
        if "controls.json" not in files or not any(name.endswith("origins.json") for name in files):
            raise ValueError("CPU proof is missing source and control evidence")
        for name, expected in files.items():
            target = (path.parent / "controls" / variant / name).resolve()
            if not target.is_relative_to(path.parent.resolve()) or file_hash(target) != expected:
                raise ValueError("CPU proof artifacts changed")
        records = json.loads((path.parent / "controls" / variant / "controls.json").read_text())
        if [r["instance_id"] for r in records] != task_ids or any(
            r["unmodified"]["resolved"] is not False
            or r["reference"]["resolved"] is not True
            or r["reference"].get("error")
            or r["unmodified"].get("error")
            for r in records
        ):
            raise ValueError("CPU controls are incomplete or do not discriminate fixes")
    return dict(report)


def prepare(
    root: Path,
    destination: Path,
    *,
    report_path: Path,
    verify_only: bool,
    cpu_report: Path | None,
    quota: dict[str, Any] | None = None,
) -> str:
    experiment, report = load_preparation(root, report_path)
    diagnostic = DiagnosticSettings.load(root, experiment.diagnostic_config)
    catalog = TaskCatalog.load(root, diagnostic.catalog_path)
    runtime = RuntimeSettings.load(experiment.runtime_config).as_dict()
    runtime.update(
        task_ids=experiment.task_ids, job_timeout_seconds=experiment.session.job_timeout_seconds
    )
    spec = {
        "native_controls_sha256": report["controls"]["sha256"],
        "local_diagnostic_config": str(experiment.diagnostic_config.relative_to(root)),
        "definition": report["definition"],
        "definition_sha256": report["definition_sha256"],
        "variants": report["variants"],
        "tasks_sha256": catalog.tasks_sha256,
        "snapshots": {i: catalog.tasks[i].snapshot_sha256 for i in experiment.task_ids},
        "sources": {
            i: {
                k: asdict(catalog.tasks[i])[k]
                for k in ("source_roots", "import_modules", "dependency_wheels")
            }
            for i in experiment.task_ids
        },
    }
    inputs = {
        "runtime.json": json.dumps(runtime).encode(),
        "experiment.json": json.dumps(spec).encode(),
    }
    inputs.update(
        {
            name + "/submission.zip": (report_path.parent / name / "submission.zip").read_bytes()
            for name in experiment.variants
        }
    )
    inputs.update(
        {"sandbox-wheels/" + p.name: p.read_bytes() for p in catalog.wheels(experiment.task_ids)}
    )
    content, paths = build_payload(root, EXPERIMENT_MODULES, inputs)
    digest = hashlib.sha256(content).hexdigest()
    if not verify_only:
        if cpu_report is None:
            raise ValueError("GPU experiments require exact-bundle CPU proof")
        load_cpu(
            cpu_report,
            digest,
            report["definition_sha256"],
            experiment.task_ids,
            {name: value["sha256"] for name, value in report["variants"].items()},
        )
        if quota is None:
            raise ValueError("Observed GPU quota is required; unknown quota blocks preparation")
        decision = experiment.session.budget().admit(**quota)
        if not decision["allowed"]:
            raise ValueError("Observed GPU quota cannot admit the bounded experiment")
    provenance = {
        "source_sha256": digest,
        "experiment_id": experiment.experiment_id,
        "definition_sha256": report["definition_sha256"],
        "preparation_sha256": file_hash(report_path),
    }
    write_notebook(
        destination,
        content,
        provenance,
        runtime,
        runtime["verification_kernel_id"] if verify_only else runtime["kernel_id"],
        verify_only=verify_only,
        diagnostic=False,
    )
    (destination / "source-bundle.zip").write_bytes(content)
    write_json(
        destination / "source-manifest.json",
        {
            **provenance,
            "preparation_path": str(report_path.relative_to(root)),
            "files": [p.relative_to(root).as_posix() for p in paths],
        },
    )
    if quota is not None:
        write_json(destination / "quota.json", experiment.session.budget().admit(**quota))
    return digest


def verify_local(root: Path, bundle: Path, sdk_config: Path) -> Path:
    data = sdk_settings(root, sdk_config)
    receipt = json.loads((data["environment"] / "prepared.json").read_text())
    if receipt["inputs"] != fingerprint(data) or receipt["installed"] != installed_state(
        root, data
    ):
        raise ValueError("Official CPU environment changed; prepare again")
    metadata = json.loads((bundle / "source-manifest.json").read_text())
    _, prepared = load_preparation(root, Path(metadata["preparation_path"]))
    content = (bundle / "source-bundle.zip").read_bytes()
    if hashlib.sha256(content).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Experiment source bundle changed")
    with tempfile.TemporaryDirectory(dir=root / "artifacts/tmp") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            # These are locally generated closed bundles; still reject path traversal and links.
            from gemma_agent_dev.candidates.archive import extract_archive

            extract_archive(
                bundle / "source-bundle.zip",
                extracted,
                sum(i.file_size for i in archive.infolist()),
            )
        spec = json.loads((extracted / "experiment.json").read_text())
        task_data = extracted / "data"
        task_data.mkdir()
        diagnostic = DiagnosticSettings.load(root, Path(spec["local_diagnostic_config"]))
        (task_data / "tasks.jsonl").symlink_to(diagnostic.tasks_file)
        (task_data / "snapshots").symlink_to(
            diagnostic.cache_dir / "snapshots", target_is_directory=True
        )
        bindings = {
            "data": str(task_data),
            "output": str(root / "artifacts/experiment-verification"),
            "native_controls": str(root / prepared["controls"]["path"]),
        }
        result = command(
            [
                str(data["environment"] / "bin/python"),
                "-m",
                "gemma_agent_dev.evaluation.session_worker",
                str(extracted),
                json.dumps(metadata),
                json.dumps(bindings),
            ],
            cwd=extracted,
            env={
                **os.environ,
                "PYTHONPATH": str(extracted / "src"),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "OTEL_SDK_DISABLED": "true",
                "PIP_NO_INDEX": "1",
                "UV_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            timeout=spec["definition"]["session"]["job_timeout_seconds"],
        )
        report_path = Path(result.strip().splitlines()[-1])
    load_cpu(
        report_path,
        metadata["source_sha256"],
        metadata["definition_sha256"],
        spec["definition"]["task_ids"],
        {name: value["sha256"] for name, value in spec["variants"].items()},
    )
    return report_path
