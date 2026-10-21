"""Generate bounded optimization bundles and verify them without remote execution."""

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import extract_archive
from gemma_agent_dev.candidates.preflight import artifact_limit
from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.common.sources import OPTIMIZATION_MODULES
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.control_evidence import verify_saved_controls
from gemma_agent_dev.evaluation.experiments import content_hash
from gemma_agent_dev.evaluation.optimization_limits import SearchLimits
from gemma_agent_dev.evaluation.optimization_preparation import GepaEnvironment, load_preparation
from gemma_agent_dev.evaluation.sdk_checks import fingerprint, installed_state
from gemma_agent_dev.evaluation.sdk_checks import settings as sdk_settings
from gemma_agent_dev.notebooks.bundle import payload, write_notebook
from gemma_agent_dev.tasks.catalog import TaskCatalog


def load_cpu(path: Path, digest: str, definition_hash: str, candidate_hash: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("kind") != "optimization-verification"
        or report.get("status") != "completed"
        or report.get("source_sha256") != digest
        or report.get("definition_sha256") != definition_hash
        or report.get("candidate", {}).get("sha256") != candidate_hash
        or not report.get("improved")
        or not report.get("candidates")
        or report.get("adopted") is not False
    ):
        raise ValueError("CPU proof does not match the optimization bundle")
    if content_hash(report["definition"]) != definition_hash:
        raise ValueError("CPU proof definition changed")
    expected_ids = (
        report["definition"]["search_task_ids"] + report["definition"]["selection_task_ids"]
    )
    if (
        report["task_ids"] != expected_ids
        or any(
            item["task_id"] not in expected_ids or item["candidate_id"] not in report["candidates"]
            for item in report["evaluations"]
        )
        or report["best_candidate_id"] not in report["candidates"]
    ):
        raise ValueError("CPU optimization task or candidate identity changed")
    limits = SearchLimits.parse(report["definition"]["limits"])
    if (
        report["metric_calls"] != len(report["evaluations"])
        or not 0 < report["metric_calls"] <= limits.max_metric_calls
        or report["reflection_calls"] != limits.max_reflection_calls
    ):
        raise ValueError("CPU optimization call counts disagree")
    verify_saved_controls(path.parent / "controls", report["controls"], report["task_ids"])
    for name, item in report["candidates"].items():
        folder = (path.parent / "candidates" / name).resolve()
        if (
            not folder.is_relative_to(path.parent.resolve())
            or file_hash(folder / "submission.zip") != item["sha256"]
            or inventory(folder / "submission") != item["files"]
        ):
            raise ValueError("Verified optimization candidate changed")
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
    optimization, prepared = load_preparation(root, report_path)
    environment = GepaEnvironment.load(root, optimization.environment)
    environment.wheel()
    diagnostics = DiagnosticSettings.load(root, optimization.diagnostic_config)
    catalog = TaskCatalog.load(root, diagnostics.catalog_path)
    task_ids = optimization.search_task_ids + optimization.selection_task_ids
    spec = {
        "definition": prepared["definition"],
        "definition_sha256": prepared["definition_sha256"],
        "candidate": prepared["candidate"],
        "tasks_sha256": catalog.tasks_sha256,
        "snapshots": {t: catalog.tasks[t].snapshot_sha256 for t in task_ids},
        "sources": {
            t: {
                k: asdict(catalog.tasks[t])[k]
                for k in ("source_roots", "import_modules", "dependency_wheels")
            }
            for t in task_ids
        },
        "native_controls": prepared["controls"],
        "max_artifact_bytes": artifact_limit(root),
        "gepa": {"wheel_name": environment.wheel_name, "wheel_sha256": environment.wheel_sha256},
    }
    inputs = {
        "optimization.json": json.dumps(spec).encode(),
        "seed/submission.zip": (report_path.parent / "seed/submission.zip").read_bytes(),
        "optimization-wheels/" + environment.wheel_name: (
            environment.wheel_cache / environment.wheel_name
        ).read_bytes(),
        **{"sandbox-wheels/" + p.name: p.read_bytes() for p in catalog.wheels(task_ids)},
    }
    content, paths = payload(root, OPTIMIZATION_MODULES, inputs)
    digest = hashlib.sha256(content).hexdigest()
    if not verify_only:
        if optimization.gpu_budget is None:
            raise ValueError("Optimization GPU time and quota must be explicitly configured")
        if cpu_report is None:
            raise ValueError("Optimization GPU bundles require exact-bundle CPU proof")
        load_cpu(cpu_report, digest, prepared["definition_sha256"], prepared["candidate"]["sha256"])
        if quota is None or not optimization.gpu_budget.admit(**quota)["allowed"]:
            raise ValueError("Observed quota cannot admit optimization")
    runtime = dict(prepared["definition"]["runtime"])
    if verify_only:
        runtime["job_timeout_seconds"] = optimization.limits.cpu_timeout_seconds
    else:
        assert optimization.gpu_budget is not None
        runtime["job_timeout_seconds"] = optimization.gpu_budget.job_timeout_seconds
    provenance = {
        "source_sha256": digest,
        "experiment_id": optimization.optimization_id,
        "definition_sha256": prepared["definition_sha256"],
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
    return digest


def verify_local(root: Path, bundle: Path, sdk_config: Path) -> Path:
    data = sdk_settings(root, sdk_config)
    receipt = json.loads((data["environment"] / "prepared.json").read_text())
    if (
        receipt["inputs"] != fingerprint(data)
        or receipt["installed"] != installed_state(root, data)
        or any(
            file_hash(data["wheel_cache"] / name) != digest
            for name, digest in receipt["wheels"].items()
        )
    ):
        raise ValueError("SDK CPU environment changed; prepare again")
    metadata = json.loads((bundle / "source-manifest.json").read_text())
    optimization, prepared = load_preparation(root, Path(metadata["preparation_path"]))
    environment = GepaEnvironment.load(root, optimization.environment)
    environment.verify(root)
    if file_hash(bundle / "source-bundle.zip") != metadata["source_sha256"]:
        raise ValueError("Optimization source bundle changed")
    with tempfile.TemporaryDirectory(dir=root / "artifacts/tmp") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(bundle / "source-bundle.zip") as archive:
            unpacked_bytes = sum(item.file_size for item in archive.infolist())
        extract_archive(bundle / "source-bundle.zip", extracted, unpacked_bytes)
        bindings = {
            "output": str(root / "artifacts/optimization-verification"),
            "sdk_python": str(data["environment"] / "bin/python"),
            "native_controls": str(root / prepared["controls"]["path"]),
        }
        result = command(
            [
                str(environment.environment / "bin/python"),
                "-m",
                "gemma_agent_dev.evaluation.optimization_worker",
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
            timeout=optimization.limits.cpu_timeout_seconds,
        )
        report_path = Path(result.strip().splitlines()[-1])
    load_cpu(
        report_path,
        metadata["source_sha256"],
        metadata["definition_sha256"],
        prepared["candidate"]["sha256"],
    )
    return report_path
