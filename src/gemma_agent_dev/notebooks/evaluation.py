"""Build a private, self-contained Kaggle script from selected project sources."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from gemma_agent_dev.candidates.preflight import artifact_limit, load_report
from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.sources import EVALUATION_MODULES
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.config import EvaluationSettings, RuntimeSettings
from gemma_agent_dev.evaluation.results import load_run
from gemma_agent_dev.notebooks.bundle import payload as build_payload
from gemma_agent_dev.notebooks.bundle import write_notebook
from gemma_agent_dev.tasks.catalog import TaskCatalog


def prepare(
    root: Path,
    destination: Path,
    *,
    verify_only: bool = False,
    report_path: Path,
    cpu_report: Path | None = None,
    bundle_dataset: str | None = None,
) -> str:
    root = root.resolve()
    settings, report = load_report(root, report_path)
    selected = settings.evaluation_config
    selected_content = selected.read_bytes()
    evaluation = EvaluationSettings.load(selected)
    config = RuntimeSettings.load(settings.runtime_config).as_dict()
    diagnostics = DiagnosticSettings.load(root, settings.diagnostic_config)
    catalog = TaskCatalog.load(root, diagnostics.catalog_path)
    if settings.task_selection is not None:
        from gemma_agent_dev.diagnostics.qualification import load_selection

        config["task_ids"] = load_selection(root, settings.task_selection)["task_ids"]
    kernel_id = config["verification_kernel_id"] if verify_only else config["kernel_id"]
    if destination.exists():
        raise FileExistsError(destination)
    inputs = {
        "runtime.json": json.dumps(config).encode(),
        "candidate.zip": (root / report_path)
        .resolve()
        .parent.joinpath("submission.zip")
        .read_bytes(),
        "candidate.json": json.dumps(
            {
                "sha256": report["candidate_sha256"],
                "files": report["files"],
                "max_unpacked_bytes": settings.max_unpacked_bytes,
                "max_artifact_bytes": artifact_limit(root),
                "snapshots": {i: catalog.tasks[i].snapshot_sha256 for i in config["task_ids"]},
                "sources": {
                    i: {
                        k: asdict(catalog.tasks[i])[k]
                        for k in ("source_roots", "import_modules", "dependency_wheels")
                    }
                    for i in config["task_ids"]
                },
                "tasks_sha256": catalog.tasks_sha256,
            }
        ).encode(),
    }
    inputs.update(
        {
            "sandbox-wheels/" + wheel.name: wheel.read_bytes()
            for wheel in catalog.wheels(config["task_ids"])
        }
    )
    payload, files = build_payload(root, EVALUATION_MODULES, inputs)
    digest = hashlib.sha256(payload).hexdigest()
    if not verify_only:
        if cpu_report is None:
            raise ValueError("GPU preparation requires the completed CPU manifest")
        cpu = load_run(cpu_report.resolve().parent)
        if (
            cpu.kind != "verification"
            or not cpu.complete
            or cpu.manifest.get("source_sha256") != digest
            or cpu.manifest.get("candidate_sha256") != report["candidate_sha256"]
            or cpu.task_ids != config["task_ids"]
            or cpu.manifest.get("wheelhouse_hashes") != config["wheelhouse_hashes"]
        ):
            raise ValueError("CPU evidence does not match this candidate and source bundle")
        evidence = cpu.manifest.get("control_artifacts")
        if (
            not isinstance(evidence, dict)
            or not evidence
            or not any(k.endswith("origins.json") for k in evidence)
        ):
            raise ValueError("CPU source and control evidence is missing")
        for name, expected in evidence.items():
            path = (cpu.path / name).resolve()
            if not path.is_relative_to(cpu.path.resolve()) or file_hash(path) != expected:
                raise ValueError("CPU control artifacts changed")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    provenance = {
        "source_sha256": digest,
        "profile": settings.profile,
        "purpose": settings.purpose,
        "git_commit": commit,
        "candidate_sha256": report["candidate_sha256"],
        "preflight_sha256": hashlib.sha256((root / report_path).read_bytes()).hexdigest(),
        "evaluation_config": {
            "path": selected.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(selected_content).hexdigest(),
            "values": evaluation.as_dict(),
        },
    }
    write_notebook(
        destination,
        payload,
        provenance,
        config,
        kernel_id,
        verify_only=verify_only,
        diagnostic=False,
        bundle_dataset=bundle_dataset,
    )
    (destination / "source-bundle.zip").write_bytes(payload)
    write_json(
        destination / "source-manifest.json",
        {
            **provenance,
            "files": [p.relative_to(root).as_posix() for p in files],
        },
    )
    return digest
