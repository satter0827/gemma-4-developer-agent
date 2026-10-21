"""Build CPU diagnostic notebooks from verified local evidence."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.common.junit import diagnostic_evidence
from gemma_agent_dev.common.sources import DIAGNOSTIC_MODULES
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.notebooks.bundle import payload as build_payload
from gemma_agent_dev.notebooks.bundle import write_notebook
from gemma_agent_dev.tasks.catalog import TaskCatalog


def verified_spec(settings: DiagnosticSettings) -> dict[str, Any]:
    from gemma_agent_dev.diagnostics.local import check_prepared

    check_prepared(settings)
    receipt = json.loads((settings.cache_dir / "local-success.json").read_text())
    local_report = Path(receipt["report"])
    if (
        not local_report.resolve().is_relative_to(settings.output_dir)
        or receipt["fingerprint"] != settings.fingerprint()
        or file_hash(local_report) != receipt["sha256"]
    ):
        raise ValueError("Successful local diagnostic evidence is missing or changed")
    local = json.loads(local_report.read_text())
    if (
        local["status"] != "completed"
        or local["fingerprint"] != settings.fingerprint()
        or len(local["cases"])
        != sum(len(t.patches(settings.read_tasks()[t.instance_id])) for t in settings.tasks)
    ):
        raise ValueError("All local diagnostic cases must complete before preparing Kaggle")
    tasks = settings.read_tasks()
    cases: list[dict[str, Any]] = []
    for task in settings.tasks:
        for name, patch in task.patches(tasks[task.instance_id]).items():
            cases.append(
                {
                    "task_id": task.instance_id,
                    "case": name,
                    "patch": patch,
                    "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                    "expected_pass": task.expected_outcome(name) == "passed",
                    "expected_outcome": task.expected_outcome(name),
                }
            )
    if any(not r["matched"] for r in local["cases"]):
        raise ValueError("Local diagnostics contain failed cases")
    for saved, case in zip(local["cases"], cases, strict=True):
        if any(
            saved[k] != case[k]
            for k in ("task_id", "case", "patch_sha256", "expected_pass", "expected_outcome")
        ):
            raise ValueError("Local diagnostic cases differ from configured inputs")
        directory = local_report.parent / case["task_id"] / case["case"]
        files = inventory(directory)
        files.pop("case.json", None)
        if (
            files != saved["artifacts"]
            or json.loads((directory / "case.json").read_text()) != saved
        ):
            raise ValueError("Local diagnostic artifacts are missing or changed")
        outcome, counts = diagnostic_evidence(
            saved["exit_code"], directory / "junit.xml", case["expected_outcome"]
        )
        if (
            counts != saved["junit"]
            or saved["passed"] != case["expected_pass"]
            or outcome != saved["outcome"]
        ):
            raise ValueError("Local diagnostic test evidence is inconsistent")
    spec = {
        "fingerprint": settings.fingerprint(),
        "task_ids": [t.instance_id for t in settings.tasks],
        "tasks_sha256": settings.tasks_sha256,
        "snapshots": {t.instance_id: t.snapshot_sha256 for t in settings.tasks},
        "sources": {
            t.instance_id: {
                k: asdict(t)[k] for k in ("source_roots", "import_modules", "dependency_wheels")
            }
            for t in settings.tasks
        },
        "runtime": settings.runtime.as_dict(),
        "timeout_seconds": settings.timeout_seconds,
        "lock_sha256": file_hash(settings.lock_file),
        "cases": cases,
        "local_report_sha256": receipt["sha256"],
    }
    return spec


def prepare(
    root: Path, destination: Path, config: Path, *, bundle_dataset: str | None = None
) -> str:
    settings = DiagnosticSettings.load(root, config)
    spec = verified_spec(settings)
    if destination.exists():
        raise FileExistsError(destination)
    inputs = {"diagnostics.json": json.dumps(spec, sort_keys=True).encode()}
    inputs.update(
        {
            "sandbox-wheels/" + wheel.name: wheel.read_bytes()
            for wheel in TaskCatalog.load(root, settings.catalog_path).wheels(
                [t.instance_id for t in settings.tasks]
            )
        }
    )
    payload, files = build_payload(root, DIAGNOSTIC_MODULES, inputs)
    digest = hashlib.sha256(payload).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    provenance = {"source_sha256": digest, "git_commit": revision}
    write_notebook(
        destination,
        payload,
        provenance,
        settings.runtime.as_dict(),
        settings.kernel_id,
        verify_only=True,
        diagnostic=True,
        bundle_dataset=bundle_dataset,
    )
    (destination / "source-bundle.zip").write_bytes(payload)
    write_json(
        destination / "source-manifest.json",
        {
            **provenance,
            "diagnostics": spec,
            "files": [p.relative_to(root).as_posix() for p in files],
        },
    )
    return digest
