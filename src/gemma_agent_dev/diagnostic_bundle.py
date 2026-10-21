"""Build CPU diagnostic notebooks from verified local evidence."""

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.task_catalog import TaskCatalog
from gemma_agent_dev.test_evidence import diagnostic_evidence


def verified_spec(settings: DiagnosticSettings) -> dict[str, Any]:
    from gemma_agent_dev.diagnostics import check_prepared

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


def prepare(root: Path, destination: Path, config: Path) -> str:
    settings = DiagnosticSettings.load(root, config)
    spec = verified_spec(settings)
    if destination.exists():
        raise FileExistsError(destination)
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "__init__.py",
            "artifacts.py",
            "config.py",
            "environment.py",
            "diagnostic_kaggle.py",
            "test_evidence.py",
            "official.py",
            "sandbox_sources.py",
            "provenance.py",
            "processes.py",
            "worker.py",
        ):
            path = root / "src/gemma_agent_dev" / name
            info = zipfile.ZipInfo(path.relative_to(root).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        for wheel in TaskCatalog.load(root, settings.catalog_path).wheels(
            [t.instance_id for t in settings.tasks]
        ):
            info = zipfile.ZipInfo("sandbox-wheels/" + wheel.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, wheel.read_bytes())
        info = zipfile.ZipInfo("diagnostics.json")
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(spec, sort_keys=True))
    payload = content.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    provenance = {"source_sha256": digest, "git_commit": revision}
    script = (root / "kaggle/entry.py").read_text() + (
        "\nif __name__ == '__main__':\n"
        f"    main({base64.b64encode(payload).decode()!r}, {provenance!r}, "
        f"diagnostic=True, timeout_seconds={settings.runtime.job_timeout_seconds!r}, "
        f"progress_interval_seconds={settings.runtime.progress_interval_seconds!r})\n"
    )
    destination.mkdir(parents=True)
    (destination / "run.py").write_text(script)
    write_json(destination / "source-manifest.json", {**provenance, "diagnostics": spec})
    runtime = settings.runtime
    write_json(
        destination / "kernel-metadata.json",
        {
            "id": settings.kernel_id,
            "title": settings.kernel_id.split("/")[-1].replace("-", " ").title(),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [runtime.wheelhouse_source],
            "competition_sources": [runtime.competition],
            "model_sources": [],
            "kernel_sources": [],
            "docker_image": runtime.docker_image,
        },
    )
    return digest
