"""Build a private, self-contained Kaggle script from selected project sources."""

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

from gemma_agent_dev.artifacts import file_hash, write_json
from gemma_agent_dev.config import EvaluationSettings, RuntimeSettings
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.preflight import load_report
from gemma_agent_dev.results import load_run
from gemma_agent_dev.task_catalog import TaskCatalog


def prepare(
    root: Path,
    destination: Path,
    *,
    verify_only: bool = False,
    report_path: Path,
    cpu_report: Path | None = None,
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
        from gemma_agent_dev.qualification import load_selection

        config["task_ids"] = load_selection(root, settings.task_selection)["task_ids"]
    kernel_id = config["verification_kernel_id"] if verify_only else config["kernel_id"]
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(
        [
            *[
                root / "src/gemma_agent_dev" / name
                for name in (
                    "__init__.py",
                    "runtime.py",
                    "config.py",
                    "artifacts.py",
                    "official.py",
                    "environment.py",
                    "results.py",
                    "candidates.py",
                    "sandbox_sources.py",
                    "provenance.py",
                    "test_evidence.py",
                    "processes.py",
                    "worker.py",
                )
            ],
            root / "kaggle/entry.py",
        ]
    )
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in {
            "configs/runtime.json": json.dumps(config).encode(),
            "candidate.zip": (root / report_path)
            .resolve()
            .parent.joinpath("submission.zip")
            .read_bytes(),
            "candidate.json": json.dumps(
                {
                    "sha256": report["candidate_sha256"],
                    "files": report["files"],
                    "max_unpacked_bytes": settings.max_unpacked_bytes,
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
        }.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        for wheel in catalog.wheels(config["task_ids"]):
            info = zipfile.ZipInfo("sandbox-wheels/" + wheel.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, wheel.read_bytes())
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Symlink in source bundle: {path}")
            info = zipfile.ZipInfo(path.relative_to(root).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info,
                path.read_bytes(),
            )
    payload = content.getvalue()
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
    script = (root / "kaggle/entry.py").read_text() + (
        "\nif __name__ == '__main__':\n"
        f"    main({base64.b64encode(payload).decode()!r}, {provenance!r}, "
        f"verify_only={verify_only!r}, timeout_seconds={config['job_timeout_seconds']!r}, "
        f"progress_interval_seconds={config['progress_interval_seconds']!r})\n"
    )
    destination.mkdir(parents=True)
    (destination / "run.py").write_text(script, encoding="utf-8")
    write_json(
        destination / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_id.split("/")[-1].replace("-", " ").title(),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": not verify_only,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [config["wheelhouse_source"]],
            "competition_sources": [config["competition"]],
            "kernel_sources": [],
            "model_sources": [] if verify_only else [config["model_source"]],
            "docker_image": config["docker_image"],
            **({} if verify_only else {"machine_shape": config["machine_shape"]}),
        },
    )
    write_json(
        destination / "source-manifest.json",
        {
            **provenance,
            "files": [p.relative_to(root).as_posix() for p in files],
        },
    )
    return digest
