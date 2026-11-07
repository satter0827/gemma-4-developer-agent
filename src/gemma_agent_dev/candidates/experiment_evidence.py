"""Qualify original archives from complete, matching paired experiment evidence."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.evidence import (
    Candidate,
    archive_files,
    checked_candidate,
    content_id,
)
from gemma_agent_dev.common.artifacts import content_hash, file_hash
from gemma_agent_dev.evaluation.analysis import SCORABLE_TERMINATIONS
from gemma_agent_dev.evaluation.control_evidence import session_controls
from gemma_agent_dev.evaluation.experiment_results import load_cpu, summarize


def saved_archive(path: Path, report: dict[str, Any], variant: str) -> Path:
    if set(report.get("candidate_archives", {})) != set(report["variants"]):
        raise ValueError("Missing immutable experiment archives")
    item = report["candidate_archives"][variant]
    if (
        not isinstance(item, dict)
        or set(item) != {"path", "sha256"}
        or not isinstance(item["path"], str)
    ):
        raise ValueError("Invalid archive reference")
    raw: Path = path.parent / item["path"]
    archive = raw.resolve()
    if (
        not archive.is_relative_to(path.parent.resolve())
        or any(p.is_symlink() for p in (raw, *raw.parents) if p.is_relative_to(path.parent))
        or item["sha256"] != report["variants"][variant]
        or file_hash(archive) != item["sha256"]
    ):
        raise ValueError("Experiment archive changed")
    return archive


def qualify_experiment(
    cpu_path: Path,
    gpu_path: Path,
    variant: str,
    source: Mapping[str, Any],
    max_bytes: int,
) -> Candidate:
    report = json.loads(gpu_path.read_text())
    summary = summarize(gpu_path, max_bytes=max_bytes)
    definition = report["definition"]
    if (
        report["status"] != "completed"
        or content_hash(definition) != report["definition_sha256"]
        or report["task_ids"] != definition["task_ids"]
        or report["actual_order"] != report["planned_order"]
        or variant not in report["variants"]
        or any(
            row["status"] != "completed" or row["termination"] not in SCORABLE_TERMINATIONS
            for row in summary["observations"]
        )
    ):
        raise ValueError("Registration requires complete, valid paired results")
    cpu = load_cpu(
        cpu_path,
        report["source_sha256"],
        report["definition_sha256"],
        report["task_ids"],
        report["variants"],
    )
    if (
        cpu["definition"] != definition
        or cpu["candidate_files"] != report["candidate_files"]
        or "evidence_inputs" not in report
        or cpu.get("evidence_inputs") != report["evidence_inputs"]
        or cpu.get("sdk_packages") != report["sdk_packages"]
    ):
        raise ValueError("CPU/GPU experiment inputs differ")
    for name in report["variants"]:
        session_controls(gpu_path, report, name)
        saved_archive(cpu_path, cpu, name)
        saved_archive(gpu_path, report, name)
    archive = saved_archive(gpu_path, report, variant)
    files = archive_files(archive, max_bytes)
    if files != report["candidate_files"][variant]:
        raise ValueError("Experiment archive files disagree")
    cells = {c["task_id"]: c["sha256"] for c in report["cells"] if c["variant"] == variant}
    if set(cells) != set(report["task_ids"]):
        raise ValueError("Missing candidate evaluation cells")
    return checked_candidate(
        {
            "id": content_id(files),
            "archive_sha256": report["variants"][variant],
            "files": files,
            "source": dict(source),
            "evidence": {
                "kind": "paired-experiment",
                "cpu_manifest_sha256": file_hash(cpu_path),
                "gpu_manifest_sha256": file_hash(gpu_path),
                "source_sha256": report["source_sha256"],
                "cpu_run_id": cpu_path.parent.name,
                "gpu_run_id": gpu_path.parent.name,
                "definition_sha256": report["definition_sha256"],
                "variant": variant,
                "cell_manifests": cells,
            },
        }
    )
