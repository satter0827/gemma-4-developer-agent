"""Validate portable experiment observations and export only reviewed public metadata."""

import json
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.config import mapping, project_path
from gemma_agent_dev.evaluation.experiment_results import summarize
from gemma_agent_dev.evaluation.record_schema import CONDITION_FIELDS, RECORD_SCHEMA_VERSION, review


def export(root: Path, session: Path, review_path: Path, destination: Path, max_bytes: int) -> Path:
    summary = summarize(session, max_bytes=max_bytes)
    sealed = json.loads(session.read_text())
    if "definition" not in sealed:
        raise ValueError("Export requires a frozen experiment definition")
    note = mapping(
        yaml.safe_load(project_path(root, str(review_path)).read_text()),
        {"record_id", "interpretation", "decision", "parent_candidates", "assessments", "prompts"},
        "reviewed experiment note",
    )
    review(note)
    definition = sealed["definition"]
    runtime = definition["runtime"]
    record = {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "record_id": note["record_id"],
        "experiment_id": sealed["experiment_id"],
        "hypothesis": definition["hypothesis"],
        "status": "evaluated" if summary["status"] == "completed" else "incomplete",
        "task_ids": definition["task_ids"],
        "definition_sha256": sealed["definition_sha256"],
        "source_sha256": sealed["source_sha256"],
        "candidate_hashes": sealed["variants"],
        "metrics": {**summary["counts"], "cell_errors": summary["cell_errors"]},
        "interpretation": note["interpretation"],
        "prompts": [],
        "observations": summary["observations"],
        "conditions": {
            **{k: v for k, v in definition["evaluation"].items() if k in CONDITION_FIELDS},
            "sdk_version": runtime["packages"]["swegemma"],
            "verification_timeout_seconds": runtime["verification_timeout_seconds"],
            "max_model_len": runtime["vllm"]["max_model_len"],
        },
        "quota_hours": summary["quota_hours"],
        **{k: note[k] for k in ("decision", "parent_candidates", "assessments")},
    }
    for item in note["prompts"]:
        mapping(item, {"prompt_id", "path"}, "reviewed prompt")
        path = project_path(root, item["path"])
        digest = file_hash(path)
        relative = path.relative_to(root / "agent").as_posix()
        if not any(files.get(relative) == digest for files in sealed["candidate_files"].values()):
            raise ValueError("Reviewed prompt does not belong to an evaluated candidate")
        record["prompts"].append({**item, "sha256": digest})
    return save(root, destination, record)


def save(root: Path, destination: Path, record: dict[str, Any]) -> Path:
    from gemma_agent_dev.evaluation.tracking import load_records

    destination = project_path(root, str(destination))
    if (
        not destination.is_relative_to(root / "docs/records/experiments")
        or destination.suffix != ".json"
    ):
        raise ValueError("Public records must belong to docs/records/experiments")
    # Validate before creating a public file, using the same parser as MLflow projection.
    load_records(root, [], records=[record])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return destination
