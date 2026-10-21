"""Verify archived evidence and keep only publishable submission metadata."""

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash
from gemma_agent_dev.candidates import extract_archive, validate_references
from gemma_agent_dev.results import load_run, read_object
from gemma_agent_dev.submission_config import timestamp
from gemma_agent_dev.task_catalog import checksum

SCHEMA_VERSION = 1
EVIDENCE_FIELDS = (
    "source_sha256",
    "candidate_sha256",
    "task_ids",
    "submission_files",
    "tasks_sha256",
    "snapshots",
    "wheelhouse_hashes",
    "source_setup",
    "evaluation",
    "runtime_config",
    "vllm_config",
    "verification_timeout_seconds",
    "dependencies",
)


def content_id(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def archive_files(path: Path, max_bytes: int) -> dict[str, str]:
    if path.stat().st_size > max_bytes:
        raise ValueError("Submission archive is too large")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        extract_archive(path, root, max_bytes)
        return validate_references(root)


def checked_candidate(value: Any) -> dict[str, Any]:
    required = {"id", "archive_sha256", "files", "source", "evidence"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid candidate capsule")
    checksum(value["archive_sha256"])
    files = value["files"]
    if not isinstance(files, dict) or not files or "agent.yaml" not in files:
        raise ValueError("Missing candidate inventory")
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid inventory path")
        checksum(digest)
    if value["id"] != content_id(files):
        raise ValueError("Candidate content identity changed")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"owner", "kernel", "version", "path"}:
        raise ValueError("Invalid immutable Notebook source")
    for name in ("owner", "kernel"):
        if not isinstance(source[name], str) or not re.fullmatch(r"[\w-]+", source[name]):
            raise ValueError("Invalid Notebook name")
    if type(source["version"]) is not int or source["version"] <= 0:
        raise ValueError("Notebook version must be positive")
    if (
        not isinstance(source["path"], str)
        or not source["path"].endswith("/submission.zip")
        or Path(source["path"]).is_absolute()
        or ".." in Path(source["path"]).parts
    ):
        raise ValueError("Invalid Notebook output path")
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {
        "cpu_manifest_sha256",
        "gpu_manifest_sha256",
        "source_sha256",
        "cpu_run_id",
        "gpu_run_id",
    }:
        raise ValueError("Missing archived qualification evidence")
    for name, item in evidence.items():
        if name.endswith("sha256"):
            checksum(item)
        elif not isinstance(item, str) or not re.fullmatch(r"[\w-]+", item):
            raise ValueError("Invalid evidence run ID")
    return value


def qualify_candidate(
    cpu_path: Path, gpu_path: Path, source: dict[str, Any], max_bytes: int
) -> dict[str, Any]:
    cpu, gpu = load_run(cpu_path.parent), load_run(gpu_path.parent, require_evaluation=True)
    if cpu.kind != "verification" or not cpu.complete or not gpu.complete:
        raise ValueError("Candidate requires complete CPU and GPU evidence")
    if (
        cpu.manifest.get("controls_passed") is not True
        or gpu.manifest.get("controls_passed") is not True
    ):
        raise ValueError("Reference controls did not pass")
    for name in EVIDENCE_FIELDS:
        if name not in cpu.manifest or cpu.manifest[name] != gpu.manifest.get(name):
            raise ValueError(f"CPU/GPU evidence disagrees: {name}")
    for run in (cpu, gpu):
        artifacts = run.manifest.get("control_artifacts")
        if not isinstance(artifacts, dict) or "controls.json" not in artifacts:
            raise ValueError("Missing control artifact inventory")
        for name, digest in artifacts.items():
            path = (run.path / name).resolve()
            if not path.is_relative_to(run.path.resolve()) or file_hash(path) != checksum(digest):
                raise ValueError("Archived control evidence changed")
        if file_hash(run.path / "submission.zip") != run.manifest["candidate_sha256"]:
            raise ValueError("Archived candidate changed")
    files = archive_files(gpu.path / "submission.zip", max_bytes)
    if files != gpu.manifest["submission_files"]:
        raise ValueError("Archive inventory disagrees with verified candidate")
    return checked_candidate(
        {
            "id": content_id(files),
            "archive_sha256": gpu.manifest["candidate_sha256"],
            "files": files,
            "source": source,
            "evidence": {
                "cpu_manifest_sha256": file_hash(cpu_path),
                "gpu_manifest_sha256": file_hash(gpu_path),
                "source_sha256": gpu.manifest["source_sha256"],
                "cpu_run_id": cpu.manifest["run_id"],
                "gpu_run_id": gpu.manifest["run_id"],
            },
        }
    )


def empty_state(competition: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "competition": competition,
        "candidates": [],
        "attempts": [],
        "submissions": [],
        "incident": None,
        "last_sync": None,
    }


def validate_state(value: Any, competition: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != set(empty_state(competition))
        or value["schema_version"] != SCHEMA_VERSION
        or value["competition"] != competition
    ):
        raise ValueError("Invalid submission state")
    if any(not isinstance(value[k], list) for k in ("candidates", "attempts", "submissions")):
        raise ValueError("Invalid state collections")
    if value["incident"] is not None and not isinstance(value["incident"], str):
        raise ValueError("Invalid incident")
    if value["last_sync"] is not None:
        timestamp(value["last_sync"])
    ids = []
    for candidate in value["candidates"]:
        ids.append(checked_candidate(candidate)["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate")
    for attempt in value["attempts"]:
        if (
            not isinstance(attempt, dict)
            or set(attempt) != {"candidate", "token", "started_at", "status", "submission_id"}
            or attempt["candidate"] not in ids
            or attempt["status"] not in {"creating", "accepted", "not_accepted"}
            or not isinstance(attempt["token"], str)
            or not re.fullmatch(r"[a-f0-9]+", attempt["token"])
        ):
            raise ValueError("Invalid submission attempt")
        timestamp(attempt["started_at"])
        ref = attempt["submission_id"]
        if (attempt["status"] == "accepted" and (type(ref) is not int or ref <= 0)) or (
            attempt["status"] != "accepted" and ref is not None
        ):
            raise ValueError("Invalid attempt receipt")
    tokens = [a["token"] for a in value["attempts"]]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate attempt token")
    refs = []
    for row in value["submissions"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "date", "status", "score", "content_id", "manual"}
            or type(row["id"]) is not int
            or row["id"] <= 0
            or type(row["manual"]) is not bool
        ):
            raise ValueError("Invalid submission record")
        timestamp(row["date"])
        if not isinstance(row["status"], str) or (
            row["score"] is not None
            and (not isinstance(row["score"], str) or not math.isfinite(float(row["score"])))
        ):
            raise ValueError("Invalid submission score or status")
        checksum(row["content_id"])
        refs.append(row["id"])
    if len(refs) != len(set(refs)):
        raise ValueError("Duplicate submission receipt")
    return value


def add_candidate(state: dict[str, Any], path: Path) -> None:
    candidate = checked_candidate(read_object(path))
    if any(c["id"] == candidate["id"] for c in state["candidates"]):
        raise ValueError("Candidate is already registered")
    state["candidates"].append(candidate)
