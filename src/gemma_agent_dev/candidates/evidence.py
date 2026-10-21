"""Verify immutable candidates against their archived CPU and GPU evidence."""

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from gemma_agent_dev.candidates.archive import extract_archive, validate_references
from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.evaluation.results import load_run, require_evaluation_evidence
from gemma_agent_dev.tasks.catalog import checksum


class CandidateSource(TypedDict):
    owner: str
    kernel: str
    version: int
    path: str


class CandidateEvidence(TypedDict):
    cpu_manifest_sha256: str
    gpu_manifest_sha256: str
    source_sha256: str
    cpu_run_id: str
    gpu_run_id: str


class Candidate(TypedDict):
    id: str
    archive_sha256: str
    files: dict[str, str]
    source: CandidateSource
    evidence: CandidateEvidence


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


def checked_candidate(value: Any) -> Candidate:
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
    return cast(Candidate, value)


def qualify_candidate(
    cpu_path: Path, gpu_path: Path, source: Mapping[str, Any], max_bytes: int
) -> Candidate:
    cpu, gpu = load_run(cpu_path.parent), load_run(gpu_path.parent, require_evaluation=True)
    require_evaluation_evidence(gpu)
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
