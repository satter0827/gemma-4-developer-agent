"""Prepare and invoke an isolated, rebuildable MLflow SQLite projection."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash, new_run_id, write_json
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.evaluation.experiments import content_hash
from gemma_agent_dev.tasks.catalog import checksum
from gemma_agent_dev.tasks.pools import identifier, task_ids


@dataclass(frozen=True)
class TrackingSettings:
    requirements: Path
    lock_file: Path
    environment: Path
    database: Path
    output_dir: Path
    python_version: str
    timeout_seconds: int
    experiment_name: str

    @classmethod
    def load(cls, root: Path, config: Path) -> "TrackingSettings":
        data = mapping(
            yaml.safe_load(project_path(root, str(config)).read_text()),
            set(cls.__dataclass_fields__),
            "tracking",
        )
        for key in ("requirements", "lock_file", "environment", "database", "output_dir"):
            data[key] = project_path(root, data[key])
        for key in ("environment", "database", "output_dir"):
            if not data[key].is_relative_to(root / "artifacts"):
                raise ValueError("MLflow storage must stay in artifacts")
        storage = [data[k] for k in ("environment", "database", "output_dir")]
        protected = [root / "artifacts" / k for k in ("venv", "results", "kaggle", "runtime-env")]
        for index, directory in enumerate(storage):
            if any(
                directory.is_relative_to(other) or other.is_relative_to(directory)
                for other in storage[index + 1 :] + protected
            ):
                raise ValueError("MLflow storage overlaps another managed directory")
        if data["python_version"] != "3.12":
            raise ValueError("MLflow requires Python 3.12")
        positive(data["timeout_seconds"], "tracking timeout", integer=True)
        identifier(data["experiment_name"])
        return cls(**data)

    def fingerprint(self) -> dict[str, str]:
        return {
            "requirements": file_hash(self.requirements),
            "lock_file": file_hash(self.lock_file),
            "python_version": self.python_version,
        }


def worker_environment(root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "UV_CACHE_DIR": str(root / "artifacts/cache/uv"),
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
        "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "DO_NOT_TRACK": "1",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "OTEL_SDK_DISABLED": "true",
    }


def installed(settings: TrackingSettings, root: Path) -> dict[str, Any]:
    return dict(
        json.loads(
            command(
                [
                    str(settings.environment / "bin/python"),
                    "-c",
                    "import importlib.metadata,json,platform;"
                    "print(json.dumps({'python':platform.python_version(),"
                    "'packages':{d.metadata['Name']:d.version "
                    "for d in importlib.metadata.distributions()}}))",
                ],
                cwd=root,
                env=worker_environment(root),
                timeout=settings.timeout_seconds,
            )
        )
    )


def prepare(root: Path, config: Path) -> Path:
    settings = TrackingSettings.load(root, config)
    receipt = settings.environment / "prepared.json"
    receipt.unlink(missing_ok=True)
    env = worker_environment(root)
    python = settings.environment / "bin/python"
    if not python.exists():
        command(
            ["uv", "venv", "--python", settings.python_version, str(settings.environment)],
            cwd=root,
            env=env,
            timeout=settings.timeout_seconds,
        )
    command(
        ["uv", "pip", "sync", "--python", str(python), "--require-hashes", str(settings.lock_file)],
        cwd=root,
        env=env,
        timeout=settings.timeout_seconds,
    )
    state = installed(settings, root)
    if (
        not state["python"].startswith(settings.python_version + ".")
        or state["packages"].get("mlflow-skinny") != "3.17.2"
    ):
        raise ValueError("MLflow installation does not match the pinned environment")
    write_json(receipt, {"inputs": settings.fingerprint(), "installed": state})
    return receipt


def load_records(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for path in paths:
        row = mapping(
            json.loads(project_path(root, str(path)).read_text()),
            {
                "record_id",
                "experiment_id",
                "hypothesis",
                "status",
                "task_ids",
                "definition_sha256",
                "source_sha256",
                "candidate_hashes",
                "metrics",
                "interpretation",
                "prompts",
            },
            "experiment record",
        )
        for key in ("record_id", "experiment_id", "hypothesis"):
            identifier(row[key])
        if row["record_id"] in seen or row["status"] not in {
            "cpu-verified",
            "evaluated",
            "incomplete",
            "rejected",
        }:
            raise ValueError("Duplicate record or invalid status")
        seen.add(row["record_id"])
        task_ids(row["task_ids"])
        checksum(row["definition_sha256"])
        checksum(row["source_sha256"])
        if not isinstance(row["candidate_hashes"], dict) or not row["candidate_hashes"]:
            raise ValueError("Missing candidate hashes")
        for key, value in row["candidate_hashes"].items():
            identifier(key)
            checksum(value)
        if not isinstance(row["metrics"], dict) or any(
            not isinstance(k, str)
            or not identifier(k)
            or type(v) not in (int, float)
            or not float("-inf") < v < float("inf")
            for k, v in row["metrics"].items()
        ):
            raise ValueError("Metrics must be finite numbers")
        if (
            not isinstance(row["interpretation"], str)
            or not row["interpretation"].strip()
            or len(row["interpretation"].splitlines()) != 1
        ):
            raise ValueError("Record needs a short, single-line interpretation")
        if not isinstance(row["prompts"], list):
            raise ValueError("Expected prompt definitions")
        prompt_ids = set()
        for prompt in row["prompts"]:
            mapping(prompt, {"prompt_id", "path", "sha256"}, "prompt")
            identifier(prompt["prompt_id"])
            path_value = project_path(root, prompt["path"])
            if (
                prompt["prompt_id"] in prompt_ids
                or not path_value.is_relative_to(root / "agent/prompts")
                or path_value.suffix != ".md"
                or file_hash(path_value) != checksum(prompt["sha256"])
            ):
                raise ValueError("Prompt definition changed or is outside authored prompts")
            prompt_ids.add(prompt["prompt_id"])
        records.append(row)
    return records


def sync(root: Path, config: Path, paths: list[Path]) -> Path:
    settings = TrackingSettings.load(root, config)
    receipt = json.loads((settings.environment / "prepared.json").read_text())
    if receipt != {"inputs": settings.fingerprint(), "installed": installed(settings, root)}:
        raise ValueError("MLflow environment changed; prepare again")
    records = load_records(root, paths)
    if not records:
        raise ValueError("Specify at least one Git record")
    output = settings.output_dir / new_run_id()
    output.mkdir(parents=True)
    # The worker receives only the validated public schema; raw artifacts never enter MLflow.
    request = {
        "database": str(settings.database),
        "experiment_name": settings.experiment_name,
        "records": [
            {
                **r,
                "content_sha256": content_hash(r),
                "prompts": [
                    {**p, "template": (root / p["path"]).read_text()} for p in r["prompts"]
                ],
            }
            for r in records
        ],
    }
    write_json(output / "request.json", request)
    try:
        log = command(
            [
                str(settings.environment / "bin/python"),
                "-m",
                "gemma_agent_dev.evaluation.tracking_worker",
                str(output / "request.json"),
                str(output / "report.json"),
            ],
            cwd=root,
            env=worker_environment(root),
            timeout=settings.timeout_seconds,
        )
        (output / "check.log").write_text(log)
    finally:
        (output / "request.json").unlink()
    return output
