"""Prepare an isolated CPU SDK environment and run offline connection checks."""

import json
import os
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash, new_run_id, python_inventory, write_json
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.common.events import logged
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.evaluation.config import RuntimeSettings


def settings(root: Path, path: Path) -> dict[str, Any]:
    data = mapping(
        yaml.safe_load(project_path(root, str(path)).read_text()),
        {
            "runtime_config",
            "requirements",
            "lock_file",
            "environment",
            "wheel_cache",
            "output_dir",
            "timeout_seconds",
            "python_version",
            "sampling",
        },
        "runtime checks",
    )
    for key in (
        "runtime_config",
        "requirements",
        "lock_file",
        "environment",
        "wheel_cache",
        "output_dir",
        "sampling",
    ):
        data[key] = project_path(root, data[key])
    positive(data["timeout_seconds"], "runtime check timeout", integer=True)
    if data["python_version"] != "3.12" or any(
        not data[k].is_file() for k in ("runtime_config", "requirements", "lock_file", "sampling")
    ):
        raise ValueError("Missing CPU runtime inputs or unsupported Python")
    if any(
        not data[k].is_relative_to(root / "artifacts")
        for k in ("environment", "wheel_cache", "output_dir")
    ):
        raise ValueError("CPU runtime storage must stay in artifacts")
    storage = [data[k] for k in ("environment", "wheel_cache", "output_dir")]
    protected = [root / "artifacts" / k for k in ("results", "venv", "cache", "kaggle", "tmp")]
    for index, directory in enumerate(storage):
        if any(
            directory.is_relative_to(other) or other.is_relative_to(directory)
            for other in storage[index + 1 :] + protected
        ):
            raise ValueError("CPU runtime storage overlaps another managed directory")
    return data


def fingerprint(data: dict[str, Any]) -> dict[str, str]:
    runtime = RuntimeSettings.load(data["runtime_config"])
    return {
        "python_version": data["python_version"],
        "requirements": file_hash(data["requirements"]),
        "lock_file": file_hash(data["lock_file"]),
        **{
            name: runtime.wheelhouse_hashes[f"{name.replace('-', '_')}-{version}-py3-none-any.whl"]
            for name, version in runtime.packages.items()
        },
    }


def installed_state(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    script = (
        "import importlib.metadata,json,platform;"
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{d.metadata['Name']:d.version "
        "for d in importlib.metadata.distributions()}}))"
    )
    state: dict[str, Any] = json.loads(
        command(
            [str(data["environment"] / "bin/python"), "-c", script],
            cwd=root,
            timeout=data["timeout_seconds"],
        )
    )
    if not state["python"].startswith(data["python_version"] + "."):
        raise ValueError("CPU runtime Python version changed")
    return state


@logged("sdk.prepare")
def prepare(root: Path, config: Path) -> Path:
    data = settings(root, config)
    (data["environment"] / "prepared.json").unlink(missing_ok=True)
    runtime = RuntimeSettings.load(data["runtime_config"])
    data["wheel_cache"].mkdir(parents=True, exist_ok=True)
    wheels = []
    for package, version in runtime.packages.items():
        name = f"{package.replace('-', '_')}-{version}-py3-none-any.whl"
        path = data["wheel_cache"] / name
        if not path.exists():
            command(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    runtime.wheelhouse_source,
                    "-f",
                    name,
                    "-p",
                    str(data["wheel_cache"]),
                    "--quiet",
                ],
                cwd=root,
                timeout=data["timeout_seconds"],
            )
        if file_hash(path) != runtime.wheelhouse_hashes[name]:
            raise ValueError("CPU SDK wheel checksum mismatch")
        wheels.append(path)
    env = {**os.environ, "UV_CACHE_DIR": str(root / "artifacts/cache/uv")}
    python = data["environment"] / "bin/python"
    if not python.exists():
        command(
            ["uv", "venv", "--python", data["python_version"], str(data["environment"])],
            cwd=root,
            env=env,
            timeout=data["timeout_seconds"],
        )
    command(
        ["uv", "pip", "sync", "--python", str(python), "--require-hashes", str(data["lock_file"])],
        cwd=root,
        env=env,
        timeout=data["timeout_seconds"],
    )
    command(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", *map(str, wheels)],
        cwd=root,
        env=env,
        timeout=data["timeout_seconds"],
    )
    destination = Path(data["environment"]) / "prepared.json"
    write_json(
        destination,
        {
            "inputs": fingerprint(data),
            "wheels": {p.name: file_hash(p) for p in wheels},
            "installed": installed_state(root, data),
        },
    )
    return destination


@logged("sdk.verify")
def check(root: Path, config: Path) -> Path:
    data = settings(root, config)
    receipt = json.loads((data["environment"] / "prepared.json").read_text())
    if receipt["inputs"] != fingerprint(data) or any(
        file_hash(data["wheel_cache"] / k) != v for k, v in receipt["wheels"].items()
    ):
        raise ValueError("CPU runtime inputs changed; prepare again")
    if receipt["installed"] != installed_state(root, data):
        raise ValueError("CPU runtime dependencies changed; prepare again")
    output = Path(data["output_dir"]) / new_run_id()
    output.mkdir(parents=True)
    try:
        result = command(
            [
                str(data["environment"] / "bin/python"),
                "-m",
                "gemma_agent_dev.evaluation.sdk_worker",
                str(data["runtime_config"]),
                str(data["sampling"]),
                str(output),
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "OTEL_SDK_DISABLED": "true",
            },
            timeout=data["timeout_seconds"],
        )
    except (OSError, ValueError, RuntimeError) as error:
        (output / "check.log").write_text(str(error))
        write_json(output / "report.json", {"status": "failed", "error": str(error)})
        raise
    (output / "check.log").write_text(result)
    report = json.loads((output / "report.json").read_text())
    if report.get("status") != "completed" or report.get("gpu_modules") != []:
        raise ValueError("CPU SDK checks did not complete without GPU imports")
    write_json(
        output / "inputs.json",
        {
            "runtime": RuntimeSettings.load(data["runtime_config"]).as_dict(),
            "sources": python_inventory(root / "src/gemma_agent_dev"),
            "sampling_sha256": file_hash(data["sampling"]),
            "worker_sha256": file_hash(root / "src/gemma_agent_dev/evaluation/sdk_worker.py"),
        },
    )
    return output
