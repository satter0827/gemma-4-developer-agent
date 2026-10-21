"""Prepare native CPU environments and replay patches without importing inference libraries."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.common.events import context, event, local_session, logged, phase
from gemma_agent_dev.common.junit import diagnostic_evidence
from gemma_agent_dev.common.processes import command, process_group
from gemma_agent_dev.diagnostics.config import DiagnosticSettings, DiagnosticTask
from gemma_agent_dev.diagnostics.origins import (
    HOST_DISPLAY_VARIABLES,
    test_environment,
    validate_origins,
)
from gemma_agent_dev.tasks.catalog import TaskCatalog


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def source_inventory(root: Path) -> dict[str, str]:
    """Preserve official internal links without allowing links outside a snapshot."""
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()) or not path.exists():
                raise ValueError(f"Unsafe snapshot link: {name}")
            files[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            files[name] = file_hash(path)
    return files


def environment(settings: DiagnosticSettings) -> dict[str, str]:
    result = {
        **os.environ,
        "UV_CACHE_DIR": str(settings.root / "artifacts/cache/uv"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(settings.root / "artifacts/cache/task-python"),
        "PYTHONNOUSERSITE": "1",
    }
    for key in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTHONHOME",
        "PYTHONPATH",
        *HOST_DISPLAY_VARIABLES,
    ):
        result.pop(key, None)
    return result


@logged("diagnostic.prepare")
def prepare(settings: DiagnosticSettings) -> Path:
    catalog = TaskCatalog.load(settings.root, settings.catalog_path)
    catalog.wheel_cache.mkdir(parents=True, exist_ok=True)
    for task in settings.tasks:
        for name, digest in task.dependency_wheels.items():
            wheel = catalog.wheel_cache / name
            if not wheel.exists():
                package, version, _ = name.split("-", 2)
                with urllib.request.urlopen(
                    f"https://pypi.org/pypi/{package}/{version}/json",
                    timeout=settings.timeout_seconds,
                ) as response:
                    metadata = json.load(response)
                url = next(
                    item["url"]
                    for item in metadata["urls"]
                    if item["filename"] == name and item["digests"]["sha256"] == digest
                )
                with urllib.request.urlopen(url, timeout=settings.timeout_seconds) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Downloaded task wheel changed")
                wheel.write_bytes(content)
    catalog.wheels([t.instance_id for t in settings.tasks])
    tasks = settings.read_tasks()
    (settings.cache_dir / "local-success.json").unlink(missing_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    snapshots = settings.cache_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "fingerprint": settings.preparation_fingerprint(),
        "platform": platform.platform(),
        "tasks_sha256": settings.tasks_sha256,
        "lock_sha256": file_hash(settings.lock_file),
        "sources": {},
        "environments": {},
    }
    for task in settings.tasks:
        archive = snapshots / f"{task.instance_id}.tgz"
        if not archive.exists():
            command(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    settings.runtime.competition,
                    "-f",
                    f"snapshots/{task.instance_id}.tgz",
                    "-p",
                    str(snapshots),
                    "--quiet",
                ],
                cwd=settings.root,
                timeout=settings.timeout_seconds,
            )
        if file_hash(archive) != task.snapshot_sha256:
            raise ValueError(f"Snapshot checksum mismatch: {task.instance_id}")
        source = settings.cache_dir / "sources" / task.instance_id
        source_receipt = source.with_suffix(".json")
        if not source.exists() or not source_receipt.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
                extracted = Path(temporary)
                with tarfile.open(archive) as tar:
                    tar.extractall(extracted, filter="data")
                head = command(
                    ["git", "rev-parse", "HEAD"], cwd=extracted, timeout=settings.timeout_seconds
                ).strip()
                # Official snapshots rewrite Git history; their HEAD is not the upstream commit.
                (extracted / ".snapshot-head").write_text(head + "\n")
                # Git history is not needed to replay a patch and is never executed.
                shutil.rmtree(extracted / ".git")
                expected_files = source_inventory(extracted)
                if source.exists():
                    if source_inventory(source) != expected_files:
                        raise ValueError(f"Cached source changed: {task.instance_id}")
                else:
                    extracted.rename(source)
                write_json(
                    source_receipt,
                    {
                        "snapshot_sha256": task.snapshot_sha256,
                        "files": expected_files,
                    },
                )
        hashes = source_inventory(source)
        receipt = json.loads(source_receipt.read_text())
        if receipt["snapshot_sha256"] != task.snapshot_sha256 or receipt["files"] != hashes:
            raise ValueError(f"Cached source changed: {task.instance_id}")
        manifest["sources"][task.instance_id] = {
            "snapshot_sha256": task.snapshot_sha256,
            "base_commit": tasks[task.instance_id]["base_commit"],
            "snapshot_head": (source / ".snapshot-head").read_text().strip(),
            "files": hashes,
        }
        task_env = settings.environments_dir / task.instance_id
        if not task_env.exists():
            command(
                ["uv", "venv", "--python", settings.python_version, str(task_env)],
                cwd=settings.root,
                env=environment(settings),
                timeout=settings.timeout_seconds,
            )
        python = task_env / "bin/python"
        installed_environment(python, settings)
        command(
            [
                "uv",
                "pip",
                "sync",
                "--python",
                str(python),
                "--require-hashes",
                str(settings.lock_file),
            ],
            cwd=settings.root,
            env=environment(settings),
            timeout=settings.timeout_seconds,
        )
        if task.dependency_wheels:
            command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--no-index",
                    "--no-deps",
                    *[str(catalog.wheel_cache / name) for name in task.dependency_wheels],
                ],
                cwd=settings.root,
                env=environment(settings),
                timeout=settings.timeout_seconds,
            )
        manifest["environments"][task.instance_id] = installed_environment(python, settings)
    destination = settings.cache_dir / "prepared.json"
    write_json(destination, manifest)
    return destination


def installed_environment(python: Path, settings: DiagnosticSettings) -> dict[str, Any]:
    script = (
        "import importlib.metadata,json,platform; "
        "print(json.dumps({'python':platform.python_version(), 'machine':platform.machine(), "
        "'packages':{d.metadata['Name']:d.version "
        "for d in importlib.metadata.distributions()}}))"
    )
    state = dict(
        json.loads(
            command(
                [str(python), "-c", script],
                cwd=settings.root,
                env=environment(settings),
                timeout=settings.timeout_seconds,
            )
        )
    )
    if not state["python"].startswith(settings.python_version + "."):
        raise ValueError(f"Unexpected task Python version: {state['python']}")
    return state


def check_prepared(settings: DiagnosticSettings) -> dict[str, Any]:
    TaskCatalog.load(settings.root, settings.catalog_path).wheels(
        [t.instance_id for t in settings.tasks]
    )
    prepared = json.loads((settings.cache_dir / "prepared.json").read_text())
    if prepared["fingerprint"] != settings.preparation_fingerprint():
        raise ValueError("Diagnostic inputs changed; prepare the environments again")
    for task in settings.tasks:
        archive = settings.cache_dir / "snapshots" / f"{task.instance_id}.tgz"
        if file_hash(archive) != task.snapshot_sha256:
            raise ValueError("Cached snapshot changed")
        source = settings.cache_dir / "sources" / task.instance_id
        if source_inventory(source) != prepared["sources"][task.instance_id]["files"]:
            raise ValueError("Cached source changed")
        python = settings.environments_dir / task.instance_id / "bin/python"
        if installed_environment(python, settings) != prepared["environments"][task.instance_id]:
            raise ValueError("Diagnostic dependencies changed; prepare the environments again")
    return dict(prepared)


def apply_patch(workspace: Path, patch: str, destination: Path, timeout: int) -> None:
    destination.write_text(patch)
    if patch.strip():
        with tempfile.NamedTemporaryFile("w", suffix=".patch", dir=destination.parent) as handle:
            handle.write(patch if patch.endswith("\n") else patch + "\n")
            handle.flush()
            command(["git", "apply", "--check", handle.name], cwd=workspace, timeout=timeout)
            command(["git", "apply", handle.name], cwd=workspace, timeout=timeout)


@logged("diagnostic.case")
def run_case(
    settings: DiagnosticSettings,
    task: DiagnosticTask,
    data: dict[str, Any],
    name: str,
    patch: str,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True)
    expected = task.expected_outcome(name)
    record: dict[str, Any] = {
        "task_id": task.instance_id,
        "case": name,
        "excluded_display_environment": list(HOST_DISPLAY_VARIABLES),
        "expected_pass": expected == "passed",
        "expected_outcome": expected,
        "patch_sha256": digest_text(patch),
        "test_patch_sha256": digest_text(data["test_patch"]),
        "passed": False,
        "matched": False,
    }
    start = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        with tempfile.TemporaryDirectory(dir=settings.root / "artifacts/tmp") as temporary:
            workspace = Path(temporary) / "workspace"
            shutil.copytree(
                settings.cache_dir / "sources" / task.instance_id, workspace, symlinks=True
            )
            command(["git", "init", "-q"], cwd=workspace, timeout=settings.timeout_seconds)
            apply_patch(workspace, patch, output / "candidate.patch", settings.timeout_seconds)
            apply_patch(
                workspace, data["test_patch"], output / "tests.patch", settings.timeout_seconds
            )
            python = settings.environments_dir / task.instance_id / "bin/python"
            env = {
                **environment(settings),
                **test_environment(
                    workspace, task.source_roots, task.import_modules, output / "source-guard"
                ),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            }
            args = [
                str(python),
                "-m",
                "pytest",
                *settings.pytest_args,
                *task.test_paths,
                f"--junitxml={output / 'junit.xml'}",
            ]
            record["command"] = args
            with (output / "test.log").open("w") as log:
                with process_group(
                    args,
                    cwd=workspace,
                    env=env,
                    stdout=log,
                ) as process:
                    record["exit_code"] = process.wait(timeout=settings.timeout_seconds)
            record["source_origins"] = validate_origins(output / "source-guard/origins.json")
            outcome, counts = diagnostic_evidence(
                record["exit_code"], output / "junit.xml", expected
            )
            record.update(
                junit=counts,
                outcome=outcome,
                passed=outcome == "passed",
                matched=outcome == expected,
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        if process is not None:
            record["exit_code"] = process.returncode
        record["error"] = f"{type(error).__name__}: {error}"
        if not (output / "test.log").exists():
            (output / "test.log").write_text(record["error"] + "\n")
    finally:
        record["duration_seconds"] = time.perf_counter() - start
        record["artifacts"] = inventory(output)
        write_json(output / "case.json", record)
    return record


def _diagnose(settings: DiagnosticSettings, output: Path) -> Path:
    (settings.cache_dir / "local-success.json").unlink(missing_ok=True)
    prepared = check_prepared(settings)
    tasks = settings.read_tasks()
    (settings.root / "artifacts/tmp").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "diagnostic",
        "status": "running",
        "engine": "native-python",
        "fingerprint": settings.fingerprint(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "prepared": prepared,
        "cases": [],
    }
    write_json(output / "report.json", report)
    try:
        for task in settings.tasks:
            patches = task.patches(tasks[task.instance_id])
            for name, patch in patches.items():
                with context(task_id=task.instance_id, stage=name):
                    record = run_case(
                        settings,
                        task,
                        tasks[task.instance_id],
                        name,
                        patch,
                        output / task.instance_id / name,
                    )
                    event(
                        "diagnostic.outcome", result="matched" if record["matched"] else "mismatch"
                    )
                report["cases"].append(record)
                write_json(output / "report.json", report)
                python = settings.environments_dir / task.instance_id / "bin/python"
                if (
                    installed_environment(python, settings)
                    != prepared["environments"][task.instance_id]
                ):
                    raise RuntimeError(
                        f"Diagnostic environment changed during case: {task.instance_id}/{name}"
                    )
                print(f"{task.instance_id}/{name}: matched={record['matched']}", flush=True)
                if name == "reference" and not record["matched"]:
                    raise RuntimeError(
                        f"Reference verification failed: {task.instance_id}; {output}"
                    )
        check_prepared(settings)
        if not all(r["matched"] for r in report["cases"]):
            raise RuntimeError(f"Diagnostic outcomes differ from expectations: {output}")
        report["status"] = "completed"
        write_json(output / "report.json", report)
        write_json(
            settings.cache_dir / "local-success.json",
            {
                "fingerprint": settings.fingerprint(),
                "report": str(output / "report.json"),
                "sha256": file_hash(output / "report.json"),
            },
        )
    except BaseException as error:
        (settings.cache_dir / "local-success.json").unlink(missing_ok=True)
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "report.json", report)
        raise
    return output


def diagnose(settings: DiagnosticSettings) -> Path:
    output = settings.output_dir / new_run_id()
    output.mkdir(parents=True)
    with local_session(settings.root, output), phase("diagnostic"):
        return _diagnose(settings, output)
