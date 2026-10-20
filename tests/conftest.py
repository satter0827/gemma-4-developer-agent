import json
import shlex
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.diagnostics import installed_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_run(tmp_path: Path) -> Callable[..., Path]:
    def create(
        name: str = "run-a",
        *,
        ids: tuple[str, ...] = ("task_a", "task_b"),
        completed: int = 2,
        status: str = "completed",
    ) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": name,
            "task_ids": ids,
            "status": status,
            "evaluation": {"max_time_minutes": 1},
            "environment": {"sandbox": "subprocess"},
            "tasks_sha256": "fixture",
            "verification_timeout_seconds": 300,
            "vllm_config": {"max_model_len": 32768},
            "submission_files": {"agent.yaml": "fixture", "prompts/system.md": "fixture"},
            "runtime_config": {
                "model_source": "fixture",
                "docker_image": "fixture",
                "vllm": {},
                "context_cache": {},
                "compaction": {},
            },
        }
        write_json(directory / "manifest.json", manifest)
        rows = [
            {
                "instance_id": task,
                "resolved": index == 0,
                "duration_seconds": 2.5,
                "agent_patch_size": 10,
                "test_exit_code": 0 if index == 0 else 1,
                "error": None,
            }
            for index, task in enumerate(ids[:completed])
        ]
        (directory / "task_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        write_json(
            directory / "summary.json",
            {
                "total_tasks": len(rows),
                "resolved": sum(row["resolved"] for row in rows),
            },
        )
        return directory

    return create


@pytest.fixture
def diagnostic_project(tmp_path: Path) -> DiagnosticSettings:
    root = tmp_path / "project"
    root.mkdir()
    (root / "artifacts/tmp").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "kaggle", root / "kaggle")
    config = json.loads((ROOT / "configs/runtime.json").read_text())
    config["task_ids"] = ["fixture_1"]
    write_json(root / "runtime.json", config)
    generated = "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-value = 0\n+value = -1\n"
    (root / "generated.patch").write_text(generated)
    task = {
        "instance_id": "fixture_1",
        "repo": "fixture/repo",
        "base_commit": "fixture",
        "problem_statement": "Return one.",
        "patch": generated.replace("value = -1", "value = 1").rstrip("\n"),
        "test_patch": (
            "--- /dev/null\n+++ b/tests/test_calc.py\n@@ -0,0 +1,3 @@\n"
            "+from calc import value\n+def test_value():\n+    assert value == 1\n"
        ),
    }
    (root / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    cache = root / "artifacts/reference/diagnostics"
    source = cache / "sources/fixture_1"
    source.mkdir(parents=True)
    (source / "calc.py").write_text("value = 0\n")
    (source / "pytest.ini").write_text("[pytest]\n")
    snapshots = cache / "snapshots"
    snapshots.mkdir()
    (snapshots / "fixture_1.tgz").write_bytes(b"fixture snapshot")
    task_env = root / "artifacts/task-envs/fixture_1/bin"
    task_env.mkdir(parents=True)
    (task_env / "python").write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    (task_env / "python").chmod(0o755)
    (root / "requirements.in").write_text("pytest\n")
    (root / "requirements.lock").write_text("fixture lock\n")
    data = {
        "runtime_path": "runtime.json",
        "tasks_file": "tasks.jsonl",
        "tasks_sha256": file_hash(root / "tasks.jsonl"),
        "cache_dir": "artifacts/reference/diagnostics",
        "environments_dir": "artifacts/task-envs",
        "output_dir": "artifacts/diagnostics",
        "requirements": "requirements.in",
        "lock_file": "requirements.lock",
        "python_version": "3.12",
        "timeout_seconds": 30,
        "pytest_args": ["-q"],
        "kernel_id": "fixture/diagnostics",
        "tasks": [
            {
                "instance_id": "fixture_1",
                "test_paths": ["tests/test_calc.py"],
                "snapshot_sha256": file_hash(snapshots / "fixture_1.tgz"),
                "generated_patch": "generated.patch",
                "generated_sha256": file_hash(root / "generated.patch"),
                "probe_replacements": [{"old": "value = -1", "new": "value = 1", "count": 1}],
                "probe_expected_pass": True,
                "generated_expected_pass": False,
            }
        ],
    }
    (root / "diagnostics.yaml").write_text(yaml.safe_dump(data))
    settings = DiagnosticSettings.load(root, Path("diagnostics.yaml"))
    write_json(
        cache / "prepared.json",
        {
            "fingerprint": settings.fingerprint(),
            "sources": {"fixture_1": {"files": inventory(source)}},
            "environments": {"fixture_1": installed_environment(task_env / "python", settings)},
        },
    )
    return settings


@pytest.fixture
def preflight_project(diagnostic_project: DiagnosticSettings) -> Path:
    import zipfile

    root = diagnostic_project.root
    for directory in ("scripts", "tests", "agent", "configs"):
        shutil.copytree(ROOT / directory, root / directory, dirs_exist_ok=True)
    for name in ("Makefile", "pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, root / name)
    archive = root / "adapters.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("adapters/main_lora/adapter_config.json", "{}")
        handle.writestr("adapters/main_lora/adapter_model.safetensors", "fixture")
    import yaml

    (root / "preflight.yaml").write_text(
        yaml.safe_dump(
            {
                "agent_dir": "agent",
                "evaluation_config": "agent/eval_config.yaml",
                "runtime_config": "runtime.json",
                "diagnostic_config": "diagnostics.yaml",
                "adapter_archive": "adapters.zip",
                "adapter_sha256": file_hash(archive),
                "output_dir": "artifacts/preflight",
                "max_unpacked_bytes": 1000000,
                "quality_timeout_seconds": 60,
            }
        )
    )
    return root


@pytest.fixture
def preflight_report(preflight_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import subprocess

    from gemma_agent_dev import preflight

    original = subprocess.run

    def quality(args: list[str], **kwargs: Any) -> Any:
        if args == ["make", "verify"]:
            kwargs["stdout"].write("Quality checks passed (fixture)\n")
            return subprocess.CompletedProcess(args, 0)
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", quality)
    return preflight.run(preflight_project, Path("preflight.yaml"))
