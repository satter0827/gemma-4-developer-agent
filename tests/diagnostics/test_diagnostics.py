import asyncio
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import types
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.cli import main
from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.common.junit import junit_counts, validate_test_evidence
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import (
    check_prepared,
    diagnose,
    environment,
    installed_environment,
    run_case,
)
from gemma_agent_dev.diagnostics.remote import verify_cases
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.notebooks.diagnostics import prepare as prepare_kaggle

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("timeout", [False, True])
def test_preparation_commands_stop_descendants_on_exit_and_timeout(
    tmp_path: Path, timeout: bool
) -> None:
    marker = tmp_path / "leaked-command-child"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
    program = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "print('command output', flush=True)\n" + ("time.sleep(30)\n" if timeout else "")
    )
    args = [sys.executable, "-c", program]
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            command(args, cwd=tmp_path, timeout=1)
    else:
        assert command(args, cwd=tmp_path, timeout=5).strip() == "command output"
    time.sleep(2.1)
    assert not marker.exists()


@pytest.mark.parametrize("code", [0, 7])
def test_command_retains_stdout_semantics_and_failure_diagnostics(
    tmp_path: Path, code: int
) -> None:
    args = [
        sys.executable,
        "-c",
        f"import sys; print('output'); print('warning', file=sys.stderr); raise SystemExit({code})",
    ]
    if code:
        with pytest.raises(RuntimeError, match=r"Command failed \(7\)") as raised:
            command(args, cwd=tmp_path, timeout=5)
        assert "output" in str(raised.value) and "warning" in str(raised.value)
    else:
        assert command(args, cwd=tmp_path, timeout=5) == "output\n"


def test_real_patch_replay_is_isolated_and_creates_evidence(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    before = inventory(settings.cache_dir / "sources")
    output = diagnose(settings)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert [case["passed"] for case in report["cases"]] == [True, False, False, True]
    assert all(case["matched"] and case["junit"]["tests"] == 1 for case in report["cases"])
    assert inventory(settings.cache_dir / "sources") == before
    assert all((output / "fixture_1" / c["case"] / "junit.xml").is_file() for c in report["cases"])
    for case in report["cases"]:
        directory = output / "fixture_1" / case["case"]
        assert file_hash(directory / "candidate.patch") == case["patch_sha256"]
        assert file_hash(directory / "tests.patch") == case["test_patch_sha256"]
    assert not (settings.root / "artifacts/results").exists()


@pytest.mark.parametrize("change", ["snapshot", "source", "dependency", "configuration"])
def test_prepared_input_drift_fails_before_replay(
    diagnostic_project: DiagnosticSettings,
    change: str,
) -> None:
    settings = diagnostic_project
    paths = {
        "snapshot": settings.cache_dir / "snapshots/fixture_1.tgz",
        "source": settings.cache_dir / "sources/fixture_1/calc.py",
        "configuration": settings.requirements,
    }
    if change == "dependency":
        receipt = json.loads((settings.cache_dir / "prepared.json").read_text())
        receipt["environments"]["fixture_1"] = {}
        write_json(settings.cache_dir / "prepared.json", receipt)
    else:
        with paths[change].open("a") as handle:
            handle.write("changed")
    with pytest.raises(ValueError):
        check_prepared(settings)
    assert not settings.output_dir.exists()


def test_reference_failure_stops_remaining_cases(diagnostic_project: DiagnosticSettings) -> None:
    settings = diagnostic_project
    task_file = json.loads(settings.tasks_file.read_text())
    task_file["patch"] = task_file["patch"].replace("value = 1", "value = 2")
    settings.tasks_file.write_text(json.dumps(task_file) + "\n")
    catalog = yaml.safe_load(settings.catalog_path.read_text())
    catalog["tasks_sha256"] = file_hash(settings.tasks_file)
    settings.catalog_path.write_text(yaml.safe_dump(catalog))
    settings = replace(settings, tasks_sha256=file_hash(settings.tasks_file))
    receipt = json.loads((settings.cache_dir / "prepared.json").read_text())
    receipt["fingerprint"] = settings.preparation_fingerprint()
    write_json(settings.cache_dir / "prepared.json", receipt)
    write_json(settings.cache_dir / "local-success.json", {"stale": True})
    with pytest.raises(RuntimeError, match="Reference"):
        diagnose(settings)
    report = json.loads(next(settings.output_dir.glob("*/report.json")).read_text())
    assert report["status"] == "failed" and len(report["cases"]) == 1
    assert not (settings.cache_dir / "local-success.json").exists()


def test_inherited_pytest_options_cannot_change_diagnostics(
    diagnostic_project: DiagnosticSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests")
    monkeypatch.setenv("PYTEST_PLUGINS", "external_plugin")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("PYTHONPATH", "/unrelated/project")
    env = environment(diagnostic_project)
    assert "PYTEST_ADDOPTS" not in env and "PYTEST_PLUGINS" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "NO_COLOR" not in env and "FORCE_COLOR" not in env


@pytest.mark.parametrize(
    "xml", ["<testsuites/>", "<testsuite><testcase><skipped/></testcase></testsuite>", "broken"]
)
def test_no_tests_or_skipped_tests_are_not_success(tmp_path: Path, xml: str) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(xml)
    with pytest.raises((ValueError, ET.ParseError)):
        junit_counts(path)


def test_local_import_does_not_load_gpu_or_official_packages() -> None:
    script = (
        "import sys; import gemma_agent_dev.diagnostics; "
        "assert not any(n in sys.modules for n in ('torch','swegemma','adk_submission','litellm'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_diagnostic_bundle_is_cpu_only_and_separate(diagnostic_project: DiagnosticSettings) -> None:
    import ast
    import base64
    import io
    import zipfile

    settings = diagnostic_project
    with pytest.raises(FileNotFoundError):
        prepare_kaggle(settings.root, settings.root / "not-ready", settings.config_path)
    diagnose(settings)
    subprocess.run(["git", "init", "-q", str(settings.root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(settings.root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    destination = settings.root / "cpu"
    digest = prepare_kaggle(settings.root, destination, settings.config_path)
    assert digest == prepare_kaggle(
        settings.root, settings.root / "cpu-again", settings.config_path
    )
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"] and not metadata["enable_internet"]
    assert not metadata["model_sources"] and "machine_shape" not in metadata
    module = ast.parse((destination / "run.py").read_text())
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    payload = base64.b64decode(ast.literal_eval(statement.value.args[0]))
    assert hashlib.sha256(payload).hexdigest() == digest
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert not any(
            n.startswith("agent/") or n.endswith("diagnostics.py") for n in archive.namelist()
        )
        spec = json.loads(archive.read("diagnostics.json"))
    assert len(spec["cases"]) == 4
    assert spec["timeout_seconds"] == settings.timeout_seconds
    assert spec["lock_sha256"] == file_hash(settings.lock_file)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_ids", ["duplicate", "duplicate"]),
        ("verification_timeout_seconds", True),
        ("extra", 1),
        ("model_source", "wrong"),
        ("model_source", "google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/999"),
        ("verification_kernel_id", "satter0927/gemma-4-developer-agent-baseline"),
    ],
)
def test_runtime_settings_reject_invalid_inputs(tmp_path: Path, field: str, value: object) -> None:
    config = RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml").as_dict()
    config[field] = value
    path = tmp_path / "runtime.json"
    write_json(path, config)
    with pytest.raises(ValueError):
        RuntimeSettings.load(path)


def test_diagnose_cli_dispatches_without_preparation_or_network(
    diagnostic_project: DiagnosticSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(diagnostic_project.root)
    assert main(["diagnose", "--config", "diagnostics.yaml"]) == 0


def test_invalid_diagnostic_config_has_no_execution_side_effects(
    diagnostic_project: DiagnosticSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = diagnostic_project
    monkeypatch.chdir(settings.root)
    data = yaml.safe_load(settings.config_path.read_text())
    data["tasks"][0]["generated_sha256"] = "0" * 64
    settings.config_path.write_text(yaml.safe_dump(data))
    before = inventory(settings.cache_dir)
    assert main(["prepare-diagnostics", "--config", "diagnostics.yaml"]) == 1
    assert inventory(settings.cache_dir) == before
    assert not settings.output_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_dir", "artifacts/results/diagnostics"),
        ("output_dir", "artifacts/reference/diagnostics/output"),
        ("environments_dir", "artifacts/venv"),
        ("cache_dir", "artifacts"),
        ("requirements", "missing.in"),
        ("lock_file", "missing.lock"),
    ],
)
def test_diagnostic_storage_and_inputs_fail_before_preparation(
    diagnostic_project: DiagnosticSettings, field: str, value: str
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.config_path.read_text())
    data[field] = value
    settings.config_path.write_text(yaml.safe_dump(data))
    before = inventory(settings.cache_dir)
    with pytest.raises(ValueError):
        DiagnosticSettings.load(settings.root, settings.config_path)
    assert inventory(settings.cache_dir) == before


def test_invalid_yaml_is_a_cli_error(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = diagnostic_project
    settings.config_path.write_text("tasks: [")
    monkeypatch.chdir(settings.root)
    assert main(["prepare-diagnostics", "--config", "diagnostics.yaml"]) == 1


@pytest.mark.parametrize(
    "exit_code,xml",
    [
        (5, "<testcase/>"),
        (-9, "<testcase/>"),
        (1, "<testcase/>"),
        (0, "<testcase><failure/></testcase>"),
    ],
)
def test_incomplete_or_inconsistent_test_results_are_rejected(
    tmp_path: Path, exit_code: int, xml: str
) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(f"<testsuite>{xml}</testsuite>")
    with pytest.raises(ValueError):
        validate_test_evidence(exit_code, path)


def test_timeout_preserves_output_and_stops_child_processes(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = replace(diagnostic_project, timeout_seconds=1)
    marker = settings.root / "leaked-child"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
    program = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "print('started before timeout', flush=True)\n"
        "time.sleep(30)\n"
    )
    python = settings.environments_dir / "fixture_1/bin/python"
    python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(program)}\n")
    data = settings.read_tasks()["fixture_1"]
    output = settings.output_dir / "timeout"
    record = run_case(settings, settings.tasks[0], data, "reference", data["patch"], output)
    assert not record["matched"] and record["exit_code"] < 0
    assert "TimeoutExpired" in record["error"]
    assert "started before timeout" in (output / "test.log").read_text()
    time.sleep(1.5)
    assert not marker.exists()


def test_preflight_failure_invalidates_previous_success(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    write_json(settings.cache_dir / "local-success.json", {"stale": True})
    (settings.cache_dir / "sources/fixture_1/calc.py").write_text("changed")
    with pytest.raises(ValueError):
        diagnose(settings)
    assert not (settings.cache_dir / "local-success.json").exists()


def test_wrong_task_python_version_is_rejected(diagnostic_project: DiagnosticSettings) -> None:
    settings = diagnostic_project
    python = settings.environments_dir / "fixture_1/bin/python"
    state = json.dumps({"python": "3.11.9", "machine": "arm64", "packages": {}})
    python.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(state)}\n")
    with pytest.raises(ValueError, match="Python version"):
        installed_environment(python, settings)


def test_preparation_downloads_once_and_rejects_cached_source_or_snapshot_changes(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.diagnostics import local as diagnostics

    settings = diagnostic_project
    repo = settings.root / "snapshot-repo"
    source = settings.cache_dir / "sources/fixture_1"
    shutil.copytree(source, repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    archive = settings.root / "snapshot.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(repo, arcname=".")
    config = yaml.safe_load(settings.catalog_path.read_text())
    config["tasks"][0]["snapshot_sha256"] = file_hash(archive)
    wheel = settings.root / "artifacts/reference/task-wheels/fixture-1-py3-none-any.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"pinned-task-dependency")
    config["tasks"][0]["dependency_wheels"] = {wheel.name: file_hash(wheel)}
    settings.catalog_path.write_text(yaml.safe_dump(config))
    settings = DiagnosticSettings.load(settings.root, settings.config_path)
    shutil.rmtree(source)
    shutil.rmtree(settings.environments_dir)
    cached = settings.cache_dir / "snapshots/fixture_1.tgz"
    cached.unlink()
    from gemma_agent_dev.common.processes import command as real_command

    calls: list[list[str]] = []

    def command(args: list[str], **kwargs: Any) -> str:
        if args[0] == "kaggle":
            calls.append(args)
            shutil.copyfile(archive, cached)
            return ""
        if args[:2] == ["uv", "venv"]:
            calls.append(args)
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o755)
            return ""
        if args[:3] == ["uv", "pip", "sync"]:
            calls.append(args)
            assert "--require-hashes" in args and args[-1] == str(settings.lock_file)
            return ""
        if args[:3] == ["uv", "pip", "install"]:
            calls.append(args)
            assert "--no-index" in args and "--no-deps" in args and args[-1] == str(wheel)
            return ""
        return real_command(args, **kwargs)

    monkeypatch.setattr(diagnostics, "command", command)
    diagnostics.prepare(settings)
    assert check_prepared(settings)["sources"]["fixture_1"]["base_commit"] == "fixture"
    assert (source / ".snapshot-head").read_text().strip() != "fixture"
    assert not (source / ".git").exists()
    write_json(settings.cache_dir / "local-success.json", {"stale": True})
    diagnostics.prepare(settings)
    assert not (settings.cache_dir / "local-success.json").exists()
    source.with_suffix(".json").unlink()
    diagnostics.prepare(settings)
    assert sum(args[0] == "kaggle" for args in calls) == 1
    assert sum(args[:2] == ["uv", "venv"] for args in calls) == 1
    assert sum(args[:3] == ["uv", "pip", "install"] for args in calls) == 3
    (source / "calc.py").write_text("changed")
    with pytest.raises(ValueError, match="Cached source changed"):
        diagnostics.prepare(settings)
    source.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Cached source changed"):
        diagnostics.prepare(settings)
    cached.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Snapshot checksum"):
        diagnostics.prepare(settings)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tasks_sha256", "invalid"),
        ("python_version", "3.11"),
        ("kernel_id", "invalid"),
        ("kernel_id", "satter0927/gemma-4-developer-agent-baseline"),
        ("timeout_seconds", 0),
        ("pytest_args", [1]),
        ("tasks", "invalid"),
        ("tasks.0.instance_id", 1),
        ("tasks.0.test_paths", ["../test.py"]),
        ("tasks.0.probe_expected_pass", 1),
        ("tasks.0.snapshot_sha256", "invalid"),
        ("tasks.0.probe_replacements", None),
        ("tasks.0.probe_replacements.0.old", ""),
        ("tasks.0.probe_replacements.0.count", 2),
        ("cache_dir", "outside-artifacts"),
        ("cache_dir", "generated.patch"),
    ],
)
def test_diagnostic_config_rejects_invalid_values_before_any_work(
    diagnostic_project: DiagnosticSettings, field: str, value: Any
) -> None:
    settings = diagnostic_project
    config = yaml.safe_load(settings.config_path.read_text())
    target = config
    keys = field.split(".")
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[keys[-1]] = value
    settings.config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        DiagnosticSettings.load(settings.root, settings.config_path)


def test_success_receipt_save_failure_marks_the_run_failed(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.diagnostics import local as diagnostics

    settings = diagnostic_project

    def fail_receipt(path: Path, value: Any) -> None:
        if path.name == "local-success.json":
            raise OSError("receipt storage failed")
        write_json(path, value)

    monkeypatch.setattr(diagnostics, "write_json", fail_receipt)
    with pytest.raises(OSError, match="storage failed"):
        diagnose(settings)
    report = json.loads(next(settings.output_dir.glob("*/report.json")).read_text())
    assert report["status"] == "failed" and "storage failed" in report["error"]
    assert not (settings.cache_dir / "local-success.json").exists()


def test_environment_drift_stops_before_the_next_case(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.diagnostics import local as diagnostics

    settings = diagnostic_project
    real_state = installed_environment(settings.environments_dir / "fixture_1/bin/python", settings)
    calls = 0

    def state(python: Path, settings: DiagnosticSettings) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real_state if calls == 1 else {**real_state, "packages": {}}

    monkeypatch.setattr(diagnostics, "installed_environment", state)
    with pytest.raises(RuntimeError, match="environment changed during case"):
        diagnose(settings)
    report = json.loads(next(settings.output_dir.glob("*/report.json")).read_text())
    assert report["status"] == "failed" and len(report["cases"]) == 1
    assert not (settings.cache_dir / "local-success.json").exists()


@pytest.mark.parametrize("damage", ["test.log", "candidate.patch", "junit.xml", "duplicate"])
def test_kaggle_preparation_rejects_missing_changed_or_duplicate_local_evidence(
    diagnostic_project: DiagnosticSettings, damage: str
) -> None:
    settings = diagnostic_project
    output = diagnose(settings)
    if damage == "duplicate":
        report_file = output / "report.json"
        report = json.loads(report_file.read_text())
        report["cases"][2] = report["cases"][1]
        write_json(report_file, report)
        receipt_file = settings.cache_dir / "local-success.json"
        receipt = json.loads(receipt_file.read_text())
        receipt["sha256"] = file_hash(report_file)
        write_json(receipt_file, receipt)
    elif damage == "test.log":
        (output / "fixture_1/reference" / damage).unlink()
    else:
        (output / "fixture_1/reference" / damage).write_text("changed")
    destination = settings.root / "must-not-create"
    with pytest.raises(ValueError):
        prepare_kaggle(settings.root, destination, settings.config_path)
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["valid", "missing", "empty", "skipped", "crash", "exception"])
def test_official_diagnostics_validate_evidence_and_stop_on_reference_failure(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    settings = diagnostic_project
    runtime = replace(
        settings.runtime, paths=replace(settings.runtime.paths, data=str(settings.root))
    )
    data = settings.read_tasks()["fixture_1"]
    task = SimpleNamespace(**data)
    patches = settings.tasks[0].patches(data)
    spec: dict[str, Any] = {
        "tasks_sha256": settings.tasks_sha256,
        "snapshots": {"fixture_1": settings.tasks[0].snapshot_sha256},
        "timeout_seconds": 17,
        "cases": [
            {
                "task_id": "fixture_1",
                "case": name,
                "patch": patch,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "expected_pass": name in {"reference", "probe"},
                "expected_outcome": "passed" if name in {"reference", "probe"} else "failed",
            }
            for name, patch in patches.items()
        ],
    }
    spec["dependency_dir"] = str(settings.root)
    spec["sources"] = {"fixture_1": {"source_roots": ["."], "import_modules": ["calc"]}}
    monkeypatch.setattr(
        "gemma_agent_dev.diagnostics.sandbox.observe_sources", lambda *a: lambda: None
    )
    calls: list[str] = []
    current_xml = ""

    class Sandbox:
        def exec(self, container: str, command: str, **kwargs: Any) -> Any:
            return SimpleNamespace(stdout=current_xml if command.startswith("cat ") else "tests")

    class Evaluator:
        def __init__(self, config: Any) -> None:
            assert config.verification_timeout_seconds == 17
            self.sandbox = Sandbox()

    async def verify(sandbox: Any, config: Any, task: Any, snapshot: Path, **kwargs: Any) -> Any:
        nonlocal current_xml
        name = spec["cases"][len(calls)]["case"]
        calls.append(name)
        if mode == "exception":
            raise LookupError("setup failed")
        passed = name in {"reference", "probe"}
        case_xml = "<testcase/>" if passed else "<testcase><failure/></testcase>"
        if mode == "skipped":
            case_xml = "<testcase><skipped/></testcase>"
        if mode == "empty":
            case_xml = ""
        current_xml = f"<testsuite>{case_xml}</testsuite>" if mode != "missing" else ""
        sandbox.exec("fixture", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc123.xml")
        return SimpleNamespace(
            test_output="test output",
            resolved=passed,
            test_exit_code=-9 if mode == "crash" else (0 if passed else 1),
            error=None,
        )

    attributes: dict[str, dict[str, Any]] = {
        "adk_submission": {"ModelRegistry": lambda: None},
        "swegemma.config": {"EvalConfig": lambda **kwargs: SimpleNamespace(**kwargs)},
        "swegemma.deduplication": {
            "resolve_task_snapshot_paths": lambda *args: (
                settings.cache_dir / "snapshots/fixture_1.tgz",
                None,
                None,
            )
        },
        "swegemma.evaluate": {"Evaluator": Evaluator},
        "swegemma.harness.verification": {"verify_task": verify},
        "swegemma.models": {"load_tasks": lambda path: [task]},
    }
    for name, members in attributes.items():
        for length in range(1, len(name.split(".")) + 1):
            parent = ".".join(name.split(".")[:length])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        module = types.ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    output = settings.output_dir / "official"
    output.mkdir(parents=True)
    report: dict[str, Any] = {"cases": []}
    if mode == "valid":
        asyncio.run(verify_cases(runtime, spec, output, report))
        assert len(calls) == 4 and all(r["matched"] for r in report["cases"])
    else:
        with pytest.raises(RuntimeError, match="outcome mismatch"):
            asyncio.run(verify_cases(runtime, spec, output, report))
        assert calls == ["reference"]
        assert not report["cases"][0]["matched"] and report["cases"][0]["error"]
    assert (output / "fixture_1/reference/case.json").is_file()


def test_generated_patch_can_be_replayed_without_probe(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    config = yaml.safe_load(settings.config_path.read_text())
    config["tasks"][0]["probe_replacements"] = []
    settings.config_path.write_text(yaml.safe_dump(config))
    selected = DiagnosticSettings.load(settings.root, settings.config_path)
    patches = selected.tasks[0].patches(selected.read_tasks()["fixture_1"])
    assert list(patches) == ["reference", "unmodified", "generated"]


@pytest.mark.parametrize("exit_code", [2, 2.0, True])
def test_collection_failure_is_not_a_normal_test_result(tmp_path: Path, exit_code: Any) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite><testcase name="tests.module">'
        '<error message="collection failure">ImportError</error></testcase></testsuite>'
    )
    with pytest.raises(ValueError):
        validate_test_evidence(exit_code, path)
    with pytest.raises(ValueError):
        junit_counts(path)


def test_only_explicit_collection_diagnostics_allow_zero_executed_tests(tmp_path: Path) -> None:
    from gemma_agent_dev.common.junit import diagnostic_evidence

    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite><testcase name="tests.module">'
        '<error message="collection failure">ImportError</error></testcase></testsuite>'
    )
    outcome, counts = diagnostic_evidence(2, path, "collection_error")
    assert (
        outcome == "collection_error" and counts["tests"] == 0 and counts["collection_errors"] == 1
    )
    with pytest.raises(ValueError):
        diagnostic_evidence(2, path, "failed")
    with pytest.raises(ValueError):
        diagnostic_evidence(2.0, path, "collection_error")  # type: ignore[arg-type]
    path.write_text('<testsuite><testcase name="actual_test"/></testsuite>')
    with pytest.raises(ValueError, match="Interrupted"):
        diagnostic_evidence(2, path, "collection_error")


@pytest.mark.parametrize(
    "names", [["reference"], ["unknown"], ["probe"], ["generated", "generated"], None]
)
def test_collection_expectations_are_validated(
    diagnostic_project: DiagnosticSettings, names: Any
) -> None:
    data = yaml.safe_load(diagnostic_project.config_path.read_text())
    data["tasks"][0]["collection_error_cases"] = names
    diagnostic_project.config_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        DiagnosticSettings.load(diagnostic_project.root, diagnostic_project.config_path)


@pytest.mark.parametrize("declared", [False, True])
def test_real_collection_failure_requires_explicit_expectation(
    diagnostic_project: DiagnosticSettings,
    declared: bool,
) -> None:
    settings = diagnostic_project
    task = replace(settings.tasks[0], collection_error_cases=["generated"] if declared else [])
    patch = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n"
        "-value = 0\n+import nonexistent_diagnostic_module\n"
    )
    record = run_case(
        settings,
        task,
        settings.read_tasks()[task.instance_id],
        "generated",
        patch,
        settings.output_dir / "collection",
    )
    assert record["exit_code"] == 2 and record["passed"] is False
    assert record["matched"] is declared
    if declared:
        assert record["outcome"] == "collection_error"
        assert record["junit"]["tests"] == 0 and record["junit"]["collection_errors"] == 1
    else:
        assert "did not finish normally" in record["error"]
