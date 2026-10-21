import asyncio
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev import analysis, qualification, runtime_checks
from gemma_agent_dev.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.config import RuntimeSettings, SamplingSettings
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.diagnostics import source_inventory
from gemma_agent_dev.gpu_budget import GpuBudget
from gemma_agent_dev.kaggle_bundle import prepare as bundle
from gemma_agent_dev.provenance import test_environment as source_environment
from gemma_agent_dev.provenance import validate_origins
from gemma_agent_dev.results import load_run
from gemma_agent_dev.sandbox_sources import observe_sources
from gemma_agent_dev.task_catalog import TaskCatalog
from gemma_agent_dev.worker import run_bounded

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("layout,correct", [(".", True), ("src", True), ("src", False)])
def test_real_pytest_rejects_installed_copy_even_when_its_tests_pass(
    tmp_path: Path, layout: str, correct: bool
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / layout
    source.mkdir(parents=True)
    (source / "example.py").write_text("value = 1\n")
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "example.py").write_text("value = 1\n")
    (workspace / "test_example.py").write_text(
        "from example import value\ndef test_value():\n assert value == 1\n"
    )
    directory = tmp_path / "guard"
    env = {
        **os.environ,
        **source_environment(workspace, [layout] if correct else ["."], ["example"], directory),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    code = (
        f"import sys;sys.path.append({str(installed)!r});import pytest;"
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "-q", str(workspace / "test_example.py")],
        env=env,
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert "1 passed" in result.stdout
    if correct:
        assert result.returncode == 0 and validate_origins(directory / "origins.json")["valid"]
    else:
        assert result.returncode == 1
        with pytest.raises(ValueError, match="workspace"):
            validate_origins(directory / "origins.json")


def test_snapshot_links_are_preserved_but_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "original").write_text("content")
    (root / "alias").symlink_to("original")
    assert source_inventory(root)["alias"] == "symlink:original"
    with pytest.raises(ValueError):
        inventory(root)
    (root / "alias").unlink()
    (root / "alias").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Unsafe"):
        source_inventory(root)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_roots", ["../outside"]),
        ("import_modules", ["bad/module"]),
        ("source_roots", []),
        ("test_paths", ["bad.txt"]),
        ("snapshot_sha256", "invalid"),
        ("instance_id", "../bad"),
    ],
)
def test_shared_catalog_rejects_bad_contracts(
    diagnostic_project: DiagnosticSettings, key: str, value: Any
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    data["tasks"][0][key] = value
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        TaskCatalog.load(settings.root, settings.catalog_path)


@pytest.mark.parametrize("damage", ["duplicate", "test_paths", "tasks_data", "missing_data"])
def test_catalog_detects_duplicates_drift_and_missing_tasks(
    diagnostic_project: DiagnosticSettings, damage: str
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    if damage == "duplicate":
        data["tasks"] *= 2
    elif damage == "test_paths":
        data["tasks"][0]["test_paths"] = ["tests/another.py"]
    elif damage == "tasks_data":
        settings.tasks_file.write_text("{}\n")
    else:
        data["tasks"][0]["instance_id"] = "missing"
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        TaskCatalog.load(settings.root, settings.catalog_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", -1),
        ("top_p", 1.1),
        ("temperature", float("nan")),
        ("thinking_config", {"thinking_budget": True, "include_thoughts": True}),
        ("thinking_config", {"thinking_budget": 20000, "include_thoughts": True}),
        ("max_output_tokens", 32768),
    ],
)
def test_sampling_rejects_invalid_values_and_context_overflow(
    tmp_path: Path, key: str, value: Any
) -> None:
    data = yaml.safe_load((ROOT / "agent/configs/sampling.yaml").read_text())
    data[key] = value
    path = tmp_path / "sampling.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        SamplingSettings.load(path, RuntimeSettings.load(ROOT / "configs/runtime.json"))


def test_gpu_admission_uses_quota_hours_and_stops_on_unknown_or_active_runs() -> None:
    budget = GpuBudget.load(ROOT, Path("configs/gpu-budget.yaml"))
    assert budget.admit(consumed=0, remaining=26.5, completed_runs=0, active=False)["allowed"]
    assert not budget.admit(consumed=2, remaining=24, completed_runs=2, active=False)["allowed"]
    assert not budget.admit(consumed=0, remaining=0.5, completed_runs=0, active=False)["allowed"]
    assert not budget.admit(consumed=0, remaining=20, completed_runs=0, active=True)["allowed"]
    assert not budget.admit(consumed=0, remaining=20, completed_runs=3, active=False)["allowed"]
    invalid_values: list[Any] = [None, float("nan"), -1]
    for value in invalid_values:
        with pytest.raises(ValueError):
            budget.admit(consumed=value, remaining=20, completed_runs=0, active=False)
    with pytest.raises(ValueError):
        GpuBudget.parse({**asdict(budget), "reserve_hours": 4})


@pytest.mark.parametrize("mode", ["valid", "missing", "ambiguous", "empty-workspace"])
def test_official_subprocess_source_guard_records_actual_test_process(
    tmp_path: Path, mode: str
) -> None:
    root = tmp_path / "sandbox"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    if mode != "empty-workspace":
        (workspace / "example.py").write_text("value = 1\n")

    class Sandbox:
        sandboxes = {"one": {"root": root, "workspace": workspace}}

        def exec(self, sandbox_id: str, command: str, **kwargs: Any) -> Any:
            if command.startswith("cat "):
                return SimpleNamespace(stdout="<testsuite><testcase/></testsuite>")
            assert ("GEMMA_SOURCE_WORKSPACE" in command) is (mode != "empty-workspace")
            if mode == "valid":
                write_json(
                    root / "source-guard/origins.json",
                    {
                        "valid": True,
                        "workspace": str(workspace),
                        "modules": {"example": str(workspace / "example.py")},
                    },
                )
            return SimpleNamespace(stdout="tests")

    specs = {"task": {"source_roots": ["."], "import_modules": ["example"]}}
    if mode == "ambiguous":
        specs["other"] = {"source_roots": [".", "src"], "import_modules": ["example"]}
    sandbox = Sandbox()
    check = observe_sources(sandbox, specs, tmp_path / "out")
    if mode == "ambiguous":
        with pytest.raises(ValueError, match="Ambiguous"):
            sandbox.exec("one", "python3 -m pytest --junitxml=/tmp/_swegemma_junit_abc123.xml")
    else:
        sandbox.exec("one", "python3 -m pytest --junitxml=/tmp/_swegemma_junit_abc123.xml")
        if mode == "valid":
            check()
            assert (tmp_path / "out/source-checks/one/origins.json").is_file()
        else:
            with pytest.raises(ValueError, match="source verification"):
                check()


@pytest.mark.parametrize(
    "damage", ["none", "missing", "hash", "status", "controls", "source", "no-evidence", "artifact"]
)
def test_gpu_bundle_requires_matching_immutable_cpu_evidence(
    preflight_project: Path, preflight_report: Path, cpu_evidence: Path, damage: str
) -> None:
    path = cpu_evidence
    manifest = json.loads(path.read_text())
    if damage == "hash":
        manifest["candidate_sha256"] = "bad"
    elif damage == "status":
        manifest["status"] = "failed"
    elif damage == "controls":
        (path.parent / "controls.json").unlink()
    elif damage == "source":
        manifest["source_sha256"] = "bad"
    elif damage == "no-evidence":
        manifest.pop("control_artifacts")
    elif damage == "artifact":
        (path.parent / "origins.json").write_text("changed")
    write_json(path, manifest)
    destination = preflight_project / "gpu"
    if damage == "none":
        assert bundle(preflight_project, destination, report_path=preflight_report, cpu_report=path)
    else:
        with pytest.raises((ValueError, OSError)):
            bundle(
                preflight_project,
                destination,
                report_path=preflight_report,
                cpu_report=None if damage == "missing" else path,
            )
        assert not destination.exists()


def test_analyze_separates_official_result_from_trace_claims_and_keeps_unknowns(
    make_run: Any, tmp_path: Path
) -> None:
    path = make_run()
    result_file = path / "task_results.jsonl"
    rows = [json.loads(line) for line in result_file.read_text().splitlines()]
    rows[0]["tool_calls"] = 4
    result_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    folder = path / "traces"
    folder.mkdir()
    trace = {
        "steps": [
            {"tool_calls": [{"function_name": "read_file", "arguments": {}}]},
            {
                "tool_calls": [{"function_name": "edit_file", "arguments": {}}],
                "observation": {"content": '{"status":"error"}'},
            },
            {
                "tool_calls": [
                    {
                        "function_name": "run_command",
                        "arguments": {"command": 'python -c "print(1)"'},
                    },
                    {
                        "function_name": "run_command",
                        "arguments": {"command": "pytest tests/test_a.py"},
                    },
                ],
                "observation": {"content": '{"stdout":"0 tests collected","exit_code":0}'},
            },
            {"tool_calls": [{"function_name": "submit_patch", "arguments": {}}]},
        ]
    }
    write_json(folder / "trace_task_a.json", trace)
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 100000)
    assert inventory(path) == before
    report = json.loads((output / "report.json").read_text())
    a = report["tasks"]["task_a"]
    b = report["tasks"]["task_b"]
    assert a["official_result"]["resolved"] is True and a["individual_test_counts"] is None
    assert a["first_edit_attempt_trace_call"] == 2 and len(a["edit_failures"]) == 1
    assert a["official_tool_calls"] == 4 and a["trace_tool_calls"] == 5
    assert b["official_tool_calls"] is None and b["trace_tool_calls"] is None
    assert (
        len(a["verification_commands"]) == 1
        and not a["verification_commands"][0]["establishes_resolution"]
    )
    assert b["time_limit_reached"] is None and b["trace_error"]


def test_intermediate_submission_never_replaces_the_final_official_patch(
    make_run: Any, tmp_path: Path
) -> None:
    path = make_run()
    result_file = path / "task_results.jsonl"
    rows = [json.loads(line) for line in result_file.read_text().splitlines()]
    rows[1]["agent_patch_size"] = 0
    result_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "traces").mkdir()
    write_json(
        path / "traces/trace_task_b.json",
        {
            "steps": [
                {
                    "tool_calls": [{"function_name": "submit_patch", "arguments": {}}],
                    "observation": {"content": json.dumps({"status": "ok", "patch_size": size})},
                }
                for size in (404, 0)
            ]
        },
    )
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 10000)
    evidence = json.loads((output / "report.json").read_text())["tasks"]["task_b"]
    assert [s["observation"]["patch_size"] for s in evidence["reported_patch_submissions"]] == [
        404,
        0,
    ]
    assert evidence["official_result"]["agent_patch_size"] == 0
    assert evidence["official_result"]["resolved"] is False and inventory(path) == before


@pytest.mark.parametrize("outcome", ["improved", "unchanged", "regressed", "invalid"])
def test_analyze_compares_only_main_instruction_under_identical_conditions(
    make_run: Any, tmp_path: Path, outcome: str
) -> None:
    a = make_run("a")
    b = make_run("b")
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["submission_files"]["prompts/system.md"] = "new"
    if outcome == "invalid":
        manifest["task_ids"].reverse()
    write_json(b / "manifest.json", manifest)
    if outcome in {"improved", "regressed"}:
        rows = [json.loads(line) for line in (b / "task_results.jsonl").read_text().splitlines()]
        for row in rows:
            row["resolved"] = outcome == "improved"
            row["test_exit_code"] = 0 if row["resolved"] else 1
        (b / "task_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(
            b / "summary.json", {"total_tasks": 2, "resolved": sum(r["resolved"] for r in rows)}
        )
    output = analysis.analyze(load_run(a), tmp_path / "analysis", 10000, load_run(b))
    assert json.loads((output / "report.json").read_text())["comparison"]["outcome"] == outcome


@pytest.mark.parametrize("timeout", [False, True])
def test_notebook_watchdog_stops_worker_descendants_and_preserves_failure(
    tmp_path: Path, timeout: bool
) -> None:
    source = tmp_path / "source"
    pkg = source / "src/gemma_agent_dev"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    marker = tmp_path / "leaked"
    child = f"import time;from pathlib import Path;time.sleep(2);Path({str(marker)!r}).touch()"
    (pkg / "worker.py").write_text(
        "import subprocess,sys,time\n"
        + f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'
        + ("time.sleep(30)\n" if timeout else 'print("finished",flush=True)\n')
    )
    storage = tmp_path / "artifacts"
    results = storage / "results/run"
    results.mkdir(parents=True)
    write_json(results / "manifest.json", {"status": "running", "source_sha256": "fixture"})
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            run_bounded(
                source,
                {"source_sha256": "fixture"},
                verify_only=False,
                diagnostic=False,
                timeout=0.2,
                progress_interval=0.05,
                storage=storage,
            )
        assert json.loads((results / "manifest.json").read_text())["phase"] == "watchdog"
    else:
        run_bounded(
            source,
            {"source_sha256": "fixture"},
            verify_only=True,
            diagnostic=False,
            timeout=5,
            progress_interval=0.05,
            storage=storage,
        )
    import time

    time.sleep(2.1)
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["valid", "reserve", "insufficient", "environment", "wrong-repo"])
def test_qualification_freezes_controls_and_uses_reserves_before_model_results(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    settings = diagnostic_project
    config_path = settings.root / "qualification.yaml"
    tasks = [
        replace(
            settings.tasks[0],
            instance_id=f"fixture_{i}",
            generated_patch=None,
            generated_sha256=None,
            probe_replacements=[],
        )
        for i in (1, 2, 3)
    ]
    settings = replace(settings, tasks=tasks)
    records = {t.instance_id: {"repo": "fixture/repo", "patch": "reference"} for t in tasks}
    if mode == "wrong-repo":
        records["fixture_1"]["repo"] = "other/other"
    monkeypatch.setattr(DiagnosticSettings, "load", lambda *args: settings)
    monkeypatch.setattr(DiagnosticSettings, "read_tasks", lambda self: records)
    monkeypatch.setattr(qualification, "check_prepared", lambda s: {})
    config_path.write_text(
        yaml.safe_dump(
            {
                "diagnostic_config": "diagnostics.yaml",
                "per_repository": 1,
                "candidates": {"repo": [t.instance_id for t in tasks]},
            }
        )
    )
    calls = []

    def case(s: Any, task: Any, data: Any, name: str, patch: str, output: Path) -> dict[str, Any]:
        calls.append((task.instance_id, name))
        output.mkdir(parents=True)
        passed = name == "reference"
        if mode == "reserve" and task.instance_id == "fixture_1":
            passed = True
        if mode == "insufficient":
            passed = False
        (output / "junit.xml").write_text(
            "<testsuite><testcase>" + ("" if passed else "<failure/>") + "</testcase></testsuite>"
        )
        result = {
            "exit_code": 0 if passed else 1,
            "task_id": task.instance_id,
            "case": name,
            "passed": passed,
            "source_origins": {"valid": True},
            "artifacts": inventory(output),
        }
        if mode == "environment":
            result["error"] = "collection error"
        write_json(output / "case.json", result)
        return result

    monkeypatch.setattr(qualification, "run_case", case)
    if mode in {"insufficient", "environment", "wrong-repo"}:
        with pytest.raises((ValueError, RuntimeError)):
            qualification.qualify(settings.root, config_path)
        assert not (settings.output_dir / "qualified.json").exists()
    else:
        pointer = qualification.qualify(settings.root, config_path)
        report = qualification.load_selection(settings.root, pointer)
        assert report["task_ids"] == ["fixture_2" if mode == "reserve" else "fixture_1"]
        assert len(calls) == (4 if mode == "reserve" else 2)
        with pytest.raises(FileExistsError):
            qualification.qualify(settings.root, config_path)
        case_path = Path(json.loads(pointer.read_text())["report"])
        case_folder = settings.root / case_path.parent / report["task_ids"][0] / "reference"
        (case_folder / "case.json").write_text("{}")
        with pytest.raises(ValueError, match="artifacts"):
            qualification.load_selection(settings.root, pointer)


def test_runtime_checks_prepare_is_separate_and_normal_check_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/runtime.json", root / "configs/runtime.json")
    for name in ("requirements.in", "requirements.lock", "sampling.yaml"):
        (root / name).write_text("fixture")
    config = {
        "runtime_config": "configs/runtime.json",
        "requirements": "requirements.in",
        "lock_file": "requirements.lock",
        "environment": "artifacts/sdk",
        "wheel_cache": "artifacts/wheels",
        "output_dir": "artifacts/checks",
        "timeout_seconds": 10,
        "python_version": "3.12",
        "sampling": "sampling.yaml",
    }
    path = root / "checks.yaml"
    path.write_text(yaml.safe_dump(config))
    runtime = json.loads((root / "configs/runtime.json").read_text())
    runtime["wheelhouse_hashes"] = {
        f"{name.replace('-', '_')}-{v}-py3-none-any.whl": __import__("hashlib")
        .sha256(b"wheel")
        .hexdigest()
        for name, v in runtime["packages"].items()
    }
    write_json(root / "configs/runtime.json", runtime)
    calls = []

    def command(args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        if "-c" in args:
            return json.dumps({"python": "3.12.1", "packages": {"fixture": "1"}})
        if args[0] == "kaggle":
            (Path(args[-2]) / args[5]).write_bytes(b"wheel")
        if args[:2] == ["uv", "venv"]:
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        if "-m" in args:
            write_json(Path(args[-1]) / "report.json", {"status": "completed", "gpu_modules": []})
        return "ok"

    monkeypatch.setattr(runtime_checks, "command", command)
    receipt = runtime_checks.prepare(root, path)
    assert receipt.is_file()
    first_downloads = sum(c[0] == "kaggle" for c in calls)
    runtime_checks.prepare(root, path)
    assert first_downloads == 3 and sum(c[0] == "kaggle" for c in calls) == 3
    calls.clear()
    result = runtime_checks.check(root, path)
    assert (result / "check.log").read_text() == "ok" and all(c[0] != "kaggle" for c in calls)
    (root / "requirements.lock").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runtime_checks.check(root, path)


@pytest.mark.parametrize("fault", ["none", "seed", "budget", "retention", "disabled"])
def test_cpu_request_contract_rejects_changed_wire_behavior(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import types

    from gemma_agent_dev.official_cpu import request_contract

    captured: dict[str, Any] = {}

    class Client:
        pass

    class Model:
        def __init__(self, **kwargs: Any) -> None:
            self.client = kwargs["llm_client"]

        async def generate_content_async(self, request: Any) -> Any:
            config = request.config.__dict__
            thinking = config["thinking_config"]
            kwargs = {
                "seed": 1 if fault == "seed" else config["seed"],
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": thinking["thinking_budget"] > 0},
                    "thinking_token_budget": 1
                    if fault == "budget"
                    else thinking["thinking_budget"],
                },
                "messages": [{"reasoning": "retained thought"}]
                if thinking["include_thoughts"] and fault != "retention"
                else [],
            }
            captured.update(kwargs)
            await self.client.acompletion(**kwargs)
            yield {"synthetic": True}

    def module(name: str, **members: Any) -> None:
        value = types.ModuleType(name)
        for key, item in members.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)

    module(
        "adk_submission.resolvers.generation",
        apply_thinking_config_to_model=lambda model, cfg: model,
    )
    module("google.adk.models.lite_llm", LiteLlm=Model, LiteLLMClient=Client)
    module("google.adk.models.llm_request", LlmRequest=lambda **kw: SimpleNamespace(**kw))
    module(
        "google.genai",
        types=SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            Content=lambda **kw: SimpleNamespace(**kw),
            Part=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    module("litellm", ModelResponse=lambda **kw: kw)
    sampling = {
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 0,
        "max_output_tokens": 16384,
        "thinking_config": {
            "thinking_budget": 0 if fault == "disabled" else 4096,
            "include_thoughts": False if fault == "disabled" else True,
        },
    }
    if fault in {"seed", "budget", "retention"}:
        with pytest.raises(ValueError):
            asyncio.run(request_contract(sampling))
    else:
        report = asyncio.run(request_contract(sampling))
        assert report["seed"] == 0 and bool(report["retained_thoughts"]) is (fault != "disabled")
    assert captured


@pytest.mark.parametrize("gpu_imported", [False, True])
def test_cpu_compilation_checks_all_aliases_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu_imported: bool
) -> None:
    import types

    from gemma_agent_dev import official_cpu

    module = types.ModuleType("swegemma.config")
    module.EvalConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swegemma.config", module)
    runtime = RuntimeSettings.load(ROOT / "configs/runtime.json")
    aliases = []
    stopped = []
    server = SimpleNamespace(
        create_model_registry=lambda **kw: SimpleNamespace(get=lambda name: aliases.append(name)),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setattr(official_cpu, "create_server", lambda *args: server)
    monkeypatch.setattr(official_cpu, "make_validator", lambda *args: lambda source: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: runtime.packages[name])

    async def request(*args: Any) -> dict[str, Any]:
        return {"seed": 0}

    monkeypatch.setattr(official_cpu, "request_contract", request)
    if gpu_imported:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        with pytest.raises(ValueError, match="GPU"):
            official_cpu.run(
                ROOT / "configs/runtime.json", ROOT / "agent/configs/sampling.yaml", tmp_path
            )
    else:
        official_cpu.run(
            ROOT / "configs/runtime.json", ROOT / "agent/configs/sampling.yaml", tmp_path
        )
        assert json.loads((tmp_path / "report.json").read_text())["status"] == "completed"
    assert len(aliases) == 3 and stopped == [True]


def test_source_plugin_records_import_errors_and_excludes_external_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    from gemma_agent_dev import provenance

    (tmp_path / "example.py").write_text("value=1")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("GEMMA_SOURCE_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GEMMA_SOURCE_MODULES", '["example"]')
    monkeypatch.setenv("GEMMA_SOURCE_EVIDENCE", str(tmp_path / "origins.json"))
    monkeypatch.setattr(provenance, "IMPORT_ERRORS", {})
    real_import = importlib.import_module

    def fail(name: str) -> Any:
        if name == "example":
            raise ImportError("missing dependency")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    provenance.pytest_sessionstart(None)
    external = types.ModuleType("third_party")
    external.__file__ = "/outside/third_party.py"
    monkeypatch.setitem(sys.modules, "example.alias", external)
    session = SimpleNamespace(exitstatus=2)
    provenance.pytest_sessionfinish(session, 2)
    evidence = validate_origins(tmp_path / "origins.json")
    assert (
        evidence["import_errors"]["example"] == "missing dependency"
        and "example.alias" not in evidence["modules"]
    )
    assert session.exitstatus == 2


@pytest.mark.parametrize("fault", ["none", "checksum", "install", "stale"])
def test_offline_sandbox_dependencies_and_stale_source_evidence_are_guarded(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "sandbox"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "example.py").write_text("value=1")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "helper-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    installs = []

    class Sandbox:
        sandboxes = {"one": {"root": root, "workspace": workspace, "venv": root / "venv"}}

        def exec(self, sid: str, command: str, **kwargs: Any) -> Any:
            if "pip install" in command:
                installs.append(command)
                return SimpleNamespace(exit_code=1 if fault == "install" else 0)
            if command.startswith("cat "):
                return SimpleNamespace(stdout="<testsuite><testcase/></testsuite>")
            if fault != "stale":
                write_json(
                    root / "source-guard/origins.json",
                    {
                        "valid": True,
                        "workspace": str(workspace),
                        "modules": {"example": str(workspace / "example.py")},
                    },
                )
            return SimpleNamespace(stdout="done")

    spec = {
        "source_roots": ["."],
        "import_modules": ["example"],
        "dependency_wheels": {wheel.name: "0" * 64 if fault == "checksum" else file_hash(wheel)},
    }
    sandbox = Sandbox()
    check = observe_sources(sandbox, {"one": spec}, tmp_path / "out", wheels)
    if fault in {"checksum", "install"}:
        with pytest.raises((ValueError, RuntimeError)):
            sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
    elif fault == "stale":
        (root / "source-guard").mkdir()
        write_json(
            root / "source-guard/origins.json",
            {
                "valid": True,
                "workspace": str(workspace),
                "modules": {"example": str(workspace / "example.py")},
            },
        )
        sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
        with pytest.raises(ValueError):
            check()
    else:
        sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
        check()
        (workspace / "example.py").unlink()
        sandbox.exec("one", "echo done")
        assert len(installs) == 1


@pytest.mark.parametrize("diagnostic", [False, True])
def test_worker_dispatches_only_the_selected_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, diagnostic: bool
) -> None:
    import types

    from gemma_agent_dev import worker

    module_name = "gemma_agent_dev.diagnostic_kaggle" if diagnostic else "gemma_agent_dev.runtime"
    module = types.ModuleType(module_name)
    calls = []
    module.run = lambda *args, **kwargs: calls.append((args, kwargs))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), "{}", "1", str(int(diagnostic))])
    worker.main()
    assert calls and (calls[0][1] == {} if diagnostic else calls[0][1] == {"verify_only": True})


@pytest.mark.parametrize(
    "consumed,active,code", [("0", False, 0), ("2", False, 1), ("0", True, 1), ("nan", False, 1)]
)
def test_cli_budget_admission_has_a_failure_exit_status(
    diagnostic_project: DiagnosticSettings,
    monkeypatch: pytest.MonkeyPatch,
    consumed: str,
    active: bool,
    code: int,
) -> None:
    from gemma_agent_dev.cli import main

    root = diagnostic_project.root
    budget = yaml.safe_load((ROOT / "configs/gpu-budget.yaml").read_text())
    budget["runtime_config"] = "runtime.json"
    (root / "budget.yaml").write_text(yaml.safe_dump(budget))
    monkeypatch.chdir(root)
    args = [
        "check-gpu-budget",
        "--config",
        "budget.yaml",
        "--consumed-hours",
        consumed,
        "--remaining-hours",
        "20",
        "--completed-runs",
        "0",
    ]
    assert main(args + (["--active"] if active else [])) == code


def test_profiles_are_explicit_and_unknown_names_do_not_start_preparation(
    preflight_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.cli import main
    from gemma_agent_dev.preflight import PreflightSettings

    monkeypatch.chdir(preflight_project)
    with pytest.raises(SystemExit) as raised:
        main(["preflight", "--config", "preflight.yaml"])
    assert raised.value.code == 2
    with pytest.raises(ValueError, match="Unknown"):
        PreflightSettings.load(preflight_project, Path("preflight.yaml"), "unknown")
    assert not (preflight_project / "artifacts/preflight").exists()


def test_catalog_cannot_pin_two_hashes_for_the_same_wheel(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    first = data["tasks"][0]
    first["dependency_wheels"] = {"helper-1-py3-none-any.whl": "0" * 64}
    data["tasks"].append(
        {
            **first,
            "instance_id": "fixture_2",
            "dependency_wheels": {"helper-1-py3-none-any.whl": "1" * 64},
        }
    )
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Conflicting"):
        TaskCatalog.load(settings.root, settings.catalog_path)


@pytest.mark.parametrize(
    "summary,expected",
    [
        ("6 failed, 18 passed in 1.24s", {"failed": 6, "passed": 18, "executed": 24}),
        ("1 skipped in 0.01s", {"skipped": 1, "executed": 0}),
        ("1 passed, 1 error in 0.01s", {"passed": 1, "errors": 1, "executed": None}),
        ("no tests ran in 0.01s", {"reported_total": 0, "executed": 0}),
        (
            "ERROR collecting tests/test_a.py\n1 error in 0.03s",
            {"collection_errors": 1, "errors": 0, "executed": 0},
        ),
        ("print-only: 24 passed", None),
    ],
)
def test_analysis_reads_reported_counts_without_replacing_official_grade(
    make_run: Any, tmp_path: Path, summary: str, expected: dict[str, int | None] | None
) -> None:
    path = make_run()
    folder = path / "test_outputs"
    folder.mkdir()
    (folder / "task_b.log").write_text(summary)
    (path / "logs").mkdir()
    (path / "logs/task_b.log").write_text("Agent execution evidence")
    (path / "patches").mkdir()
    (path / "patches/task_b.patch").write_text("--- a/example.py\n+++ b/example.py\n")
    before = inventory(path)
    output = analysis.analyze(load_run(path), tmp_path / "analysis", 10000)
    report = json.loads((output / "report.json").read_text())
    evidence = report["tasks"]["task_b"]
    assert evidence["official_result"]["resolved"] is False
    counts = evidence["individual_test_counts"]
    if expected is None:
        assert counts is None
    else:
        assert {key: counts[key] for key in expected} == expected
    assert inventory(path) == before and str(folder / "task_b.log") in report["source_hashes"]
    for name in ("logs/task_b.log", "patches/task_b.patch"):
        assert report["source_hashes"][str(path / name)] == file_hash(path / name)


@pytest.mark.parametrize(
    "error,time_limit,tool_limit",
    [
        (None, None, None),
        ("Agent exceeded session timeout (1 min)", True, None),
        ("Agent exceeded tool call budget (10 calls)", None, True),
        ("Verification timed out after 300 seconds", None, None),
        ("Sandbox execution error: litellm.ContextWindowExceededError: input too long", None, None),
    ],
)
def test_analysis_does_not_misread_configured_limits_as_exhausted_budgets(
    make_run: Any, error: str | None, time_limit: bool | None, tool_limit: bool | None
) -> None:
    path = make_run()
    rows = [json.loads(line) for line in (path / "task_results.jsonl").read_text().splitlines()]
    rows[1]["error"] = error
    (path / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "traces").mkdir()
    write_json(
        path / "traces/trace_task_b.json",
        {
            "steps": [
                {
                    "tool_calls": [{"function_name": "get_status", "arguments": {}}],
                    "observation": {
                        "content": '{"timeout":60,"tool_call_limit":30,"max_tokens":16384}'
                    },
                }
            ]
        },
    )
    evidence = analysis.task_evidence(load_run(path), "task_b", 10000)
    assert (
        evidence["time_limit_reached"] is time_limit
        and evidence["tool_limit_reached"] is tool_limit
    )
    assert evidence["verification_timeout"] is (
        True if error and error.startswith("Verification") else None
    )
    assert evidence["context_window_exceeded"] is (
        True if error and "ContextWindowExceededError" in error else None
    )


def test_diagnostic_order_survives_sorted_mapping_serialization(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata

    from gemma_agent_dev import diagnostic_kaggle, environment

    settings = diagnostic_project
    root = settings.root
    runtime = settings.runtime.as_dict()
    runtime["paths"]["output"] = str(root / "artifacts/remote-results")
    ids = ["fixture_z", "fixture_a"]
    spec = {
        "runtime": runtime,
        "task_ids": ids,
        "fingerprint": "fixture",
        "tasks_sha256": settings.tasks_sha256,
        "snapshots": {key: "0" * 64 for key in reversed(ids)},
        "timeout_seconds": settings.timeout_seconds,
        "lock_sha256": "0" * 64,
        "cases": [{"task_id": key, "case": "reference"} for key in ids],
    }
    (root / "diagnostics.json").write_text(json.dumps(spec, sort_keys=True))
    monkeypatch.setattr(environment, "install_wheelhouse", lambda *args: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: settings.runtime.packages[name])
    captured = []

    async def verify(runtime: Any, spec: Any, output: Any, report: Any) -> None:
        captured.append(runtime.task_ids)
        assert [case["task_id"] for case in spec["cases"]] == runtime.task_ids

    monkeypatch.setattr(diagnostic_kaggle, "verify_cases", verify)
    diagnostic_kaggle.run(root, {"source_sha256": "fixture"})
    assert captured == [ids]


def test_analysis_preserves_unknown_for_truncation_and_provider_timeouts(make_run: Any) -> None:
    assert analysis.reported_test_counts("1 passed in 0.01s\n[Display truncated]") is None
    path = make_run()
    rows = [json.loads(line) for line in (path / "task_results.jsonl").read_text().splitlines()]
    rows[1]["error"] = "Agent failed due to provider timeout"
    (path / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert analysis.task_evidence(load_run(path), "task_b", 10000)["time_limit_reached"] is None


def test_compare_displays_explicit_task_order(
    diagnostic_project: DiagnosticSettings,
    make_run: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from gemma_agent_dev.cli import main

    root = diagnostic_project.root
    folder = root / "artifacts/results"
    folder.mkdir()
    for name in ["left", "right"]:
        shutil.copytree(make_run(name, ids=("task_z", "task_a")), folder / name)
    monkeypatch.chdir(root)
    assert main(["compare", "left", "right"]) == 0
    rows = [
        line.split("\t")[0]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("task_z\t", "task_a\t"))
    ]
    assert rows == ["task_z", "task_a"]


@pytest.mark.parametrize("damage", ["permuted", "missing", "duplicate"])
def test_analysis_compares_actual_sdk_task_order_not_only_selected_ids(
    make_run: Any, tmp_path: Path, damage: str
) -> None:
    left = make_run("left")
    right = make_run("right")
    manifest = json.loads((right / "manifest.json").read_text())
    manifest["submission_files"]["prompts/system.md"] = "changed"
    write_json(right / "manifest.json", manifest)
    rows = [json.loads(line) for line in (right / "task_results.jsonl").read_text().splitlines()]
    if damage == "permuted":
        rows[0]["task_index"], rows[1]["task_index"] = 2, 1
    elif damage == "missing":
        rows[0].pop("task_index")
    else:
        rows[1]["task_index"] = 1
    (right / "task_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = analysis.analyze(load_run(left), tmp_path / "analysis", 10000, load_run(right))
    report = json.loads((output / "report.json").read_text())
    assert (
        report["comparison"]["outcome"] == "invalid"
        and report["comparison"]["task_order_verified"] is False
    )
    assert "actual_task_order" in report["comparison"]["differences"]
