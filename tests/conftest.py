import json
import shlex
import shutil
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import content_hash, file_hash, inventory, write_json
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import installed_environment
from gemma_agent_dev.evaluation import experiment_preparation
from gemma_agent_dev.evaluation import session as session_module
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.notebooks import experiments as bundles

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
            "snapshots": {i: "fixture" for i in ids},
            "wheelhouse_hashes": {"fixture.whl": "fixture"},
            "dependencies": {"fixture": "1"},
            "source_setup": {"fixture": "source"},
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
                "task_index": index + 1,
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
    (root / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    shutil.copytree(ROOT / "src", root / "src")
    config = RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml").as_dict()
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
    data: dict[str, Any] = {
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
                "collection_error_cases": [],
            }
        ],
    }
    catalog = {
        "wheel_cache": "artifacts/reference/task-wheels",
        "tasks_file": data.pop("tasks_file"),
        "tasks_sha256": data.pop("tasks_sha256"),
        "tasks": [
            {
                "instance_id": "fixture_1",
                "test_paths": data["tasks"][0].pop("test_paths"),
                "snapshot_sha256": data["tasks"][0].pop("snapshot_sha256"),
                "source_roots": ["."],
                "import_modules": ["calc"],
                "dependency_wheels": {},
            }
        ],
    }
    (root / "catalog.yaml").write_text(yaml.safe_dump(catalog))
    data["catalog_path"] = "catalog.yaml"
    (root / "diagnostics.yaml").write_text(yaml.safe_dump(data))
    settings = DiagnosticSettings.load(root, Path("diagnostics.yaml"))
    write_json(
        cache / "prepared.json",
        {
            "fingerprint": settings.preparation_fingerprint(),
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
                "profiles": "configs/evaluation/profiles.yaml",
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
    from gemma_agent_dev.candidates import preflight

    @contextmanager
    def quality(args: list[str], **kwargs: Any) -> Iterator[Any]:
        assert args == ["make", "verify"]
        assert kwargs["env"]["UV_OFFLINE"] == "1"
        kwargs["stdout"].write("Quality checks passed (fixture)\n")
        yield SimpleNamespace(args=args, wait=lambda **_: 0)

    monkeypatch.setattr(preflight, "process_group", quality)
    return preflight.run(preflight_project, Path("preflight.yaml"), "smoke")


@pytest.fixture
def cpu_evidence(preflight_project: Path, preflight_report: Path) -> Path:
    from gemma_agent_dev.evaluation.config import RuntimeSettings
    from gemma_agent_dev.notebooks.evaluation import prepare

    folder = preflight_project / "artifacts/cpu"
    folder.mkdir(parents=True)
    digest = prepare(
        preflight_project,
        preflight_project / "cpu-source",
        report_path=preflight_report,
        verify_only=True,
    )
    config = RuntimeSettings.load(preflight_project / "runtime.json")
    write_json(folder / "origins.json", {"valid": True})
    write_json(
        folder / "controls.json",
        [
            {
                "instance_id": task,
                "repo": "fixture/repo",
                **{
                    label: {
                        "resolved": label == "reference",
                        "exit_code": 0 if label == "reference" else 1,
                        "error": None,
                        "duration_seconds": 1,
                    }
                    for label in ("unmodified", "reference")
                },
            }
            for task in config.task_ids
        ],
    )
    write_json(
        folder / "manifest.json",
        {
            "schema_version": 1,
            "kind": "verification",
            "run_id": "cpu",
            "status": "completed",
            "task_ids": config.task_ids,
            "controls_passed": True,
            "source_sha256": digest,
            "candidate_sha256": json.loads(preflight_report.read_text())["candidate_sha256"],
            "wheelhouse_hashes": config.wheelhouse_hashes,
            "control_artifacts": {
                name: file_hash(folder / name) for name in ("origins.json", "controls.json")
            },
        },
    )
    return folder / "manifest.json"


@pytest.fixture
def experiment_project(preflight_project: Path) -> Path:
    root = preflight_project
    pool = {
        "pool_id": "fixture-pool",
        "difficulty": "Easy",
        "tasks_sha256": file_hash(root / "tasks.jsonl"),
        "sdk_version": "0.2.11",
        "tuning": ["fixture_1"],
        "confirmation": ["unused_1"],
        "uses": [
            {"experiment_id": "fixture-experiment", "purpose": "tuning", "task_ids": ["fixture_1"]}
        ],
    }
    (root / "pool.yaml").write_text(yaml.safe_dump(pool))
    config = yaml.safe_load((root / "configs/evaluation/easy-sequential-v1.yaml").read_text())
    config.update(
        experiment_id="fixture-experiment",
        hypothesis="fixture-hypothesis",
        pool="pool.yaml",
        diagnostic_config="diagnostics.yaml",
        preflight_config="preflight.yaml",
        runtime_config="runtime.json",
        evaluation_config="agent/eval_config.yaml",
    )
    config["session"]["preparation_timeout_seconds"] = 60
    config["session"]["evaluation_timeout_seconds"] = 660
    (root / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return root


@pytest.fixture
def frozen_experiment(experiment_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def quality(settings: Any, output: Path) -> None:
        (output / "quality.log").write_text("fixture quality")

    monkeypatch.setattr(experiment_preparation, "verify_quality", quality)
    return experiment_preparation.prepare(experiment_project, Path("experiment.yaml"))


session: Any = session_module


@pytest.fixture
def session_fixture(
    experiment_project: Path, frozen_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], list[str]]:
    root = experiment_project
    bundle = root / "bundle"
    bundles.prepare(root, bundle, report_path=frozen_experiment, verify_only=True, cpu_report=None)
    from gemma_agent_dev.candidates.archive import extract_archive

    extracted = root / "extracted"
    extract_archive(bundle / "source-bundle.zip", extracted, 10000000)
    runtime = json.loads((extracted / "runtime.json").read_text())
    runtime["paths"]["output"] = str(root / "artifacts/results")
    runtime["paths"]["data"] = str(root / "data")
    write_json(extracted / "runtime.json", runtime)
    spec = json.loads((extracted / "experiment.json").read_text())
    spec["definition"]["runtime"] = runtime
    spec["definition_sha256"] = content_hash(spec["definition"])
    write_json(extracted / "experiment.json", spec)
    data = root / "data"
    data.mkdir()
    (data / "snapshots").mkdir()
    (data / "tasks.jsonl").write_bytes((root / "tasks.jsonl").read_bytes())
    (data / "snapshots/fixture_1.tgz").write_bytes(b"fixture snapshot")
    calls = []

    def module(name: str, **attrs: Any) -> None:
        for index in range(1, len(name.split(".")) + 1):
            key = ".".join(name.split(".")[:index])
            m = types.ModuleType(key)
            m.__path__ = []
            monkeypatch.setitem(sys.modules, key, m)
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)

    module(
        "swegemma.models",
        load_tasks=lambda p: [
            SimpleNamespace(
                instance_id="fixture_1", repo="fixture/repo", patch="fix", test_patch="test"
            )
        ],
    )
    module(
        "swegemma.deduplication",
        resolve_task_snapshot_paths=lambda p, t, r: (data / "snapshots/fixture_1.tgz", None, None),
    )
    module("swegemma.config", EvalConfig=lambda **kw: SimpleNamespace(**kw))
    module("google.adk.agents.context_cache_config", ContextCacheConfig=lambda **kw: kw)
    module("google.adk.apps._configs", EventsCompactionConfig=lambda **kw: kw)
    module("litellm")
    module("torch", cuda=SimpleNamespace(device_count=lambda: 4))

    @dataclass
    class ServerConfig:
        cache: bool = False

    class Server:
        config = ServerConfig()
        adapter_manifest = None
        log_path = str(root / "model.log")

        def create_model_registry(self, **kw: Any) -> dict[str, str]:
            calls.append("registry")
            return {}

        def start(self) -> None:
            calls.append("start")

        def stop(self) -> None:
            calls.append("stop")
            Path(self.log_path).unlink(missing_ok=True)

    Path(Server.log_path).write_text("server log")

    def create(directory: Path, config: dict[str, Any]) -> Any:
        assert config["vllm"]["extra_args"] == ["--no-enable-prefix-caching"]
        calls.append("create")
        Path(Server.log_path).write_text("server log")
        return Server()

    monkeypatch.setattr(session, "create_server", create)
    monkeypatch.setattr(session, "install_wheelhouse", lambda *a, **kw: calls.append("install"))
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.session.importlib.metadata.version",
        lambda name: runtime["packages"][name],
    )
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.session.importlib.metadata.distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name}, version=version)
            for name, version in runtime["packages"].items()
        ],
    )
    monkeypatch.setattr(session, "make_validator", lambda *a: lambda p: calls.append("compile"))

    async def request_contract(sampling: Any) -> Any:
        return {"generation": sampling}

    monkeypatch.setattr("gemma_agent_dev.evaluation.sdk_worker.request_contract", request_contract)

    async def controls(config: Any, tasks: Any, output: Path, *args: Any) -> bool:
        calls.append("controls")
        write_json(output / "origins.json", {"valid": True})
        write_json(
            output / "controls.json",
            [
                {
                    "instance_id": task,
                    "unmodified": {"resolved": False, "error": None},
                    "reference": {"resolved": True, "error": None},
                }
                for task in config.task_ids
            ],
        )
        return True

    monkeypatch.setattr(session, "verify_controls", controls)

    async def evaluate(config: Any, *args: Any) -> Any:
        calls.append("evaluate")
        assert len(config.task_ids) == 1
        task = config.task_ids[0]
        row = {
            "instance_id": task,
            "task_index": 1,
            "resolved": True,
            "duration_seconds": 1,
            "agent_patch_size": 1,
            "test_exit_code": 0,
            "error": None,
        }
        (config.results_dir / "task_results.jsonl").write_text(json.dumps(row) + "\n")
        write_json(config.results_dir / "summary.json", {"total_tasks": 1, "resolved": 1})
        for path, text in {
            f"logs/{task}.log": "log",
            f"traces/trace_{task}.json": "{}",
            f"patches/{task}.patch": "patch",
            f"test_outputs/{task}.log": "1 passed in 0.01s",
        }.items():
            p = config.results_dir / path
            p.parent.mkdir(exist_ok=True)
            p.write_text(text)
        return SimpleNamespace(total=1, resolved=1)

    monkeypatch.setattr(session, "evaluate", evaluate)
    metadata = json.loads((bundle / "source-manifest.json").read_text())
    metadata["definition_sha256"] = spec["definition_sha256"]
    return extracted, metadata, calls
