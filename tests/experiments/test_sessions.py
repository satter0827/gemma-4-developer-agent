import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.evaluation import session as session_module
from gemma_agent_dev.evaluation.experiment_results import summarize
from gemma_agent_dev.notebooks import experiments as bundles

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
    monkeypatch.setattr(session, "install_wheelhouse", lambda *a: calls.append("install"))
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.session.importlib.metadata.version",
        lambda name: runtime["packages"][name],
    )
    monkeypatch.setattr(
        session, "generation_contract", lambda *a: [{"model": "same", "sampling": "same"}]
    )
    monkeypatch.setattr(session, "make_validator", lambda *a: lambda p: calls.append("compile"))

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
    return extracted, metadata, calls


def test_one_server_and_independent_cells(session_fixture: Any) -> None:
    root, provenance, calls = session_fixture
    path = session.run(root, provenance)
    report = json.loads(path.read_text())
    assert report["status"] == "completed"
    assert calls.count("start") == calls.count("stop") == calls.count("create") == 1
    assert calls.count("evaluate") == 2 and calls.count("compile") == 2
    assert calls.count("registry") == 4
    assert report["actual_order"] == report["planned_order"]
    assert len({c["manifest"] for c in report["cells"]}) == 2
    result = summarize(path)
    assert result["counts"]["both"] == 1 and result["formal_score"] is None
    assert (path.parent / "vllm.log").read_text() == "server log"
    second = session.run(root, provenance)
    from gemma_agent_dev.evaluation.cli import load_runs

    runs, errors = load_runs(path.parent.parent.resolve())
    assert second != path and not errors and len(runs) == 4
    assert len({run.manifest["run_id"] for run in runs}) == 4


@pytest.mark.parametrize(
    "fault",
    [
        "candidate",
        "snapshot",
        "tasks",
        "sampling",
        "controls",
        "evaluation",
        "save",
        "cleanup",
        "startup",
        "gpu",
        "packages",
    ],
)
def test_failure_persisted_and_server_stopped(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root, provenance, calls = session_fixture
    runtime = json.loads((root / "runtime.json").read_text())
    if fault == "candidate":
        (root / "sequential-v1/submission.zip").write_bytes(b"changed")
    elif fault == "snapshot":
        Path(runtime["paths"]["data"]).joinpath("snapshots/fixture_1.tgz").write_bytes(b"changed")
    elif fault == "tasks":
        Path(runtime["paths"]["data"]).joinpath("tasks.jsonl").write_text("changed")
    elif fault == "sampling":
        original = session.validate_references

        def altered(p: Path) -> Any:
            files = original(p)
            if p.name == "sequential-v1":
                files["configs/sampling.yaml"] = "changed"
            return files

        monkeypatch.setattr(session, "validate_references", altered)
    elif fault == "controls":

        async def bad(*a: Any) -> bool:
            return False

        monkeypatch.setattr(session, "verify_controls", bad)
    elif fault == "evaluation":

        async def fail(*a: Any) -> Any:
            raise RuntimeError("evaluate failed")

        monkeypatch.setattr(session, "evaluate", fail)
    elif fault == "save":
        original_write = session.write_json

        def write(path: Path, value: Any) -> None:
            if path.name == "manifest.json" and value.get("status") == "completed":
                raise OSError("save failed")
            original_write(path, value)

        monkeypatch.setattr(session, "write_json", write)
    elif fault in {"cleanup", "startup"}:
        create = session.create_server

        def broken(*a: Any) -> Any:
            server = create(*a)

            def fail() -> None:
                calls.append("stop" if fault == "cleanup" else "start")
                raise RuntimeError("server failed")

            setattr(server, "stop" if fault == "cleanup" else "start", fail)
            return server

        monkeypatch.setattr(session, "create_server", broken)
    elif fault == "gpu":
        sys.modules["torch"].cuda.device_count = lambda: 0
    else:
        monkeypatch.setattr(
            "gemma_agent_dev.evaluation.session.importlib.metadata.version", lambda name: "changed"
        )
    with pytest.raises((ValueError, RuntimeError, OSError)):
        session.run(root, provenance)
    output = next(Path(runtime["paths"]["output"]).glob("*/session.json"))
    assert json.loads(output.read_text())["status"] == "failed"
    if "create" in calls:
        assert calls[-1] == "stop"


def test_pair_not_started_when_remaining_time_is_insufficient(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, provenance, calls = session_fixture
    elapsed = [0.0]
    monkeypatch.setattr("gemma_agent_dev.evaluation.session.time.monotonic", lambda: elapsed[0])
    create = session.create_server

    def delayed(*args: Any) -> Any:
        server = create(*args)
        start = server.start

        def start_delayed() -> None:
            start()
            elapsed[0] = 5000

        server.start = start_delayed
        return server

    monkeypatch.setattr(session, "create_server", delayed)
    path = session.run(root, provenance)
    report = json.loads(path.read_text())
    assert report["status"] == "incomplete" and report["actual_order"] == []
    assert "evaluate" not in calls and calls[-1] == "stop"
    assert summarize(path)["counts"]["not_run"] == 1


def test_cpu_proof_checks_artifacts_and_bundle_identity(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, provenance, calls = session_fixture
    from gemma_agent_dev.evaluation import sequential_checks

    monkeypatch.setattr(sequential_checks, "run", lambda *a: {"status": "completed"})
    path = session.run(root, provenance, verify_only=True)
    report = json.loads(path.read_text())
    assert (
        report["kind"] == "experiment-verification"
        and "start" not in calls
        and "evaluate" not in calls
    )
    bundles.load_cpu(
        path,
        provenance["source_sha256"],
        provenance["definition_sha256"],
        report["task_ids"],
        report["variants"],
    )
    with pytest.raises(ValueError):
        bundles.load_cpu(
            path, "changed", provenance["definition_sha256"], report["task_ids"], report["variants"]
        )
    with pytest.raises(ValueError):
        summarize(path)
    first = next((path.parent / "controls").rglob("origins.json"))
    first.write_text("{}")
    with pytest.raises(ValueError, match="artifacts"):
        bundles.load_cpu(
            path,
            provenance["source_sha256"],
            provenance["definition_sha256"],
            report["task_ids"],
            report["variants"],
        )


def test_local_proof_keeps_native_and_sdk_environments_separate(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, provenance, calls = session_fixture
    spec = json.loads((root / "experiment.json").read_text())
    project = root.parent
    prepared = json.loads(
        next((project / "artifacts/experiments").glob("*/preparation.json")).read_text()
    )
    native = project / prepared["controls"]["path"]
    runtime = json.loads((root / "runtime.json").read_text())
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.sequential_checks.run", lambda *a: {"status": "completed"}
    )
    bindings = {
        "data": runtime["paths"]["data"],
        "output": str(project / "artifacts/local-proof"),
        "native_controls": str(native),
    }
    path = session.run(root, provenance, verify_only=True, bindings=bindings)
    report = json.loads(path.read_text())
    assert report["control_engine"] == "native-python"
    assert "controls" not in calls and "install" not in calls and "start" not in calls
    bundles.load_cpu(
        path,
        provenance["source_sha256"],
        spec["definition_sha256"],
        report["task_ids"],
        report["variants"],
    )
    with pytest.raises(ValueError, match="restricted"):
        session.run(root, provenance, bindings=bindings)
    from gemma_agent_dev.evaluation.control_evidence import copy_native_controls

    with pytest.raises(ValueError, match="changed"):
        copy_native_controls(native, "changed", ["fixture_1"], project / "bad-proof")
    data = json.loads(native.read_text())
    data["cases"] *= 2
    write_json(native, data)
    with pytest.raises(ValueError, match="duplicated"):
        copy_native_controls(native, file_hash(native), ["fixture_1"], project / "bad-proof")


def test_inline_sampling_difference_blocks_shared_model(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, provenance, calls = session_fixture
    monkeypatch.setattr(
        session,
        "generation_contract",
        lambda p: [{"temperature": 0.2 if p.name == "workflow-v1" else 0.3}],
    )
    with pytest.raises(ValueError, match="identical adapters"):
        session.run(root, provenance)
    assert "start" not in calls and "create" not in calls
