import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.evaluation import session as session_module
from gemma_agent_dev.evaluation.experiment_results import load_cpu, summarize

session: Any = session_module


def test_one_server_and_independent_cells(session_fixture: Any) -> None:
    root, provenance, calls = session_fixture
    path = session.run(root, provenance)
    report = json.loads(path.read_text())
    assert report["status"] == "completed"
    assert calls.count("start") == calls.count("stop") == calls.count("create") == 1
    assert calls.count("evaluate") == 2 and calls.count("compile") == 2
    assert calls.count("controls") == 1
    assert report["control_evidence"]["directory"] == "controls/shared"
    assert calls.count("registry") == 4
    assert report["actual_order"] == report["planned_order"]
    assert len({c["manifest"] for c in report["cells"]}) == 2
    result = summarize(path, max_bytes=2000000)
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
    assert summarize(path, max_bytes=2000000)["counts"]["not_run"] == 1


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
    load_cpu(
        path,
        provenance["source_sha256"],
        provenance["definition_sha256"],
        report["task_ids"],
        report["variants"],
    )
    report["cpu_worker_completion_required"] = True
    write_json(path, report)
    with pytest.raises(ValueError, match="cleanup"):
        load_cpu(
            path,
            provenance["source_sha256"],
            provenance["definition_sha256"],
            report["task_ids"],
            report["variants"],
        )
    report["cpu_worker_completed"] = True
    write_json(path, report)
    load_cpu(
        path,
        provenance["source_sha256"],
        provenance["definition_sha256"],
        report["task_ids"],
        report["variants"],
    )
    with pytest.raises(ValueError):
        load_cpu(
            path, "changed", provenance["definition_sha256"], report["task_ids"], report["variants"]
        )
    with pytest.raises(ValueError):
        summarize(path, max_bytes=2000000)
    first = next((path.parent / "controls").rglob("origins.json"))
    first.write_text("{}")
    with pytest.raises(ValueError, match="artifacts"):
        load_cpu(
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
    load_cpu(
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
        lambda p: [
            {
                "model": "same",
                "generation": {"temperature": 0.2 if p.name == "workflow-v1" else 0.3},
            }
        ],
    )
    with pytest.raises(ValueError, match="identical adapters"):
        session.run(root, provenance)
    assert "start" not in calls and "create" not in calls


@pytest.mark.parametrize("fault", ["initialization", "write", "close"])
def test_logging_fault_preserves_official_outcome_and_cleanup(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, fault: str
) -> None:
    from gemma_agent_dev.common import events

    root, provenance, calls = session_fixture
    original = events.EventHandler.close

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("secret credentials")

    if fault == "initialization":
        monkeypatch.setattr(events.EventHandler, "__init__", fail)
    elif fault == "write":
        monkeypatch.setattr(events.EventHandler, "doRollover", fail)
        monkeypatch.setattr(events.EventHandler, "shouldRollover", lambda *a: True)
    else:

        def close(handler: Any) -> None:
            original(handler)
            fail()

        monkeypatch.setattr(events.EventHandler, "close", close)
    path = session.run(root, {**provenance, "job_id": "job-current"})
    report = json.loads(path.read_text())
    assert report["status"] == "completed" and calls[-1] == "stop"
    assert summarize(path, max_bytes=2000000)["counts"]["both"] == 1
    assert events.load_health(path.parent)["status"] == "degraded"
    for cell in report["cells"]:
        assert json.loads((path.parent / cell["manifest"]).read_text())["job_id"] == "job-current"
    captured = capsys.readouterr()
    assert captured.err.count("Cannot save") == 1 and "secret" not in captured.err


@pytest.mark.parametrize(
    "damage", ["none", "archive", "cpu", "cell", "sdk", "incomplete", "controls", "variant"]
)
def test_paired_registration_uses_original_archive_and_validates_all_evidence(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from gemma_agent_dev.candidates.experiment_evidence import qualify_experiment
    from gemma_agent_dev.evaluation import sequential_checks

    root, provenance, calls = session_fixture
    monkeypatch.setattr(sequential_checks, "run", lambda *a: {"status": "completed"})
    cpu = session.run(root, provenance, verify_only=True)
    gpu = session.run(root, provenance)
    report = json.loads(gpu.read_text())
    variant = "sequential-v1"
    if damage == "archive":
        (gpu.parent / report["candidate_archives"][variant]["path"]).write_bytes(b"changed")
    elif damage == "cpu":
        value = json.loads(cpu.read_text())
        value["evidence_inputs"]["tasks_sha256"] = "changed"
        write_json(cpu, value)
    elif damage in {"cell", "sdk"}:
        cell = report["cells"][0]
        path = gpu.parent / cell["manifest"]
        value = json.loads(path.read_text())
        if damage == "cell":
            value["runtime_config"]["context_cache"] = {}
        else:
            value["dependencies"]["swegemma"] = "changed"
        write_json(path, value)
        cell["sha256"] = file_hash(path)
    elif damage == "incomplete":
        report["actual_order"] = report["actual_order"][:1]
    elif damage == "controls":
        next((gpu.parent / "controls").rglob("origins.json")).write_text("{}")
    elif damage == "variant":
        variant = "foreign"
    write_json(gpu, report)
    source = {
        "owner": "owner",
        "kernel": "kernel",
        "version": 1,
        "path": "results/run/archives/sequential-v1/submission.zip",
    }
    if damage != "none":
        with pytest.raises((ValueError, OSError)):
            qualify_experiment(cpu, gpu, variant, source, 10000000)
    else:
        candidate = qualify_experiment(cpu, gpu, variant, source, 10000000)
        assert candidate["evidence"]["kind"] == "paired-experiment"
        assert candidate["archive_sha256"] == file_hash(root / variant / "submission.zip")


@pytest.mark.parametrize("phase", ["model_startup", "evaluation"])
def test_complete_stage_deadline_stops_the_session_and_preserves_missing_score(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import asyncio
    import time

    from gemma_agent_dev.common.artifacts import content_hash

    root, provenance, calls = session_fixture
    runtime = json.loads((root / "runtime.json").read_text())
    spec = json.loads((root / "experiment.json").read_text())
    if phase == "evaluation":
        spec["definition"]["session"]["evaluation_timeout_seconds"] = 1

        async def slow(*args: Any) -> Any:
            await asyncio.sleep(2)

        monkeypatch.setattr(session, "evaluate", slow)
    else:
        runtime["vllm"]["startup_timeout"] = 1
        write_json(root / "runtime.json", runtime)
        spec["definition"]["runtime"] = runtime
        original = session.create_server

        def create(*args: Any) -> Any:
            server = original(*args)
            server.start = lambda: time.sleep(2)
            return server

        monkeypatch.setattr(session, "create_server", create)
    spec["definition_sha256"] = content_hash(spec["definition"])
    write_json(root / "experiment.json", spec)
    with pytest.raises(TimeoutError):
        session.run(root, provenance)
    path = next(Path(runtime["paths"]["output"]).glob("*/session.json"))
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and report["stop_reason"] == phase + "_deadline_exceeded"
    assert calls[-1] == "stop"
    if phase == "evaluation":
        result = summarize(path, max_bytes=2000000)
        assert result["counts"]["incomplete"] == 1 and result["counts"]["neither"] == 0
        assert all(item["resolved"] is None for item in result["observations"])
        assert len(report["actual_order"]) == 1
