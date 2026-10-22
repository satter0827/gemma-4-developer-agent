import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.cli import main
from gemma_agent_dev.common.artifacts import content_hash, file_hash, write_json
from gemma_agent_dev.evaluation import sdk_checks, sequential_checks, session
from gemma_agent_dev.notebooks import experiments as bundles
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.workflows.records import Flow, exclusive, inspect


def invoke(arguments: list[str], capsys: Any) -> tuple[int, dict[str, Any]]:
    code = main(["workflow", *arguments, "--json"])
    return code, json.loads(capsys.readouterr().out)


def sdk_boundary(root: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    (root / "sdk.yaml").write_text("synthetic SDK input")
    environment = root / "artifacts/sdk"
    environment.mkdir(parents=True)
    data = {"environment": environment}
    state = {"python": "3.12", "packages": {"swegemma": "0.2.11"}}
    write_json(environment / "prepared.json", {"inputs": {"sdk": "fixed"}, "installed": state})
    monkeypatch.setattr(sdk_checks, "settings", lambda *a: data)
    monkeypatch.setattr(sdk_checks, "fingerprint", lambda *a: {"sdk": "fixed"})
    monkeypatch.setattr(sdk_checks, "installed_state", lambda *a: state)
    monkeypatch.setattr(bundles, "sdk_settings", lambda *a: data)
    monkeypatch.setattr(bundles, "fingerprint", lambda *a: {"sdk": "fixed"})
    monkeypatch.setattr(bundles, "installed_state", lambda *a: state)
    monkeypatch.setattr(sequential_checks, "run", lambda *a: {"status": "completed"})

    def worker(arguments: list[str], **kwargs: Any) -> str:
        assert arguments[2] == "gemma_agent_dev.evaluation.session_worker"
        manifest = session.run(
            Path(arguments[3]),
            json.loads(arguments[4]),
            verify_only=True,
            bindings=json.loads(arguments[5]),
        )
        return json.dumps({"kind": "experiment-verification-receipt", "manifest": str(manifest)})

    monkeypatch.setattr(bundles, "command", worker)
    return worker


def test_prepare_public_cli_to_cpu_evidence_and_resume(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    extracted, _, calls = session_fixture
    root = extracted.parent
    sdk_boundary(root, monkeypatch)
    monkeypatch.chdir(root)
    arguments = ["prepare", "--config", "experiment.yaml", "--sdk-config", "sdk.yaml"]
    code, report = invoke(arguments, capsys)
    assert code == 0 and report["stage"] == "cpu_verified"
    assert report["completion_budget"]["pair_count"] == 1
    assert report["completion"]["quality"] == "completed"
    assert report["completion"]["gpu_evaluation"] == "not_run"
    cpu = json.loads(Path(report["artifacts"]["cpu_verification"]).read_text())
    assert cpu["kind"] == "experiment-verification" and cpu["cells"] == []
    assert "control_evidence" in cpu and calls.count("start") == 0
    receipt = report["artifacts"]["workflow"]
    count = len(calls)
    code, resumed = invoke([*arguments, "--resume", receipt], capsys)
    assert code == 0 and resumed["workflow_id"] == report["workflow_id"]
    assert len(calls) == count
    code, status = invoke(["status", "--run", receipt], capsys)
    assert code == 0 and status["stage"] == "cpu_verified"
    Path(report["artifacts"]["bundle"]).write_text("changed")
    code, status = invoke(["status", "--run", receipt], capsys)
    assert code == 1 and status["reason_codes"] == ["ValueError"]


@pytest.mark.parametrize("error_type", [OSError, AttributeError])
def test_public_prepare_restarts_interrupted_stage_without_reusing_orphan_evidence(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, error_type: type[Exception]
) -> None:
    extracted, _, calls = session_fixture
    root = extracted.parent
    worker = sdk_boundary(root, monkeypatch)
    manifests: list[Path] = []

    def interrupted(*args: Any, **kwargs: Any) -> str:
        result = str(worker(*args, **kwargs))
        manifests.append(Path(json.loads(result)["manifest"]))
        if len(manifests) == 1:
            raise error_type("Worker receipt interrupted")
        return result

    monkeypatch.setattr(bundles, "command", interrupted)
    monkeypatch.chdir(root)
    arguments = ["prepare", "--config", "experiment.yaml", "--sdk-config", "sdk.yaml"]
    code, failed = invoke(arguments, capsys)
    assert code == 1 and failed["stage"] == "cpu_verification"
    assert failed["reason_codes"] == [error_type.__name__]
    assert "freeze" in failed["artifacts"] and "Worker receipt interrupted" in failed["error"]
    digest = file_hash(manifests[0])
    code, resumed = invoke([*arguments, "--resume", failed["artifacts"]["workflow"]], capsys)
    assert code == 0 and resumed["workflow_id"] == failed["workflow_id"]
    assert len(manifests) == 2 and manifests[0] != manifests[1]
    assert file_hash(manifests[0]) == digest and calls.count("start") == 0


def test_cpu_worker_emits_structured_receipt_without_progress_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from gemma_agent_dev.evaluation import session_worker

    def run(*args: Any, **kwargs: Any) -> Path:
        print("CPU progress")
        write_json(tmp_path / "session.json", {"status": "completed"})
        return tmp_path / "session.json"

    monkeypatch.setenv("GEMMA_PROCESS_REGISTRY", str(tmp_path / "processes.jsonl"))
    monkeypatch.setenv("GEMMA_PROCESS_OWNER", "fixture")
    monkeypatch.setattr(session_worker, "run", run)
    monkeypatch.setattr(sys, "addaudithook", lambda hook: None)
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), "{}", "{}"])
    session_worker.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["manifest"] == str(tmp_path / "session.json")
    assert captured.err == "CPU progress\n"


def test_diagnose_public_cli_and_failed_resume(
    diagnostic_project: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    root = diagnostic_project.root
    monkeypatch.chdir(root)
    code, report = invoke(["diagnose", "--config", "diagnostics.yaml"], capsys)
    assert code == 0 and report["stage"] == "patch_diagnosed"
    receipt = report["artifacts"]["workflow"]
    code, result = invoke(["diagnose", "--config", "diagnostics.yaml", "--resume", receipt], capsys)
    assert code == 0 and result["workflow_id"] == report["workflow_id"]
    (root / "generated.patch").write_text("changed")
    code, result = invoke(["diagnose", "--config", "diagnostics.yaml", "--resume", receipt], capsys)
    assert code == 1 and result["status"] == "failed"


def test_report_and_candidate_use_real_evidence_services(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    extracted, provenance, _ = session_fixture
    root = extracted.parent
    monkeypatch.setattr(sequential_checks, "run", lambda *a: {"status": "completed"})
    cpu = session.run(extracted, provenance, verify_only=True)
    gpu = session.run(extracted, provenance)
    monkeypatch.chdir(root)
    code, report = invoke(["report", "--input", str(gpu)], capsys)
    assert code == 0 and report["stage"] == "results_analyzed"
    summary = json.loads(Path(report["artifacts"]["analysis"]).read_text())
    assert summary["counts"]["both"] == 1
    source = root / "source.json"
    write_json(
        source,
        {
            "owner": "owner",
            "kernel": "kernel",
            "version": 17,
            "path": "results/run/archives/workflow-v1/submission.zip",
        },
    )
    saved = json.loads(gpu.read_text())
    archive = gpu.parent / saved["candidate_archives"]["workflow-v1"]["path"]
    sdk_python = root / "sdk-api/bin/python"
    sdk_python.parent.mkdir(parents=True)
    sdk_python.symlink_to(sys.executable)

    def download(self: Any, operation: str, **kwargs: Any) -> dict[str, Any]:
        assert operation == "kernel" and kwargs["source"]["version"] == 17
        assert self.python == sdk_python
        shutil.copyfile(archive, kwargs["destination"])
        return {"bytes": archive.stat().st_size}

    monkeypatch.setattr(KaggleGateway, "call", download)
    code, result = invoke(
        [
            "candidate",
            "--cpu",
            str(cpu),
            "--gpu",
            str(gpu),
            "--variant",
            "workflow-v1",
            "--source",
            str(source),
            "--policy",
            "configs/submission/policy.yaml",
            "--api-python",
            str(sdk_python),
        ],
        capsys,
    )
    assert code == 0 and result["stage"] == "candidate_prepared"
    candidate = json.loads(Path(result["artifacts"]["candidate_evidence"]).read_text())
    assert candidate["archive_sha256"] == file_hash(archive)
    assert not (root / "artifacts/submission-state").exists()


def test_record_failures_resume_lock_and_input_identity(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("fixed")
    files = {"input.txt": file_hash(input_path)}
    with exclusive(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with exclusive(tmp_path):
                pass
    with pytest.raises(RuntimeError):
        with Flow(tmp_path, "test", {"x": 1}, files, None) as flow:
            receipt = flow.path
            flow.step("failed", lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert inspect(tmp_path, receipt)["status"] == "failed"
    output = tmp_path / "output/report.json"
    output.parent.mkdir()
    write_json(output, {"completed": True})
    with Flow(tmp_path, "test", {"x": 1}, files, receipt) as resumed:
        assert resumed.step("failed", lambda: output) == output
        resumed.finish("done", [])
    with pytest.raises(ValueError, match="inputs"):
        Flow(tmp_path, "test", {"x": 2}, files, receipt)
    input_path.write_text("changed")
    with pytest.raises(ValueError, match="evidence"):
        inspect(tmp_path, receipt)


def test_invalid_workflow_arguments_have_json_result(capsys: Any) -> None:
    code, report = invoke(["prepare"], capsys)
    assert code == 2 and report["stage"] == "arguments"
    assert report["reason_codes"] == ["invalid_arguments"]


def test_workflow_save_failure_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.workflows import records

    flow = Flow(tmp_path, "test", {}, {}, None)
    original = write_json

    def save(path: Path, data: Any) -> None:
        if data["status"] == "failed":
            raise OSError("disk full")
        original(path, data)

    monkeypatch.setattr(records, "write_json", save)
    with pytest.raises(RuntimeError, match="original") as error:
        with flow:
            flow.step("failing", lambda: (_ for _ in ()).throw(RuntimeError("original")))
    assert any("disk full" in note for note in error.value.__notes__)


@pytest.mark.parametrize(
    "arguments",
    [
        ["status", "--run", "missing.json"],
        ["diagnose", "--config", "../outside.yaml"],
        ["report", "--input", "missing.json", "--remote-version", "1"],
        ["report", "--remote-kernel", "owner/kernel"],
    ],
)
def test_workflow_errors_preserve_machine_readable_contract(
    diagnostic_project: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any, arguments: list[str]
) -> None:
    monkeypatch.chdir(diagnostic_project.root)
    code, report = invoke(arguments, capsys)
    assert code == 1 and report["status"] == "failed"
    assert {
        "workflow_id",
        "status",
        "stage",
        "artifacts",
        "reason_codes",
        "next_actions",
    } <= report.keys()


def test_complete_budget_is_not_replaced_by_quota() -> None:
    from gemma_agent_dev.evaluation.config import EvaluationSettings, RuntimeSettings
    from gemma_agent_dev.evaluation.experiments import Experiment

    root = Path(__file__).resolve().parents[2]
    experiment = Experiment.load(root, Path("configs/evaluation/easy-skills-v3.yaml"))
    runtime = RuntimeSettings.load(experiment.runtime_config).as_dict()
    evaluation = EvaluationSettings.load(experiment.evaluation_config).as_dict()
    result = experiment.session.completion(evaluation, runtime, len(experiment.task_ids))
    assert result["known_estimated_seconds"] == 9420 and result["total_upper_seconds"] is None
    assert not result["allowed"]
    with pytest.raises(ValueError, match="Complete experiment"):
        experiment.session.require_completion(evaluation, runtime, len(experiment.task_ids))
    from dataclasses import replace

    policy = replace(
        experiment.session, preparation_timeout_seconds=1200, evaluation_timeout_seconds=660
    )
    assert policy.completion(evaluation, runtime, 6)["missing_seconds"] == 5220
    exact = replace(policy, job_timeout_seconds=10620)
    exact.require_completion(evaluation, runtime, 6)


def test_shared_controls_bind_conditions_and_cell_references(session_fixture: Any) -> None:
    extracted, provenance, calls = session_fixture
    path = session.run(extracted, provenance)
    report = json.loads(path.read_text())
    assert calls.count("controls") == 1
    report["control_evidence"]["condition_sha256"] = content_hash("changed")
    write_json(path, report)
    from gemma_agent_dev.evaluation.experiment_results import summarize

    with pytest.raises(ValueError, match="conditions"):
        summarize(path, max_bytes=2000000)


def test_preparation_expiration_saves_failure_and_stops_model(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    extracted, provenance, calls = session_fixture
    spec = json.loads((extracted / "experiment.json").read_text())
    spec["definition"]["session"]["preparation_timeout_seconds"] = 1
    write_json(extracted / "experiment.json", spec)

    async def slow(*args: Any) -> bool:
        await asyncio.sleep(2)
        return True

    monkeypatch.setattr(session, "verify_controls", slow)
    with pytest.raises(TimeoutError, match="Execution"):
        session.run(extracted, provenance)
    assert calls.count("start") == 0 and calls[-1] == "stop"
    report = next((extracted.parent / "artifacts/results").glob("*/session.json"))
    assert json.loads(report.read_text())["status"] == "failed"


def test_model_conditions_preserve_unknown_memory_and_measure_disks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sysconf(name: str) -> int:
        return 100

    monkeypatch.setattr(os, "sysconf", sysconf)
    runtime = {"model_source": "fixed", "paths": {"model": str(tmp_path)}}
    measured = session.model_conditions(runtime, tmp_path)
    assert measured["available_memory_bytes"] == 10000
    assert measured["model_disk"]["free"] > 0


def test_preparation_deadline_includes_notebook_bootstrap(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    extracted, provenance, calls = session_fixture
    monkeypatch.setenv("GEMMA_JOB_STARTED_MONOTONIC", str(time.monotonic() - 61))
    with pytest.raises(TimeoutError, match="Execution"):
        session.run(extracted, provenance)
    assert not calls
    path = next((extracted.parent / "artifacts/results").glob("*/session.json"))
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and report["timings"]["bootstrap_seconds"] >= 61


def test_new_candidates_change_only_instructions_and_keep_fixed_six_tasks() -> None:
    from gemma_agent_dev.candidates.changes import agent_fields
    from gemma_agent_dev.evaluation.experiments import Experiment

    root = Path(__file__).resolve().parents[2]
    parent = Experiment.load(root, Path("configs/evaluation/easy-skills-v3.yaml"))
    for name, baseline, candidate in (
        ("easy-completion-v1", "workflow-v1", "workflow-completion-v1"),
        ("easy-skill-use-v1", "workflow-skills-v1", "workflow-skill-use-v1"),
    ):
        experiment = Experiment.load(root, Path(f"configs/evaluation/{name}.yaml"))
        assert experiment.task_ids == parent.task_ids
        left = agent_fields(root / f"agent/{baseline}.yaml")
        right = agent_fields(root / f"agent/{candidate}.yaml")
        assert left.pop("instruction") != right.pop("instruction")
        assert left == right
        assert experiment.session.preparation_timeout_seconds == 1200


def test_prepare_reuses_explicit_quality_through_public_cli(
    session_fixture: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from gemma_agent_dev.candidates import quality

    extracted, _, _ = session_fixture
    root = extracted.parent
    sdk_boundary(root, monkeypatch)

    @contextmanager
    def process(*args: Any, **kwargs: Any) -> Any:
        kwargs["stdout"].write("synthetic quality boundary")
        yield SimpleNamespace(wait=lambda **kw: 0)

    monkeypatch.setattr(quality, "process_group", process)
    proof = quality.run(root, root / "artifacts/quality-proof", 60)
    monkeypatch.chdir(root)
    code, report = invoke(
        [
            "prepare",
            "--config",
            "experiment.yaml",
            "--sdk-config",
            "sdk.yaml",
            "--quality-report",
            str(proof),
        ],
        capsys,
    )
    assert code == 0 and report["stage"] == "cpu_verified"
    preparation = json.loads(Path(report["artifacts"]["freeze"]).read_text())
    assert preparation["quality"]["sha256"] == file_hash(proof)
    (root / "agent/prompts/workflow.md").write_text("changed")
    code, report = invoke(
        [
            "prepare",
            "--config",
            "experiment.yaml",
            "--sdk-config",
            "sdk.yaml",
            "--quality-report",
            str(proof),
        ],
        capsys,
    )
    assert code == 1 and "inputs changed" in report["error"]
