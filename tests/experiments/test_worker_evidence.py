import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.evaluation.experiment_results import validate_worker_completion
from gemma_agent_dev.notebooks.worker import seal_sessions


def evidence(root: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    job = root / "artifacts/jobs/job-a"
    result = root / "artifacts/results/run/session.json"
    job.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    provenance = {"job_id": job.name, "source_sha256": "a" * 64, "definition_sha256": "b" * 64}
    receipt = {"status": "completed", "provenance": provenance, "stopped_process_groups": []}
    write_json(job / "watchdog.json", receipt)
    (job / "processes.jsonl").write_text("")
    report = {
        "kind": "experiment-session",
        "status": "completed",
        "worker_completion_required": True,
        **provenance,
    }
    write_json(result, report)
    seal_sessions(root / "artifacts", job, provenance, receipt)
    return result, json.loads(result.read_text()), job, receipt


@pytest.mark.parametrize(
    "damage", [None, "missing", "registry", "watchdog", "path", "identity", "cleanup"]
)
def test_completed_result_requires_matching_final_worker_evidence(
    tmp_path: Path, damage: str | None
) -> None:
    path, report, job, receipt = evidence(tmp_path)
    if damage == "missing":
        report.pop("worker_evidence")
    elif damage == "registry":
        (job / "processes.jsonl").write_text("changed")
    elif damage == "watchdog":
        (job / "watchdog.json").unlink()
    elif damage == "path":
        report["worker_evidence"]["watchdog.json"]["path"] = "artifacts/jobs/other/watchdog.json"
    elif damage in {"identity", "cleanup"}:
        if damage == "identity":
            receipt["provenance"]["source_sha256"] = "other"
        else:
            receipt["status"] = "failed"
        write_json(job / "watchdog.json", receipt)
        report["worker_evidence"]["watchdog.json"]["sha256"] = file_hash(job / "watchdog.json")
    if damage:
        with pytest.raises((ValueError, OSError)):
            validate_worker_completion(path, report)
    else:
        validate_worker_completion(path, report)


def test_cleanup_failure_invalidates_current_completion_and_preserves_original_error(
    tmp_path: Path,
) -> None:
    path, report, job, receipt = evidence(tmp_path)
    receipt.update(status="failed", error="worker cleanup failed")
    write_json(job / "watchdog.json", receipt)
    seal_sessions(tmp_path / "artifacts", job, receipt["provenance"], receipt)
    failed = json.loads(path.read_text())
    assert failed["status"] == "failed" and failed["phase"] == "watchdog"
    failed["error"] = "original evaluation failure"
    write_json(path, failed)
    seal_sessions(tmp_path / "artifacts", job, receipt["provenance"], receipt)
    assert json.loads(path.read_text())["error"] == "original evaluation failure"
    validate_worker_completion(path, json.loads(path.read_text()))
    validate_worker_completion(path, {"worker_completion_required": True, "status": "failed"})
    validate_worker_completion(path, {"status": "completed"})


def test_versioned_transfer_includes_worker_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.submission.api import KaggleGateway
    from gemma_agent_dev.workflows import transfer

    remote = tmp_path / "remote"
    path, _, _, _ = evidence(remote)
    payload = {str(p.relative_to(remote)): p.read_bytes() for p in remote.rglob("*") if p.is_file()}
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_bytes(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_bytes()
    )
    policy = repo / "policy.yaml"
    policy.write_bytes(
        (Path(__file__).resolve().parents[2] / "configs/submission/policy.yaml").read_bytes()
    )

    def gateway(self: Any, operation: str, **kwargs: Any) -> Any:
        assert kwargs["source"]["version"] == 17
        if operation == "files":
            return [{"path": name} for name in payload]
        content = payload[kwargs["source"]["path"]]
        Path(kwargs["destination"]).write_bytes(content)
        return {"bytes": len(content)}

    def summarize(path: Path, **kwargs: Any) -> None:
        validate_worker_completion(path, json.loads(path.read_text()))

    monkeypatch.setattr(KaggleGateway, "call", gateway)
    monkeypatch.setattr(transfer, "summarize", summarize)
    imported = transfer.fetch(
        repo, "owner/kernel", 17, str(path.relative_to(remote)), policy, Path(sys.executable)
    )
    validate_worker_completion(imported, json.loads(imported.read_text()))


def test_cpu_worker_does_not_publish_completion_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from gemma_agent_dev.evaluation import session_worker

    manifest = tmp_path / "session.json"

    @contextmanager
    def failing(*args: Any) -> Any:
        yield
        raise RuntimeError("cleanup failed")

    def run(*args: Any, **kwargs: Any) -> Path:
        write_json(manifest, {"status": "completed", **args[1]})
        return manifest

    monkeypatch.setenv("GEMMA_PROCESS_REGISTRY", str(tmp_path / "processes.jsonl"))
    monkeypatch.setenv("GEMMA_PROCESS_OWNER", "job")
    monkeypatch.setattr(session_worker, "own_subprocesses", failing)
    monkeypatch.setattr(session_worker, "run", run)
    monkeypatch.setattr(sys, "addaudithook", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), "{}", "{}"])
    with pytest.raises(RuntimeError, match="cleanup failed"):
        session_worker.main()
    report = json.loads(manifest.read_text())
    assert report["status"] == "failed" and report["cpu_worker_completion_required"]
    assert report.get("cpu_worker_completed") is not True
    assert not capsys.readouterr().out
