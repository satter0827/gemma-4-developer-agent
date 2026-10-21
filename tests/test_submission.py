"""Exercise submission safety with fake remote calls and a real Git CAS."""

import copy
import json
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.artifacts import file_hash
from gemma_agent_dev.submission import (
    decide,
    description,
    execute_run,
    incident,
    resolve_attempt,
    synchronize,
    top_submissions,
)
from gemma_agent_dev.submission_api import KaggleGateway
from gemma_agent_dev.submission_config import SubmissionSettings, timestamp
from gemma_agent_dev.submission_state import (
    add_candidate,
    archive_files,
    checked_candidate,
    content_id,
    empty_state,
    qualify_candidate,
    validate_state,
)
from gemma_agent_dev.submission_store import StateStore

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 21, 23, 15, tzinfo=UTC)  # Oct 22 08:15 JST


@pytest.fixture
def settings() -> SubmissionSettings:
    return SubmissionSettings.load(ROOT, Path("configs/submission.yaml"))


@pytest.fixture
def candidate(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("agent.yaml", "name: synthetic\nmodel: synthetic\ninstruction: synthetic\n")
    files = archive_files(archive, 100000)
    capsule = checked_candidate(
        {
            "id": content_id(files),
            "archive_sha256": file_hash(archive),
            "files": files,
            "source": {
                "owner": "owner",
                "kernel": "kernel",
                "version": 10,
                "path": "results/run/submission.zip",
            },
            "evidence": {
                "cpu_manifest_sha256": "a" * 64,
                "gpu_manifest_sha256": "b" * 64,
                "source_sha256": "c" * 64,
                "cpu_run_id": "cpu",
                "gpu_run_id": "gpu",
            },
        }
    )
    return capsule, archive


class MemoryStore(StateStore):
    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.loaded: dict[str, Any] | None = None

    def load(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.loaded)

    def save(self, state: dict[str, Any]) -> None:
        self.states.append(copy.deepcopy(state))


class FakeGateway(KaggleGateway):
    def __init__(self, archive: Path):
        self.archive = archive
        self.calls: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.limits = {"remaining": 1, "today": 0}
        self.fail: str | None = None
        self.expired = False

    def call(self, operation: str, *, execute: bool = False, **payload: Any) -> Any:
        self.calls.append(operation)
        if operation in {"upload", "create"}:
            assert execute
        if self.fail == operation:
            raise RuntimeError("synthetic failure")
        if operation == "history":
            return copy.deepcopy(self.history)
        if operation == "limits":
            return self.limits
        if operation in {"kernel", "download"}:
            Path(payload["destination"]).write_bytes(self.archive.read_bytes())
            return {"bytes": self.archive.stat().st_size}
        if operation == "upload":
            return {"token": "sensitive-upload-token"}
        if operation == "create":
            return {"expired": True} if self.expired else {"id": 123}
        raise AssertionError(operation)


def prepared(candidate: dict[str, Any]) -> dict[str, Any]:
    state = empty_state("gemma-4-developer-agent")
    state["candidates"] = [candidate]
    return state


def remote(
    ref: int, date: datetime = NOW, *, score: str | None = "0.01", description: str = "manual"
) -> dict[str, Any]:
    return {
        "id": ref,
        "date": date.isoformat(),
        "score": score,
        "status": "COMPLETE" if score is not None else "PENDING",
        "description": description,
    }


@pytest.mark.parametrize(
    "minute,allowed", [(14, False), (15, True), (54, True), (55, False), (59, False)]
)
def test_clock_window(settings: SubmissionSettings, minute: int, allowed: bool) -> None:
    now = NOW.replace(minute=minute)
    assert settings.send_window(now) is allowed
    assert settings.create_before(now).astimezone(UTC) == NOW.replace(minute=55)
    assert not settings.send_window(datetime(2026, 12, 3, 23, 15, tzinfo=UTC))
    with pytest.raises(ValueError):
        timestamp("2026-10-21T23:15")


def test_dry_run_and_backup(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    state, store, api = prepared(capsule), MemoryStore(), FakeGateway(archive)
    out = execute_run(state, store, api, settings, mode="morning", execute=False, clock=lambda: NOW)
    assert out["reason"] == "ready" and not store.states and not state["attempts"]
    assert set(api.calls) == {"history", "limits"}
    out = execute_run(state, store, api, settings, mode="morning", execute=True, clock=lambda: NOW)
    assert out["reason"] == "accepted"
    assert store.states[0]["attempts"][0]["status"] == "creating"
    assert "sensitive-upload-token" not in json.dumps(store.states)
    assert api.calls.count("create") == 1
    api.calls.clear()
    out = execute_run(
        store.states[-1],
        store,
        api,
        settings,
        mode="backup",
        execute=True,
        clock=lambda: NOW + timedelta(minutes=30),
    )
    assert out["reason"] == "already_accepted" and not api.calls


def test_uncertain_receipt_never_retried(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    store, api = MemoryStore(), FakeGateway(archive)
    api.fail = "create"
    first = execute_run(
        prepared(capsule), store, api, settings, mode="morning", execute=True, clock=lambda: NOW
    )
    assert first["reason"] == "acceptance_unknown" and first["notify"]
    latest = store.states[-1]
    second = execute_run(
        latest,
        store,
        api,
        settings,
        mode="backup",
        execute=True,
        clock=lambda: NOW + timedelta(minutes=30),
    )
    assert not second["notify"] and api.calls.count("create") == 1
    api.history = [remote(123, description=description(latest["attempts"][0]))]
    synchronize(latest, api, settings, NOW)
    assert latest["attempts"][0]["submission_id"] == 123
    assert not latest["submissions"][0]["manual"]


def test_manual_hold_identity_and_utc(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    state, api = prepared(capsule), FakeGateway(archive)
    api.history = [remote(1, NOW - timedelta(hours=23, minutes=30))]
    limits = synchronize(state, api, settings, NOW)
    assert decide(state, settings, limits, NOW)[1] == "manual_hold"
    # At exactly 24h the hold expires, but the identical candidate is already submitted.
    assert decide(state, settings, limits, NOW + timedelta(minutes=30))[1] == "no_candidate"
    state["submissions"][0]["content_id"] = "different"
    assert decide(state, settings, limits, NOW + timedelta(minutes=30))[1] == "ready"
    api.limits = {"remaining": 0, "today": 1}
    assert decide(state, settings, api.limits, NOW)[1] == "daily_limit"
    state["submissions"][0]["date"] = NOW.isoformat()
    assert decide(state, settings, limits, NOW)[1] == "daily_limit"


def test_pending_does_not_block_other_candidate(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, _ = candidate
    state = prepared(capsule)
    state["submissions"] = [
        {**remote(1, NOW - timedelta(days=2), score=None), "content_id": "older", "manual": False}
    ]
    assert decide(state, settings, {"remaining": 1, "today": 0}, NOW)[1] == "ready"
    assert top_submissions(state, 2) == []
    state["submissions"] += [
        remote(2, NOW, score="0.1"),
        remote(3, NOW + timedelta(seconds=1), score="0.1"),
    ]
    assert [r["id"] for r in top_submissions(state, 2)] == [3, 2]


def test_cutoff_after_upload(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    api, store = FakeGateway(archive), MemoryStore()

    def clock() -> datetime:
        return NOW.replace(minute=55) if "upload" in api.calls else NOW

    out = execute_run(
        prepared(capsule), store, api, settings, mode="morning", execute=True, clock=clock
    )
    assert out["reason"] == "outside_window" and "create" not in api.calls
    api, store = FakeGateway(archive), MemoryStore()
    api.expired = True
    out = execute_run(
        prepared(capsule), store, api, settings, mode="morning", execute=True, clock=lambda: NOW
    )
    assert out["reason"] == "outside_window"
    assert store.states[-1]["attempts"][0]["status"] == "not_accepted"


def test_monitor_failures_and_end(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    state, api, store = prepared(capsule), FakeGateway(archive), MemoryStore()
    out = execute_run(state, store, api, settings, mode="monitor", execute=True, clock=lambda: NOW)
    assert out["reason"] == "monitor_only" and "upload" not in api.calls
    api.fail = "history"
    with pytest.raises(RuntimeError):
        execute_run(state, store, api, settings, mode="morning", execute=False, clock=lambda: NOW)
    out = execute_run(state, store, api, settings, mode="morning", execute=True, clock=lambda: NOW)
    assert out["notify"] and out["reason"] == "operation_failed"
    api.fail = None
    out = execute_run(
        state,
        store,
        api,
        settings,
        mode="morning",
        execute=True,
        clock=lambda: timestamp(settings.competition_deadline) + timedelta(minutes=1),
    )
    assert out["reason"] == "after_deadline"
    out = execute_run(
        state,
        store,
        api,
        settings,
        mode="monitor",
        execute=True,
        clock=lambda: timestamp(settings.monitoring_end),
    )
    assert out["stop"] and "create" not in api.calls


def test_changed_archive_and_invalid_history(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    capsule["archive_sha256"] = "0" * 64
    api, store = FakeGateway(archive), MemoryStore()
    out = execute_run(
        prepared(capsule), store, api, settings, mode="morning", execute=True, clock=lambda: NOW
    )
    assert out["notify"] and "upload" not in api.calls
    state = prepared(capsule)
    api.history = [remote(1), remote(1)]
    with pytest.raises(ValueError):
        synchronize(state, api, settings, NOW)
    api.history = [remote(1, score="NaN")]
    with pytest.raises(ValueError):
        synchronize(state, api, settings, NOW)
    api.history = []
    api.limits = {"remaining": -1, "today": 0}
    with pytest.raises(ValueError):
        synchronize(state, api, settings, NOW)
    api.limits = {"remaining": 1, "today": 0}
    state["submissions"] = [{"id": 999}]
    with pytest.raises(ValueError, match="incomplete"):
        synchronize(state, api, settings, NOW)


def test_resolve_and_capsule_validation(
    candidate: tuple[dict[str, Any], Path], tmp_path: Path
) -> None:
    capsule, _ = candidate
    state = prepared(capsule)
    attempt = {
        "candidate": capsule["id"],
        "token": "abcdef",
        "started_at": NOW.isoformat(),
        "status": "creating",
        "submission_id": None,
    }
    state["attempts"] = [attempt]
    with pytest.raises(ValueError):
        resolve_attempt(state, "unknown", None)
    with pytest.raises(ValueError):
        resolve_attempt(state, "abcdef", 123)
    state["submissions"] = [
        {
            "id": 123,
            "content_id": capsule["id"],
            "date": NOW.isoformat(),
            "score": "0.01",
            "status": "COMPLETE",
            "manual": True,
        }
    ]
    resolve_attempt(state, "abcdef", 123)
    assert attempt["status"] == "accepted"
    attempt["status"] = "creating"
    resolve_attempt(state, "abcdef", None)
    assert attempt["status"] == "not_accepted"
    assert validate_state(state, state["competition"]) is state
    assert incident(state, "a") and not incident(state, "a") and not incident(state, None)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError):
        add_candidate(state, path)
    edits: list[dict[str, Any]] = [
        {"id": "changed"},
        {"source": {}},
        {"files": {}},
        {"evidence": {}},
    ]
    for edit in edits:
        with pytest.raises(ValueError):
            checked_candidate({**capsule, **edit})
    with pytest.raises(ValueError):
        validate_state({**state, "schema_version": 999}, state["competition"])


def test_state_cas(tmp_path: Path) -> None:
    remote_path, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(remote_path)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(work), "remote", "add", "origin", str(remote_path)], check=True
    )
    first, second = [StateStore(work, "codex/state", "competition", 10) for _ in range(2)]
    assert first.load() is None
    state = empty_state("competition")
    first.save(state)
    assert first.load() == second.load() == state
    first.save(state)
    with pytest.raises(RuntimeError):
        second.save({**state, "incident": "concurrent"})


def test_archived_evidence_is_independent_of_checkout(
    tmp_path: Path, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate
    common: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "task_ids": ["task"],
        "source_sha256": "a" * 64,
        "candidate_sha256": file_hash(archive),
        "submission_files": capsule["files"],
        "tasks_sha256": "b" * 64,
        "snapshots": {},
        "wheelhouse_hashes": {},
        "source_setup": {},
        "evaluation": {},
        "runtime_config": {},
        "vllm_config": {},
        "dependencies": {},
        "verification_timeout_seconds": 30,
        "controls_passed": True,
    }
    paths = []
    for kind in ("verification", "evaluation"):
        directory = tmp_path / kind
        directory.mkdir()
        (directory / "submission.zip").write_bytes(archive.read_bytes())
        controls = [
            {
                "instance_id": "task",
                "unmodified": {"resolved": False, "exit_code": 1, "duration_seconds": 1},
                "reference": {"resolved": True, "exit_code": 0, "duration_seconds": 1},
            }
        ]
        (directory / "controls.json").write_text(json.dumps(controls))
        manifest = {
            **common,
            "kind": kind,
            "run_id": kind,
            "control_artifacts": {"controls.json": file_hash(directory / "controls.json")},
        }
        (directory / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "instance_id": "task",
                    "resolved": False,
                    "duration_seconds": 1,
                    "agent_patch_size": 1,
                    "test_exit_code": 1,
                }
            )
        )
        (directory / "summary.json").write_text('{"total_tasks":1,"resolved":0}')
        manifest["evaluation_artifacts"] = {
            name: file_hash(directory / name) for name in ("task_results.jsonl", "summary.json")
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    qualified = qualify_candidate(paths[0], paths[1], capsule["source"], 100000)
    assert qualified["id"] == capsule["id"]
    original = paths[1].read_text()
    edits: list[dict[str, Any]] = [
        {"status": "failed"},
        {"controls_passed": False},
        {"source_sha256": "c" * 64},
        {"control_artifacts": {}},
        {"evaluation_artifacts": {}},
        {"submission_files": {"agent.yaml": "d" * 64}},
    ]
    for edit in edits:
        paths[1].write_text(json.dumps({**json.loads(original), **edit}))
        with pytest.raises(ValueError):
            qualify_candidate(paths[0], paths[1], capsule["source"], 100000)
    paths[1].write_text(original)
    (paths[0].parent / "controls.json").write_text("[]")
    with pytest.raises(ValueError):
        qualify_candidate(paths[0], paths[1], capsule["source"], 100000)


def test_cli_lifecycle_is_readonly_by_default(
    candidate: tuple[dict[str, Any], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import yaml

    from gemma_agent_dev import cli, submission_cli

    capsule, archive = candidate
    config = yaml.safe_load((ROOT / "configs/submission.yaml").read_text())
    config["seed_candidates"] = ["seed.json"]
    (tmp_path / "submission.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "seed.json").write_text(json.dumps(capsule))
    store, api = MemoryStore(), FakeGateway(archive)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(submission_cli, "StateStore", lambda *a: store)
    monkeypatch.setattr(submission_cli, "KaggleGateway", lambda *a: api)
    prefix = ["submission", "--config", "submission.yaml"]
    assert cli.main([*prefix, "init"]) == 0 and not store.states
    assert cli.main([*prefix, "plan"]) == 1  # state must be initialized explicitly
    assert cli.main([*prefix, "init", "--execute"]) == 0
    store.loaded = store.states[-1]
    assert cli.main([*prefix, "init"]) == 1
    count = len(store.states)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    assert cli.main([*prefix, "plan", "--execute"]) == 0
    assert len(store.states) == count and "upload" not in api.calls
    assert "stop=false" in (tmp_path / "outputs").read_text()
    assert "未送信確認" in (tmp_path / "summary.md").read_text()
    assert cli.main([*prefix, "run", "--mode", "monitor", "--execute"]) == 0
    assert cli.main([*prefix, "sync", "--execute"]) == 0
    store.loaded["attempts"] = [
        {
            "candidate": capsule["id"],
            "token": "abcdef",
            "started_at": NOW.isoformat(),
            "status": "creating",
            "submission_id": None,
        }
    ]
    assert (
        cli.main([*prefix, "resolve", "--token", "abcdef", "--confirm-not-accepted", "--execute"])
        == 0
    )
    assert store.states[-1]["attempts"][0]["status"] == "not_accepted"
    monkeypatch.setattr(submission_cli, "qualify_candidate", lambda *a: capsule)
    registration = [
        *prefix,
        "register",
        "--cpu-manifest",
        "cpu.json",
        "--gpu-manifest",
        "gpu.json",
        "--owner",
        "owner",
        "--kernel",
        "kernel",
        "--version",
        "10",
        "--kernel-path",
        "run/submission.zip",
        "--output",
        "capsule.json",
    ]
    assert cli.main(registration) == 0 and (tmp_path / "capsule.json").is_file()
    store.loaded = empty_state(config["competition"])
    assert cli.main([*registration, "--execute"]) == 0
    store.loaded = None
    assert cli.main([*registration, "--execute"]) == 1
    assert "登録候補" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        {"competition": "invalid name"},
        {"state_branch": "main"},
        {"state_branch": "codex/../bad"},
        {"morning_cutoff": "08:00"},
        {"monitoring_end": "2020-01-01T00:00:00Z"},
        {"seed_candidates": "not-a-list"},
        {"manual_hold_hours": 0},
    ],
)
def test_invalid_policy(tmp_path: Path, change: dict[str, Any]) -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/submission.yaml").read_text())
    config["seed_candidates"] = []
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({**config, **change}))
    with pytest.raises(ValueError):
        SubmissionSettings.load(tmp_path, path)


def test_recheck_limits_and_durable_intent_cutoff(
    settings: SubmissionSettings, candidate: tuple[dict[str, Any], Path]
) -> None:
    capsule, archive = candidate

    class QuotaChanged(FakeGateway):
        def call(self, operation: str, *, execute: bool = False, **payload: Any) -> Any:
            if operation == "limits" and "upload" in self.calls:
                self.limits = {"remaining": 0, "today": 1}
            return super().call(operation, execute=execute, **payload)

    api: FakeGateway = QuotaChanged(archive)
    store = MemoryStore()
    out = execute_run(
        prepared(capsule), store, api, settings, mode="morning", execute=True, clock=lambda: NOW
    )
    assert out["reason"] == "daily_limit" and "create" not in api.calls
    api, store = FakeGateway(archive), MemoryStore()
    out = execute_run(
        prepared(capsule),
        store,
        api,
        settings,
        mode="morning",
        execute=True,
        clock=lambda: NOW.replace(minute=55) if store.states else NOW,
    )
    assert out["reason"] == "outside_window" and "create" not in api.calls
    assert store.states[-1]["attempts"][0]["status"] == "not_accepted"
