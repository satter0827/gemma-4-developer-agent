import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common import events

ROOT = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for p in path.glob("*.jsonl*") for line in p.read_text().splitlines()]


def test_json_events_context_levels_and_redaction(tmp_path: Path, capsys: Any) -> None:
    settings = events.LogSettings.load(ROOT)
    with events.session(tmp_path, settings, "execution-1"):
        events.event("test.debug", level="DEBUG")
        with events.context(task_id="task_1", variant="baseline"):
            with events.phase("test", count=2):
                events.event("test.outcome", resolved=False)
        events.event("test.after")
        try:
            raise ValueError("password=secret task body and Authorization Bearer secret")
        except ValueError:
            events.event("test.exception", level="ERROR", exception=True)
    data = rows(tmp_path)
    assert len(data) == 5
    assert data[0]["task_id"] == "task_1" and data[0]["count"] == 2
    assert "task_id" not in data[-2]
    assert all(r["timestamp"].endswith("Z") and r["execution_id"] == "execution-1" for r in data)
    assert data[1]["resolved"] is False and data[2]["duration_seconds"] >= 0
    assert data[-1]["exception"]["type"] == "ValueError"
    assert "secret" not in json.dumps(data)
    captured = capsys.readouterr()
    assert not captured.out and "test.exception" in captured.err and "secret" not in captured.err
    assert not events.CONTEXT.get() and not events.ACTIVE.get()


@pytest.mark.parametrize(
    "field,value",
    [
        ("headers", "secret"),
        ("task_id", "text with spaces"),
        ("count", float("nan")),
        ("count", object()),
    ],
)
def test_log_fields_are_allowlisted(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        events.checked({field: value})
    assert events.checked({"count": None, "resolved": False, "exit_code": -1})


@pytest.mark.parametrize("damage", ["keys", "level", "max_bytes", "backup_count"])
def test_log_policy_is_explicit(damage: str) -> None:
    raw = asdict(events.LogSettings.load(ROOT))
    raw[damage if damage != "level" else "file_level"] = 0 if damage != "level" else "TRACE"
    with pytest.raises(ValueError):
        events.LogSettings.parse(raw)


def test_rotation_and_logging_restoration(tmp_path: Path) -> None:
    logger = logging.getLogger("gemma_agent_dev")
    before = logger.level, logger.propagate, logger.handlers[:]
    policy = events.LogSettings("DEBUG", "ERROR", 500, 2)
    with events.session(tmp_path, policy, "rotation"):
        for _ in range(20):
            events.event("rotation.progress", count=1)
    assert len(list(tmp_path.glob("*.jsonl*"))) == 3
    assert rows(tmp_path)
    assert before == (logger.level, logger.propagate, logger.handlers)


@pytest.mark.parametrize("original", [False, True])
def test_disk_failure_preserves_results_and_original_error(
    tmp_path: Path, original: bool, capsys: Any
) -> None:
    finished = []
    expected = RuntimeError("original") if original else None
    from contextlib import nullcontext

    with pytest.raises(RuntimeError) if original else nullcontext():
        with events.session(tmp_path, events.LogSettings.load(ROOT), "disk"):
            handler = logging.getLogger("gemma_agent_dev").handlers[0]
            assert isinstance(handler, events.EventHandler)
            handler.handleError(logging.makeLogRecord({}))
            handler.handleError(logging.makeLogRecord({}))
            with events.phase("work"):
                events.event("work.after_failure")
                finished.append("results_saved")
                if expected:
                    raise expected
    assert finished == ["results_saved"]
    assert events.load_health(tmp_path) == {"status": "degraded", "reason_codes": ["event_write"]}
    assert capsys.readouterr().err.count("Cannot save") == 1


def test_inactive_and_invalid_events_and_bundle_policy(tmp_path: Path) -> None:
    events.event("inactive.event")
    with pytest.raises(ValueError):
        events.event("bad event")
    with events.bundle_session(tmp_path, tmp_path / "out", {}):
        assert not events.ACTIVE.get()
    (tmp_path / "logging.json").write_bytes(events.policy(ROOT))
    with events.bundle_session(tmp_path, tmp_path / "out", {"experiment_id": "trial"}):

        @events.logged("wrapped")
        def run() -> int:
            return 1

        assert run() == 1
    assert rows(tmp_path / "out/logs/events")[0]["experiment_id"] == "trial"


def test_unstructured_messages_never_escape_on_formatter_failure(
    tmp_path: Path, capsys: Any
) -> None:
    with events.session(tmp_path, events.LogSettings.load(ROOT), "bad-log"):
        logging.getLogger("gemma_agent_dev").warning("Authorization Bearer secret-content")
    captured = capsys.readouterr()
    assert not captured.out and "secret-content" not in captured.err
    assert "raw message omitted" in captured.err


def test_log_initialization_failure_is_safe(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "file"
    path.write_text("existing")
    with events.session(path, events.LogSettings.load(ROOT), "init") as health:
        assert "initialization" in health.reason_codes
    assert "Cannot save" in capsys.readouterr().err


def test_child_process_receives_execution_id_and_separate_file(tmp_path: Path) -> None:
    import os
    import sys

    from gemma_agent_dev.common.processes import command

    settings = events.LogSettings.load(ROOT)
    script = """import json,os,sys
from pathlib import Path
from gemma_agent_dev.common.events import LogSettings,session,event
settings=LogSettings.parse(json.loads(sys.argv[2]))
with session(Path(sys.argv[1]),settings,os.environ['GEMMA_EXECUTION_ID']):
    event('child.completed')
"""
    with events.session(tmp_path, settings, "shared-run"):
        assert (
            command(
                [sys.executable, "-c", script, str(tmp_path), json.dumps(asdict(settings))],
                cwd=ROOT,
                env=dict(os.environ),
                timeout=20,
            )
            == ""
        )
    data = rows(tmp_path)
    assert len(list(tmp_path.glob("*.jsonl"))) == 2
    assert {item["execution_id"] for item in data} == {"shared-run"}
    assert len({item["process_id"] for item in data}) == 2


@pytest.mark.parametrize("fault", ["close", "status", "read", "malformed"])
def test_health_storage_and_close_never_change_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, fault: str
) -> None:
    if fault == "close":
        original = events.EventHandler.close

        def close(handler: Any) -> None:
            original(handler)
            raise OSError("secret")

        monkeypatch.setattr(events.EventHandler, "close", close)
    elif fault == "status":

        def fail(*args: Any) -> None:
            raise OSError("secret")

        monkeypatch.setattr(events, "write_json", fail)
    with events.session(tmp_path, events.LogSettings.load(ROOT), "health") as health:
        events.event("health.completed")
    if fault in {"close", "status"}:
        assert health.as_dict()["status"] == "degraded"
        assert capsys.readouterr().err.count("Cannot save") == 1
    if fault == "read":
        (tmp_path / "logging-status.json").unlink()
        (tmp_path / "logging-status.json").mkdir()
    elif fault == "malformed":
        (tmp_path / "logging-status.json").write_text("{}")
    if fault in {"read", "malformed"}:
        assert events.load_health(tmp_path)["status"] == "degraded"
    assert "secret" not in capsys.readouterr().err
    assert events.load_health(tmp_path / "missing")["status"] == "unknown"
