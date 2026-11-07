import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gemma_agent_dev.common.ownership import own_subprocesses, stop_owned

ROOT = Path(__file__).resolve().parents[2]
SDK_PYTHON = ROOT / "artifacts/runtime-env/bin/python"


def test_only_owned_groups_stop_and_constructor_is_restored(tmp_path: Path) -> None:
    original = subprocess.Popen
    registry = tmp_path / "processes.jsonl"
    with subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    ) as unrelated:
        try:
            with own_subprocesses(registry, "job"):
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
                )
                assert json.loads(registry.read_text())["pid"] == child.pid
                assert unrelated.poll() is None
            assert child.wait(timeout=5) < 0
            assert unrelated.poll() is None and subprocess.Popen is original
            assert stop_owned(registry, "job") == []
            with pytest.raises(ValueError, match="another job"):
                stop_owned(registry, "other")
        finally:
            os.killpg(unrelated.pid, signal.SIGKILL)
            unrelated.wait()


@pytest.mark.parametrize("identity", [None, "reused"])
def test_stale_identity_is_not_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str | None
) -> None:
    from gemma_agent_dev.common import ownership

    registry = tmp_path / "processes.jsonl"
    registry.write_text(json.dumps(dict(pid=os.getpid(), identity="original", owner="job")) + "\n")
    monkeypatch.setattr(ownership, "_identity", lambda _: identity)
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("unowned group signalled"))
    assert stop_owned(registry, "job") == []
    assert stop_owned(tmp_path / "absent", "job") == []


@pytest.mark.skipif(not SDK_PYTHON.exists(), reason="Official SDK CPU environment is not prepared")
def test_outer_monitor_stops_sdk_detached_command_after_parent_crash(tmp_path: Path) -> None:
    registry = tmp_path / "processes.jsonl"
    ready, leaked = tmp_path / "ready", tmp_path / "leaked"
    command = (
        f"from pathlib import Path;import time;Path({str(ready)!r}).touch();"
        f"time.sleep(2);Path({str(leaked)!r}).touch()"
    )
    script = """import os,sys
from pathlib import Path
from gemma_agent_dev.common.ownership import own_subprocesses
from adk_eval_core.sandbox.subprocess_sandbox import execute_subprocess_command
with own_subprocesses(Path(sys.argv[1]),'job'):
    execute_subprocess_command([sys.executable,'-c',sys.argv[2]],Path(sys.argv[3]),dict(os.environ),timeout=30)
"""
    with subprocess.Popen(
        [str(SDK_PYTHON), "-c", script, str(registry), command, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        start_new_session=True,
    ) as parent:
        try:
            deadline = time.monotonic() + 20
            while not ready.exists() and time.monotonic() < deadline:
                assert parent.poll() is None
                time.sleep(0.02)
            assert ready.exists(), "SDK command did not start"
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
            assert stop_owned(registry, "job")
            time.sleep(2.2)
            assert not leaked.exists(), "SDK command survived external worker termination"
        finally:
            stop_owned(registry, "job")
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait()


def test_cleanup_fault_keeps_original_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="original") as error:
        with own_subprocesses(tmp_path / "processes.jsonl", "job"):
            (tmp_path / "processes.jsonl").write_text("interrupted write")
            raise RuntimeError("original")
    assert any("Incomplete" in note for note in error.value.__notes__)


def test_registration_failure_stops_started_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.common import ownership

    monkeypatch.setattr(ownership, "_identity", lambda _: None)
    original = subprocess.Popen
    with pytest.raises(RuntimeError, match="identity"):
        with own_subprocesses(tmp_path / "processes.jsonl", "job"):
            subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
            )
    assert subprocess.Popen is original


def test_model_launch_has_owned_group_without_detaching_other_commands(tmp_path: Path) -> None:
    from gemma_agent_dev.common.ownership import isolated_launch

    registry = tmp_path / "processes.jsonl"
    with own_subprocesses(registry, "job"):
        with isolated_launch():
            model = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        command = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        try:
            assert os.getpgid(model.pid) == model.pid
            assert os.getpgid(command.pid) == os.getpgrp()
            assert len(registry.read_text().splitlines()) == 1
        finally:
            command.terminate()
            command.wait()
    assert model.returncode is not None and model.returncode < 0
