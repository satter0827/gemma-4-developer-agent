import os
import sys
from pathlib import Path

import pytest

from gemma_agent_dev.processes import process_group


def test_process_cleanup_preserves_normal_exit_and_log(tmp_path: Path) -> None:
    with (tmp_path / "process.log").open("w") as log:
        with process_group(
            [sys.executable, "-c", "print('completed'); raise SystemExit(7)"],
            cwd=tmp_path,
            env=dict(os.environ),
            stdout=log,
        ) as process:
            assert process.wait(timeout=5) == 7
    assert (tmp_path / "process.log").read_text().strip() == "completed"


def test_interruption_stops_the_process_before_returning(tmp_path: Path) -> None:
    with (tmp_path / "process.log").open("w") as log:
        with pytest.raises(KeyboardInterrupt):
            with process_group(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                env=dict(os.environ),
                stdout=log,
            ) as process:
                raise KeyboardInterrupt
    assert process.poll() is not None and process.returncode < 0


def test_child_is_reaped_even_if_group_signalling_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(pid: int, signal: int) -> None:
        raise PermissionError("cannot signal group")

    monkeypatch.setattr(os, "killpg", fail)
    with (tmp_path / "process.log").open("w") as log:
        with pytest.raises(PermissionError, match="cannot signal"):
            with process_group(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=dict(os.environ),
                stdout=log,
            ) as process:
                process.wait(timeout=5)
    assert process.returncode == 0


def test_cleanup_error_does_not_replace_the_original_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(pid: int, signal: int) -> None:
        raise PermissionError("cannot signal group")

    monkeypatch.setattr(os, "killpg", fail)
    with (tmp_path / "process.log").open("w") as log:
        with pytest.raises(RuntimeError, match="original failure") as raised:
            with process_group(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=dict(os.environ),
                stdout=log,
            ) as process:
                process.wait(timeout=5)
                raise RuntimeError("original failure")
    assert any("cannot signal group" in note for note in raised.value.__notes__)
