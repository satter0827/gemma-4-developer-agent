import glob
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev import environment
from gemma_agent_dev.artifacts import file_hash


@pytest.mark.parametrize("fail_install", [False, True])
def test_official_wheel_installation_is_offline_and_cleans_staging_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_install: bool
) -> None:
    wheels, output = tmp_path / "wheels", tmp_path / "output"
    wheels.mkdir()
    output.mkdir()
    names = [
        "engine-1.0cu128-py3-none-any.whl",
        "helper-1.0+cu128-py3-none-any.whl",
        "cutlass-1.0-py3-none-any.whl",
    ]
    for name in names:
        (wheels / name).write_bytes(name.encode())
    pth = tmp_path / "cutlass.pth"
    pth.touch()
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(glob, "glob", lambda pattern: [str(pth)])
    calls: list[list[str]] = []

    def install(args: list[str], **kwargs: Any) -> None:
        calls.append(args)
        assert kwargs["check"] is True
        assert all(option in args for option in ("--no-index", "--no-deps", "--force-reinstall"))
        staged = sorted((output / "wheel-links").glob("*.whl"))
        assert len(staged) == 2 and all(path.is_symlink() for path in staged)
        assert (output / "wheel-links/engine-1.0+cu128-py3-none-any.whl").is_file()
        assert file_hash(staged[0]) == file_hash(wheels / names[0])
        if fail_install:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", install)
    if fail_install:
        with pytest.raises(subprocess.CalledProcessError):
            environment.install_wheelhouse(wheels, output, 123)
    else:
        environment.install_wheelhouse(wheels, output, 123)
    assert len(calls) == 1 and not pth.exists()
    assert not (output / "wheel-links").exists()
    assert os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] == "123"


def test_empty_wheelhouse_fails_before_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "environ", dict(os.environ))
    with pytest.raises(FileNotFoundError, match="No official wheels"):
        environment.install_wheelhouse(tmp_path, tmp_path, 123)
    assert not (tmp_path / "wheel-links").exists()
