import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gemma_agent_dev.candidates import quality


@pytest.mark.parametrize("damage", [None, "source", "log", "exit", "missing", "order"])
def test_quality_reuse_checks_complete_commands_inputs_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    for name in ["Makefile", "pyproject.toml", "uv.lock"]:
        (tmp_path / name).write_text("source")
    calls = []

    @contextmanager
    def process(args: list[str], **kwargs: Any) -> Iterator[Any]:
        calls.append(args)
        kwargs["stdout"].write("checks passed")
        yield SimpleNamespace(wait=lambda **kw: 0)

    monkeypatch.setattr(quality, "process_group", process)
    path = quality.run(tmp_path, tmp_path / "artifacts/quality", 30)
    assert calls == [["make", c] for c in quality.COMMANDS]
    report = json.loads(path.read_text())
    if damage == "source":
        (tmp_path / "Makefile").write_text("changed")
    elif damage == "log":
        (path.parent / report["steps"][0]["log"]).write_text("changed")
    elif damage == "exit":
        report["steps"][0]["exit_code"] = 1
    elif damage == "missing":
        report["status"] = "running"
    elif damage == "order":
        report["steps"].reverse()
    if damage in {"exit", "missing", "order"}:
        path.write_text(json.dumps(report))
    if damage:
        with pytest.raises(ValueError):
            quality.reuse(tmp_path, path)
    else:
        assert quality.reuse(tmp_path, path)["path"] == str(path)


def test_failed_quality_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["Makefile", "pyproject.toml", "uv.lock"]:
        (tmp_path / name).write_text("source")

    @contextmanager
    def process(*args: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(wait=lambda **kw: 1)

    monkeypatch.setattr(quality, "process_group", process)
    with pytest.raises(subprocess.CalledProcessError):
        quality.run(tmp_path, tmp_path / "artifacts/quality", 30)
    assert (
        json.loads((tmp_path / "artifacts/quality/quality.json").read_text())["status"] == "failed"
    )
