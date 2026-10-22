import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.cli import main
from gemma_agent_dev.common.artifacts import inventory
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.workflows.transfer import fetch

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs/submission/policy.yaml"


@pytest.mark.parametrize("damage", [None, "interrupted", "duplicate", "escape", "missing", "size"])
def test_versioned_retrieval_validates_and_atomically_publishes_results(
    tmp_path: Path, make_run: Any, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    policy = root / "policy.yaml"
    policy.write_bytes(POLICY.read_bytes())
    source = make_run()
    payload = {"results/run/" + name: (source / name).read_bytes() for name in inventory(source)}
    calls: list[dict[str, Any]] = []

    def gateway(self: Any, operation: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        assert kwargs["source"]["version"] == 17
        if operation == "files":
            paths = list(payload)
            if damage == "duplicate":
                paths.append(paths[0])
            if damage == "escape":
                paths.append("results/run/../other")
            if damage == "missing":
                paths.remove("results/run/manifest.json")
            return [{"path": path} for path in paths]
        assert operation == "kernel"
        if damage == "interrupted":
            raise OSError("transfer interrupted")
        data = payload[kwargs["source"]["path"]]
        Path(kwargs["destination"]).write_bytes(data)
        return {"bytes": kwargs["max_bytes"] + 1 if damage == "size" else len(data)}

    monkeypatch.setattr(KaggleGateway, "call", gateway)
    arguments = (
        root,
        "owner/kernel",
        17,
        "results/run/manifest.json",
        policy,
        Path(sys.executable),
    )
    if damage:
        with pytest.raises((ValueError, OSError)):
            fetch(*arguments)
        directory = root / "artifacts/imports"
        assert not directory.exists() or not list(directory.iterdir())
    else:
        path = fetch(*arguments)
        receipt = json.loads((path.parents[2] / "import.json").read_text())
        assert receipt["source"]["version"] == 17
        assert path.read_bytes() == (source / "manifest.json").read_bytes()
        assert len(calls) == len(payload) + 1


@pytest.mark.parametrize(
    "kernel,version,path",
    [
        ("owner", 1, "results/run/session.json"),
        ("owner/kernel", 0, "results/run/session.json"),
        ("owner/kernel", 1, "../run/session.json"),
        ("owner/kernel", 1, "session.json"),
        ("owner/kernel", 1, "results/run/other.json"),
    ],
)
def test_retrieval_rejects_unspecified_or_unsafe_remote_identity(
    tmp_path: Path, kernel: str, version: int, path: str
) -> None:
    with pytest.raises(ValueError, match="fixed Notebook"):
        fetch(tmp_path, kernel, version, path, POLICY, Path(sys.executable))


def test_remote_report_uses_fixed_version_and_public_cli(
    diagnostic_project: Any, make_run: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    source = make_run()
    sdk_python = diagnostic_project.root / "sdk-api/bin/python"
    sdk_python.parent.mkdir(parents=True)
    sdk_python.symlink_to(sys.executable)
    payload = {"results/run/" + name: (source / name).read_bytes() for name in inventory(source)}

    def gateway(self: Any, operation: str, **kwargs: Any) -> Any:
        assert kwargs["source"]["version"] == 17
        assert self.python == sdk_python
        if operation == "files":
            return [{"path": name} for name in payload]
        data = payload[kwargs["source"]["path"]]
        Path(kwargs["destination"]).write_bytes(data)
        return {"bytes": len(data)}

    monkeypatch.setattr(KaggleGateway, "call", gateway)
    root = diagnostic_project.root
    (root / "policy.yaml").write_bytes(POLICY.read_bytes())
    monkeypatch.chdir(root)
    assert (
        main(
            [
                "workflow",
                "report",
                "--remote-kernel",
                "owner/kernel",
                "--remote-version",
                "17",
                "--remote-path",
                "results/run/manifest.json",
                "--policy",
                "policy.yaml",
                "--api-python",
                str(sdk_python),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "results_analyzed"
    assert Path(report["artifacts"]["analysis"]).exists()
