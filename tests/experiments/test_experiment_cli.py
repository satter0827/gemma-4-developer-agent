import json
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.cli import main


@pytest.mark.parametrize(
    "command", ["validate-experiment", "prepare-experiment-tasks", "freeze-experiment"]
)
def test_experiment_cli_explicit_actions(
    experiment_project: Path, monkeypatch: pytest.MonkeyPatch, command: str, capsys: Any
) -> None:
    monkeypatch.chdir(experiment_project)
    monkeypatch.setattr(
        "gemma_agent_dev.diagnostics.local.prepare", lambda settings: Path("prepared.json")
    )
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.experiment_preparation.prepare",
        lambda *a: Path("preparation.json"),
    )
    assert main([command, "--config", "experiment.yaml"]) == 0
    text = capsys.readouterr().out
    if command == "validate-experiment":
        assert json.loads(text)["planned_order"][0]["variant"] == "workflow-v1"
    else:
        assert ".json" in text


def test_tracking_and_cpu_commands(
    experiment_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(experiment_project)
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.tracking.prepare", lambda *a: Path("prepared.json")
    )
    monkeypatch.setattr("gemma_agent_dev.evaluation.tracking.sync", lambda *a: Path("synced"))
    monkeypatch.setattr(
        "gemma_agent_dev.notebooks.experiments.verify_local", lambda *a: Path("cpu.json")
    )
    monkeypatch.setattr(
        "gemma_agent_dev.evaluation.experiment_results.summarize",
        lambda *a: {"counts": {"both": 1}},
    )
    assert main(["prepare-tracking", "--config", "tracking.yaml"]) == 0
    assert main(["sync-tracking", "--config", "tracking.yaml", "--record", "record.json"]) == 0
    assert main(["verify-experiment", "--bundle", "bundle", "--sdk-config", "sdk.yaml"]) == 0
    path = experiment_project / "session.json"
    path.touch()
    assert main(["summarize-experiment", "--session", str(path)]) == 0
    assert json.loads((path.parent / "paired-summary.json").read_text())["counts"]["both"] == 1


def test_notebook_experiment_route(
    experiment_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(experiment_project)
    calls = []

    def prepare(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "digest"

    monkeypatch.setattr("gemma_agent_dev.notebooks.experiments.prepare", prepare)
    assert (
        main(
            [
                "prepare-kaggle",
                "--name",
                "experiment",
                "--experiment-report",
                "report.json",
                "--verify-only",
            ]
        )
        == 0
    )
    assert calls[0]["verify_only"] and calls[0]["quota"] is None
    for extra in [
        ["--diagnostics-config", "diagnostics.yaml"],
        ["--preflight-report", "report.json"],
    ]:
        with pytest.raises(SystemExit):
            main(
                [
                    "prepare-kaggle",
                    "--name",
                    "experiment",
                    "--experiment-report",
                    "report.json",
                    *extra,
                ]
            )
    with pytest.raises(SystemExit):
        main(["prepare-kaggle", "--name", "experiment", "--quota", "quota.json"])
