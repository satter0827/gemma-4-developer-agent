import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.evaluation import experiment_preparation as preparation


def test_freeze_two_selected_closures_and_validate_provenance(
    experiment_project: Path, frozen_experiment: Path
) -> None:
    experiment, report = preparation.load_preparation(experiment_project, frozen_experiment)
    assert report["status"] == "completed" and list(report["variants"]) == list(experiment.variants)
    assert "prompts/workflow.md" in report["variants"]["workflow-v1"]["files"]
    assert "prompts/system.md" not in report["variants"]["workflow-v1"]["files"]
    assert "prompts/verify-v1.md" in report["variants"]["sequential-v1"]["files"]
    assert "sub_agents/code_analyzer.yaml" not in report["variants"]["sequential-v1"]["files"]
    assert (
        report["variants"]["sequential-v1"]["sha256"] != report["variants"]["workflow-v1"]["sha256"]
    )


@pytest.mark.parametrize(
    "damage", ["input", "candidate", "controls", "case", "status", "definition"]
)
def test_freeze_rejects_changed_evidence(
    experiment_project: Path, frozen_experiment: Path, damage: str
) -> None:
    root = experiment_project
    d = json.loads(frozen_experiment.read_text())
    if damage == "input":
        (root / "agent/prompts/verify-v1.md").write_text("changed")
    elif damage == "candidate":
        (frozen_experiment.parent / "sequential-v1/submission.zip").write_bytes(b"changed")
    elif damage == "controls":
        (root / d["controls"]["path"]).write_text("{}")
    elif damage == "case":
        (root / d["controls"]["path"]).parent.joinpath("fixture_1/reference/case.json").write_text(
            "{}"
        )
    else:
        d["status" if damage == "status" else "definition_sha256"] = "changed"
        write_json(frozen_experiment, d)
    with pytest.raises((ValueError, KeyError)):
        preparation.load_preparation(root, frozen_experiment)


def test_freeze_failure_is_persisted(
    experiment_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("quality failed")

    monkeypatch.setattr(preparation, "verify_quality", fail)
    with pytest.raises(RuntimeError, match="quality failed"):
        preparation.prepare(experiment_project, Path("experiment.yaml"))
    report = next((experiment_project / "artifacts/experiments").glob("*/preparation.json"))
    assert json.loads(report.read_text())["status"] == "failed"


def test_order_is_checked_before_creating_candidates(experiment_project: Path) -> None:
    p = experiment_project / "pool.yaml"
    d = yaml.safe_load(p.read_text())
    d["tuning"].append("fixture_2")
    d["uses"][0]["task_ids"].append("fixture_2")
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="exact frozen"):
        preparation.prepare(experiment_project, Path("experiment.yaml"))
    assert not (experiment_project / "artifacts/experiments").exists()
