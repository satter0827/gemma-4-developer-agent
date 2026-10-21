from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.evaluation.session_policy import SessionPolicy
from gemma_agent_dev.tasks import rotation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[dict[str, Any]]]:
    policy = SessionPolicy.parse(
        yaml.safe_load((ROOT / "configs/evaluation/easy-skills-v1.yaml").read_text())["session"]
    )
    tasks = [f"{repo}_{index}" for repo in ["fastapi", "rich", "requests"] for index in range(1, 5)]
    pool = SimpleNamespace(
        tuning=tasks, uses=[{"experiment_id": "other", "task_ids": ["fastapi_2"]}]
    )
    experiment = SimpleNamespace(
        experiment_id="current",
        pool=tmp_path / "pool.yaml",
        task_ids=[],
        session=policy,
        definition=lambda root: {
            "runtime": {"verification_timeout_seconds": 120, "vllm": {"startup_timeout": 1200}},
            "evaluation": {"max_time_minutes": 5},
        },
    )
    monkeypatch.setattr("gemma_agent_dev.tasks.rotation.Experiment.load", lambda *a: experiment)
    monkeypatch.setattr("gemma_agent_dev.tasks.rotation.TaskPool.load", lambda *a: pool)
    records = [
        dict(
            task_ids=["fastapi_4", "rich_4", "requests_4"],
            metrics={"model_startup_seconds": 100, "evaluation_seconds": 300},
            record_id="old",
            observations=[
                {"task_id": "rich_4", "run_id": "run", "status": "completed", "resolved": False},
                {"task_id": "requests_4", "run_id": None, "status": "not_run", "resolved": None},
            ],
        )
    ]
    monkeypatch.setattr(rotation, "load_records", lambda *a: records)
    (tmp_path / "records").mkdir()
    config = dict(
        experiment="experiment.yaml",
        task_count=6,
        repository_order=["fastapi", "rich", "requests"],
        seed=0,
        records="records",
    )
    path = tmp_path / "rotation.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, records


def test_rotation_is_unused_first_and_independent_of_scores(planner: Any) -> None:
    path, records = planner
    first = rotation.plan(path.parent, path)
    records[0]["observations"][0]["resolved"] = True
    second = rotation.plan(path.parent, path)
    assert first["task_ids"] == second["task_ids"] and len(set(first["task_ids"])) == 6
    assert [task.split("_")[0] for task in first["task_ids"]] == ["fastapi", "rich", "requests"] * 2
    assert "rich_4" not in first["task_ids"] and "fastapi_2" not in first["task_ids"]
    assert first["attempted"] == first["completed"] == ["rich_4"]
    assert first["cpu_setup_seconds"] is None and first["confirmation_tasks_used"] == []
    assert first["historical_extrapolations"][0]["wall_seconds"] == 1600


@pytest.mark.parametrize("damage", ["seed", "repos", "missing-repo", "count", "directory"])
def test_rotation_rejects_invalid_or_exhausted_plans(planner: Any, damage: str) -> None:
    path, _ = planner
    data = yaml.safe_load(path.read_text())
    if damage == "seed":
        data["seed"] = -1
    elif damage == "repos":
        data["repository_order"] = ["fastapi", "fastapi"]
    elif damage == "missing-repo":
        data["repository_order"] = ["fastapi"]
    elif damage == "directory":
        data["records"] = "missing"
    else:
        data["task_count"] = 100
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        rotation.plan(path.parent, path)
