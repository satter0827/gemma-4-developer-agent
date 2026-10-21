import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.candidates.archive import referenced_files
from gemma_agent_dev.candidates.preflight import PreflightSettings
from gemma_agent_dev.common.artifacts import content_hash
from gemma_agent_dev.evaluation.experiments import Experiment
from gemma_agent_dev.tasks.pools import TaskPool, identifier, task_ids

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_easy_pool_counts_roles_and_first_use() -> None:
    pool = TaskPool.load(ROOT, Path("configs/tasks/easy.yaml"))
    assert len(pool.tuning) == 36 and len(pool.confirmation) == 12
    assert Counter(i.split("_")[0] for i in pool.tuning) == {
        "fastapi": 14,
        "rich": 15,
        "requests": 7,
    }
    assert Counter(i.split("_")[0] for i in pool.confirmation) == {
        "fastapi": 5,
        "rich": 5,
        "requests": 2,
    }
    assert pool.selection("easy-sequential-v1") == [
        "fastapi_14873",
        "fastapi_14479",
        "requests_7328",
        "requests_7433",
        "rich_3130",
        "rich_3278",
    ]
    ledger = (ROOT / "docs/reference/tasks.md").read_text()
    assert set(pool.tuning + pool.confirmation) == {
        line.split("|")[1].strip() for line in ledger.splitlines() if "| Easy " in line
    }
    with pytest.raises(ValueError, match="explicit"):
        pool.selection("unknown")


@pytest.mark.parametrize("value", [None, "UPPER", "../a", 1, ""])
def test_invalid_ids(value: Any) -> None:
    with pytest.raises(ValueError):
        identifier(value)


@pytest.mark.parametrize("value", [[], ["a", "a"], None, [1], ["bad/id"]])
def test_invalid_task_lists(value: Any) -> None:
    with pytest.raises(ValueError):
        task_ids(value)


@pytest.mark.parametrize(
    "damage", ["overlap", "purpose", "uses", "duplicate", "foreign", "tier", "hash"]
)
def test_pool_rejects_invalid_role_history(experiment_project: Path, damage: str) -> None:
    p = experiment_project / "pool.yaml"
    d = yaml.safe_load(p.read_text())
    if damage == "overlap":
        d["confirmation"] = d["tuning"]
    elif damage == "purpose":
        d["uses"][0]["purpose"] = "unknown"
    elif damage == "uses":
        d["uses"] = "none"
    elif damage == "duplicate":
        d["uses"] *= 2
    elif damage == "foreign":
        d["uses"][0]["task_ids"] = ["unused_1"]
    elif damage == "tier":
        d["difficulty"] = "Medium"
    else:
        d["tasks_sha256"] = "bad"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        TaskPool.load(experiment_project, p)


def test_explicit_abba_order_and_resolved_definition(experiment_project: Path) -> None:
    root = experiment_project
    pool = yaml.safe_load((root / "pool.yaml").read_text())
    pool["tuning"] = ["fixture_2", "fixture_1"]
    pool["uses"][0]["task_ids"] = pool["tuning"]
    (root / "pool.yaml").write_text(yaml.safe_dump(pool))
    experiment = Experiment.load(root, Path("experiment.yaml"))
    assert experiment.plan() == [
        {"task_id": "fixture_2", "variant": "workflow-v1"},
        {"task_id": "fixture_2", "variant": "sequential-v1"},
        {"task_id": "fixture_1", "variant": "sequential-v1"},
        {"task_id": "fixture_1", "variant": "workflow-v1"},
    ]
    d = experiment.definition(root)
    assert content_hash(d) == content_hash(json.loads(json.dumps(d)))
    assert "prompts/workflow.md" in d["variants"]["workflow-v1"]
    assert "prompts/verify-v1.md" not in d["variants"]["workflow-v1"]
    assert experiment.session.pair_seconds(d["evaluation"], 300) == 1140
    budget = experiment.session.budget()
    assert not budget.admit(consumed=2.8303, remaining=20, completed_runs=2, active=False)[
        "allowed"
    ]
    assert budget.admit(consumed=0, remaining=4, completed_runs=0, active=False)["allowed"]
    with pytest.raises(ValueError):
        budget.admit(
            consumed=0,
            remaining=None,  # type: ignore[arg-type]  # Reject an unknown external quota.
            completed_runs=0,
            active=False,
        )


@pytest.mark.parametrize(
    "damage", ["missing", "same", "outside", "count", "pair", "cache", "reserve", "timeout"]
)
def test_invalid_experiment(experiment_project: Path, damage: str) -> None:
    p = experiment_project / "experiment.yaml"
    d = yaml.safe_load(p.read_text())
    if damage == "missing":
        d["evaluation_config"] = "missing.yaml"
    elif damage == "same":
        d["variants"]["sequential-v1"] = d["variants"]["workflow-v1"]
    elif damage == "outside":
        d["variants"]["sequential-v1"] = "configs/evaluation/comparison.yaml"
    elif damage == "count":
        del d["variants"]["sequential-v1"]
    elif damage == "pair":
        d["session"]["job_timeout_seconds"] = 1000
    elif damage == "cache":
        d["session"]["enable_prefix_caching"] = True
    elif damage == "reserve":
        d["session"]["cleanup_reserve_seconds"] = 5400
    else:
        d["session"]["job_timeout_seconds"] = 0
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        Experiment.load(experiment_project, p)


def test_unused_prompts_do_not_invalidate_selected_profile(preflight_project: Path) -> None:
    root = preflight_project
    before = PreflightSettings.load(root, Path("preflight.yaml"), "smoke").inputs()
    unused = root / "agent/prompts/verify-v1.md"
    unused.write_text("changed unused instruction")
    assert PreflightSettings.load(root, Path("preflight.yaml"), "smoke").inputs() == before
    (root / "agent/sub_agents/code_analyzer.yaml").write_text("changed selected child")
    assert PreflightSettings.load(root, Path("preflight.yaml"), "smoke").inputs() != before


def test_selected_reference_closure_rejects_cycle_and_escape(tmp_path: Path) -> None:
    p = tmp_path / "agent.yaml"
    p.write_text("instruction: !include unused.md")
    (tmp_path / "unused.md").write_text("included")
    (tmp_path / "bad.yaml").write_text("invalid: !include nowhere")
    assert set(referenced_files(tmp_path, p)) == {"agent.yaml", "unused.md"}
    for text in [
        "instruction: !include agent.yaml",
        "instruction: !include ../outside.md",
        "sub_agents: [{config_path: 12}]",
    ]:
        p.write_text(text)
        with pytest.raises(ValueError):
            referenced_files(tmp_path, p)


def test_unrelated_pool_use_does_not_change_prior_experiment(experiment_project: Path) -> None:
    p = experiment_project / "pool.yaml"
    before = Experiment.load(experiment_project, Path("experiment.yaml")).definition(
        experiment_project
    )
    data = yaml.safe_load(p.read_text())
    data["uses"].append(
        {"experiment_id": "next-experiment", "purpose": "tuning", "task_ids": ["fixture_1"]}
    )
    p.write_text(yaml.safe_dump(data))
    after = Experiment.load(experiment_project, Path("experiment.yaml")).definition(
        experiment_project
    )
    assert before == after
