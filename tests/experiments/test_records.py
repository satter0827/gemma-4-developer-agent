import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.evaluation.experiment_records import save
from gemma_agent_dev.evaluation.record_schema import validate_extra
from gemma_agent_dev.evaluation.tracking import load_records

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def record() -> dict[str, Any]:
    row = json.loads((ROOT / "docs/records/experiments/easy-sequential-cpu-v1.json").read_text())
    row.update(
        record_schema_version=2,
        record_id="observation-test",
        conditions={"max_time_minutes": 5, "sdk_version": "0.2.11"},
        quota_hours=None,
        decision="unevaluated",
        parent_candidates={name: name for name in row["candidate_hashes"]},
        assessments=[
            {
                "category": "verification",
                "certainty": "unknown",
                "basis": "Model accuracy has not been evaluated.",
            }
        ],
        observations=[],
    )
    return dict(row)


def observed(task: str, variant: str) -> dict[str, Any]:
    return dict(
        task_id=task,
        variant=variant,
        run_id="run-1",
        resolved=False,
        duration_seconds=12.5,
        status="completed",
        termination="completed",
        implementation_changed=None,
        tool_argument_errors=None,
    )


def test_public_records_preserve_unknown_values_without_private_artifacts(
    record: dict[str, Any],
) -> None:
    record["observations"] = [observed(record["task_ids"][0], "workflow-v1")]
    parsed = load_records(ROOT, [], records=[record])[0]
    assert parsed["quota_hours"] is None
    assert parsed["observations"][0]["implementation_changed"] is None
    with pytest.raises(ValueError, match="paths or"):
        load_records(ROOT, [Path("anything")], records=[record])


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "extra",
        "quota",
        "condition",
        "condition-limit",
        "condition-model",
        "decision",
        "parents",
        "certainty",
        "category",
        "basis",
        "duplicate",
        "foreign",
        "boolean",
        "duration",
        "status",
        "termination",
        "incomplete",
        "receipt",
        "not-run",
        "missing-pair",
        "assessments",
        "observations",
    ],
)
def test_invalid_public_observations_are_rejected(record: dict[str, Any], damage: str) -> None:
    item = observed(record["task_ids"][0], "workflow-v1")
    record["observations"] = [item]
    if damage == "schema":
        record["record_schema_version"] = 3
    elif damage == "extra":
        del record["quota_hours"]
    elif damage == "quota":
        record["quota_hours"] = -1
    elif damage == "condition":
        record["conditions"] = {"raw_trace": "forbidden"}
    elif damage == "condition-limit":
        record["conditions"]["max_time_minutes"] = float("nan")
    elif damage == "condition-model":
        record["conditions"]["model"] = "raw text"
    elif damage == "decision":
        record["decision"] = "automatic"
    elif damage == "parents":
        record["parent_candidates"] = {"workflow-v1": "workflow-v1"}
    elif damage == "certainty":
        record["assessments"][0]["certainty"] = "automatic"
    elif damage == "category":
        record["assessments"][0]["category"] = "other"
    elif damage == "basis":
        record["assessments"][0]["basis"] = "two\nlines"
    elif damage == "duplicate":
        record["observations"].append(copy.deepcopy(item))
    elif damage == "foreign":
        item["variant"] = "foreign"
    elif damage == "boolean":
        item["resolved"] = 0
    elif damage == "duration":
        item["duration_seconds"] = -1
    elif damage == "status":
        item["status"] = "success"
    elif damage == "termination":
        item["termination"] = "raw error"
    elif damage == "incomplete":
        item["status"] = "incomplete"
    elif damage == "receipt":
        item["run_id"] = None
    elif damage == "not-run":
        item.update(status="not_run", resolved=None)
    elif damage == "missing-pair":
        record["status"] = "evaluated"
    elif damage == "assessments":
        record["assessments"] = {}
    else:
        record["observations"] = None
    with pytest.raises((ValueError, KeyError)):
        validate_extra(record)


def test_evaluated_record_requires_complete_cross_product(record: dict[str, Any]) -> None:
    record["status"] = "evaluated"
    record["observations"] = [
        observed(task, variant)
        for task in record["task_ids"]
        for variant in record["candidate_hashes"]
    ]
    validate_extra(record)
    record["observations"][0].update(status="incomplete", resolved=None)
    with pytest.raises(ValueError, match="complete"):
        validate_extra(record)


def test_public_record_creation_is_immutable_and_location_limited(
    record: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gemma_agent_dev.evaluation.tracking as tracking

    monkeypatch.setattr(tracking, "load_records", lambda *a, **kw: [record])
    path = tmp_path / "docs/records/experiments/record.json"
    assert save(tmp_path, path, record) == path
    with pytest.raises(FileExistsError):
        save(tmp_path, path, record)
    with pytest.raises(ValueError, match="Public records"):
        save(tmp_path, tmp_path / "elsewhere.json", record)


def test_aggregate_mismatch_and_unknown_started_cells(record: dict[str, Any]) -> None:
    from gemma_agent_dev.evaluation.experiment_results import observation

    row = observation(None, "task", "candidate", 100, attempted=True)
    assert row["status"] == "incomplete" and row["resolved"] is None and row["run_id"] is None
    record["task_ids"] = record["task_ids"][:1]
    record["observations"] = [
        observed(record["task_ids"][0], name) for name in record["candidate_hashes"]
    ]
    record["metrics"] = dict(
        both=0, baseline_only=0, candidate_only=0, neither=1, errors=0, incomplete=0, not_run=0
    )
    validate_extra(record)
    record["metrics"]["both"] = 1
    with pytest.raises(ValueError, match="aggregates disagree"):
        validate_extra(record)


def test_reviewed_export_uses_frozen_numbers_and_path_specific_prompts(
    record: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from gemma_agent_dev.common.artifacts import file_hash, write_json
    from gemma_agent_dev.evaluation import experiment_records as records

    (tmp_path / "agent/prompts").mkdir(parents=True)
    prompt = tmp_path / "agent/prompts/main.md"
    prompt.write_text("authored instruction")
    note = dict(
        record_id="reviewed",
        interpretation="Reviewed outcome.",
        decision="inconclusive",
        parent_candidates={"a": "parent", "b": "parent"},
        assessments=[],
        prompts=[dict(prompt_id="main-v1", path="agent/prompts/main.md")],
    )
    review = tmp_path / "review.yaml"
    review.write_text(yaml.safe_dump(note))
    sealed: dict[str, Any] = dict(
        experiment_id="trial",
        definition_sha256="0" * 64,
        source_sha256="1" * 64,
        variants={"a": "2" * 64, "b": "3" * 64},
        candidate_files={"a": {"prompts/main.md": file_hash(prompt)}},
        definition=dict(
            hypothesis="test",
            task_ids=["task_1"],
            evaluation={"max_time_minutes": 5},
            runtime={
                "packages": {"swegemma": "0.2.11"},
                "verification_timeout_seconds": 120,
                "vllm": {"max_model_len": 32768},
            },
        ),
    )
    session = tmp_path / "session.json"
    write_json(session, sealed)
    monkeypatch.setattr(
        records,
        "summarize",
        lambda *a, **kw: dict(
            status="completed", counts={"both": 0}, cell_errors=0, observations=[], quota_hours=None
        ),
    )
    saved: list[dict[str, Any]] = []

    def save_review(root: Path, destination: Path, row: dict[str, Any]) -> Path:
        saved.append(row)
        return destination

    monkeypatch.setattr(records, "save", save_review)
    records.export(tmp_path, session, review, Path("record.json"), 1000)
    assert saved[0]["quota_hours"] is None and saved[0]["prompts"][0]["sha256"] == file_hash(prompt)
    assert saved[0]["decision"] == "inconclusive"
    sealed["candidate_files"]["a"] = {"other.md": file_hash(prompt)}
    write_json(session, sealed)
    with pytest.raises(ValueError, match="prompt"):
        records.export(tmp_path, session, review, Path("record.json"), 1000)
    sealed.pop("definition")
    write_json(session, sealed)
    with pytest.raises(ValueError, match="definition"):
        records.export(tmp_path, session, review, Path("record.json"), 1000)
