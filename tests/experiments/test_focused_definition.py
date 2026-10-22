from pathlib import Path

import pytest

from gemma_agent_dev.candidates.archive import generation_contract
from gemma_agent_dev.evaluation.config import RuntimeSettings, SamplingSettings
from gemma_agent_dev.evaluation.experiments import Experiment

ROOT = Path(__file__).resolve().parents[2]


def test_full_paired_bounds_and_additional_quota_are_explicit() -> None:
    experiment = Experiment.load(ROOT, Path("configs/evaluation/easy-focused-v1.yaml"))
    definition = experiment.definition(ROOT)
    bound = experiment.session.completion(definition["evaluation"], definition["runtime"], 6)
    assert bound["allowed"] and bound["total_upper_seconds"] == 10620
    assert bound["evaluation_seconds"] == 12 * 660 and bound["cost_kind"] == "upper_bound"
    budget = experiment.session.budget()
    assert budget.admit(consumed=0, remaining=21.5, completed_runs=0, active=False)["allowed"]
    assert budget.admit(consumed=5.9, remaining=15.6, completed_runs=1, active=False)["allowed"]
    assert not budget.admit(consumed=6.2, remaining=15.3, completed_runs=1, active=False)["allowed"]
    assert not budget.admit(consumed=0, remaining=21.5, completed_runs=0, active=True)["allowed"]
    from dataclasses import replace

    with pytest.raises(ValueError, match="cannot fit"):
        replace(experiment.session, job_timeout_seconds=5400).require_completion(
            definition["evaluation"], definition["runtime"], 6
        )
    assert not replace(experiment.session, evaluation_timeout_seconds=None).completion(
        definition["evaluation"], definition["runtime"], 6
    )["allowed"]
    confirm = Experiment.load(ROOT, Path("configs/evaluation/easy-focused-confirmation-v1.yaml"))
    assert confirm.task_ids == [
        "fastapi_13537",
        "rich_3882",
        "requests_6644",
        "fastapi_15589",
        "rich_3894",
        "requests_6629",
    ]
    assert confirm.variants == experiment.variants
    assert confirm.plan()[2]["variant"] == "workflow-focused-v1"
    corrected = Experiment.load(ROOT, Path("configs/evaluation/easy-focused-verification-v2.yaml"))
    assert corrected.task_ids == experiment.task_ids
    assert list(corrected.variants) == ["workflow-v1", "workflow-focused-v2"]
    assert generation_contract(ROOT / "agent", corrected.variants["workflow-focused-v2"]) == (
        generation_contract(ROOT / "agent", experiment.variants["workflow-focused-v1"])
    )


def test_actual_focused_generation_is_validated_without_fixed_file_name(tmp_path: Path) -> None:
    runtime = RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml")
    actual = generation_contract(ROOT / "agent", ROOT / "agent/workflow-focused-v1.yaml")
    assert len(actual) == 1
    config = SamplingSettings.parse(actual[0]["generation"], runtime)
    assert config.max_output_tokens == 8192 and config.thinking_config["thinking_budget"] == 4096
    (tmp_path / "agent.yaml").write_text(
        "model: fixed\ngenerate_content_config: !include renamed.yaml\n"
    )
    (tmp_path / "sampling.yaml").write_text("unused: true\n")
    (tmp_path / "renamed.yaml").write_text("temperature: 0.2\nmax_output_tokens: 999999\n")
    value = generation_contract(tmp_path)[0]["generation"]
    assert value["max_output_tokens"] == 999999
    with pytest.raises(ValueError):
        SamplingSettings.parse(value, runtime)
