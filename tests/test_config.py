from pathlib import Path

import pytest
import yaml

from gemma_agent_dev.config import EvaluationSettings

ROOT = Path(__file__).resolve().parents[1]


def test_official_budgets_are_explicit() -> None:
    assert EvaluationSettings.load(ROOT / "agent/eval_config.yaml").as_dict() == {
        "timeout_seconds": 60,
        "max_tool_calls": 10,
        "max_time_minutes": 1,
        "max_turns": 50,
    }


@pytest.mark.parametrize("value", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_time_budget_is_rejected(tmp_path: Path, value: object) -> None:
    values = EvaluationSettings.load(ROOT / "agent/eval_config.yaml").as_dict()
    values["max_time_minutes"] = value
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"evaluation": values}))
    with pytest.raises(ValueError):
        EvaluationSettings.load(path)


def test_missing_budget_does_not_fall_back_to_cli_default(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("evaluation:\n  max_tool_calls: 10\n")
    with pytest.raises(ValueError, match="Expected evaluation keys"):
        EvaluationSettings.load(path)
