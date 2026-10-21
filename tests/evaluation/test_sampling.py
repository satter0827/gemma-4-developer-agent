from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.evaluation.config import RuntimeSettings, SamplingSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", -1),
        ("top_p", 1.1),
        ("temperature", float("nan")),
        ("thinking_config", {"thinking_budget": True, "include_thoughts": True}),
        ("thinking_config", {"thinking_budget": 20000, "include_thoughts": True}),
        ("max_output_tokens", 32768),
    ],
)
def test_sampling_rejects_invalid_values_and_context_overflow(
    tmp_path: Path, key: str, value: Any
) -> None:
    data = yaml.safe_load((ROOT / "agent/configs/sampling.yaml").read_text())
    data[key] = value
    path = tmp_path / "sampling.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        SamplingSettings.load(path, RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml"))
