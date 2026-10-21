from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.budget import GpuBudget

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_admission_uses_quota_hours_and_stops_on_unknown_or_active_runs() -> None:
    budget = GpuBudget.load(ROOT, Path("configs/evaluation/gpu-budget.yaml"))
    assert budget.admit(consumed=0, remaining=26.5, completed_runs=0, active=False)["allowed"]
    assert not budget.admit(consumed=2, remaining=24, completed_runs=2, active=False)["allowed"]
    assert not budget.admit(consumed=0, remaining=0.5, completed_runs=0, active=False)["allowed"]
    assert not budget.admit(consumed=0, remaining=20, completed_runs=0, active=True)["allowed"]
    assert not budget.admit(consumed=0, remaining=20, completed_runs=3, active=False)["allowed"]
    invalid_values: list[Any] = [None, float("nan"), -1]
    for value in invalid_values:
        with pytest.raises(ValueError):
            budget.admit(consumed=value, remaining=20, completed_runs=0, active=False)
    with pytest.raises(ValueError):
        GpuBudget.parse({**asdict(budget), "reserve_hours": 4})


@pytest.mark.parametrize(
    "consumed,active,code", [("0", False, 0), ("2", False, 1), ("0", True, 1), ("nan", False, 1)]
)
def test_cli_budget_admission_has_a_failure_exit_status(
    diagnostic_project: DiagnosticSettings,
    monkeypatch: pytest.MonkeyPatch,
    consumed: str,
    active: bool,
    code: int,
) -> None:
    from gemma_agent_dev.cli import main

    root = diagnostic_project.root
    budget = yaml.safe_load((ROOT / "configs/evaluation/gpu-budget.yaml").read_text())
    budget["runtime_config"] = "runtime.json"
    (root / "budget.yaml").write_text(yaml.safe_dump(budget))
    monkeypatch.chdir(root)
    args = [
        "check-gpu-budget",
        "--config",
        "budget.yaml",
        "--consumed-hours",
        consumed,
        "--remaining-hours",
        "20",
        "--completed-runs",
        "0",
    ]
    assert main(args + (["--active"] if active else [])) == code
