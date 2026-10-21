"""Admit sequential GPU jobs using observed quota and a conservative next-job bound."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.evaluation.config import RuntimeSettings

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class GpuBudget:
    quota_hours: float
    reserve_hours: float
    max_runs: int
    job_timeout_seconds: int
    quota_per_wall_hour: float

    @classmethod
    def load(cls, root: Path, path: Path) -> "GpuBudget":
        data = mapping(
            yaml.safe_load(project_path(root, str(path)).read_text()),
            {"runtime_config", "quota_hours", "reserve_hours", "max_runs", "quota_per_wall_hour"},
            "GPU budget file",
        )
        runtime = RuntimeSettings.load(project_path(root, data.pop("runtime_config")))
        return cls.parse({**data, "job_timeout_seconds": runtime.job_timeout_seconds})

    @classmethod
    def parse(cls, value: Any) -> "GpuBudget":
        data = mapping(value, {f.name for f in fields(cls)}, "GPU budget")
        for key, item in data.items():
            positive(item, key, integer=key in {"max_runs", "job_timeout_seconds"})
        if data["reserve_hours"] >= data["quota_hours"]:
            raise ValueError("GPU reserve must be below the total budget")
        return cls(**data)

    def admit(
        self, *, consumed: float, remaining: float, completed_runs: int, active: bool
    ) -> dict[str, Any]:
        for name, item in (("consumed", consumed), ("remaining", remaining)):
            if type(item) not in (int, float) or not 0 <= item < float("inf"):
                raise ValueError(f"Unknown or invalid observed GPU quota: {name}")
        if type(completed_runs) is not int or completed_runs < 0 or type(active) is not bool:
            raise ValueError("Invalid GPU run state")
        bound = self.job_timeout_seconds / SECONDS_PER_HOUR * self.quota_per_wall_hour
        allowed = (
            not active
            and completed_runs < self.max_runs
            and consumed + bound + self.reserve_hours <= self.quota_hours
            and remaining >= bound + self.reserve_hours
        )
        return {
            "allowed": allowed,
            "consumed_quota_hours": consumed,
            "remaining_quota_hours": remaining,
            "next_job_max_quota_hours": bound,
            "reserve_quota_hours": self.reserve_hours,
            "completed_runs": completed_runs,
            "active": active,
        }
