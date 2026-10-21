"""Bound a shared model session and admit only complete task pairs."""

from dataclasses import asdict, dataclass, fields
from typing import Any

from gemma_agent_dev.common.config import mapping, positive
from gemma_agent_dev.evaluation.budget import GpuBudget

VARIANTS_PER_PAIR = 2
SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class SessionPolicy:
    job_timeout_seconds: int
    cleanup_reserve_seconds: int
    per_evaluation_overhead_seconds: int
    enable_prefix_caching: bool
    quota_hours: float
    reserve_hours: float
    quota_per_wall_hour: float
    max_runs: int

    @classmethod
    def parse(cls, raw: Any) -> "SessionPolicy":
        data = mapping(raw, {f.name for f in fields(cls)}, "session policy")
        if data["enable_prefix_caching"] is not False:
            raise ValueError("Paired experiments require prefix caching disabled")
        for name in set(data) - {"enable_prefix_caching"}:
            positive(data[name], name, integer=name.endswith("seconds") or name == "max_runs")
        result = cls(**data)
        result.budget()
        if result.cleanup_reserve_seconds >= result.job_timeout_seconds:
            raise ValueError("Cleanup reserve exceeds the session deadline")
        return result

    def budget(self) -> GpuBudget:
        return GpuBudget.parse(
            {k: v for k, v in asdict(self).items() if k in GpuBudget.__dataclass_fields__}
        )

    def pair_seconds(self, evaluation: dict[str, Any], verification_seconds: int) -> float:
        return float(
            VARIANTS_PER_PAIR
            * (
                evaluation["max_time_minutes"] * SECONDS_PER_MINUTE
                + verification_seconds
                + self.per_evaluation_overhead_seconds
            )
            + self.cleanup_reserve_seconds
        )
