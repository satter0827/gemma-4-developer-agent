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
    preparation_timeout_seconds: int | None = None
    evaluation_timeout_seconds: int | None = None

    @classmethod
    def parse(cls, raw: Any) -> "SessionPolicy":
        names = {f.name for f in fields(cls)}
        # Historical definitions have no preparation bound; unknown never admits a GPU job.
        for optional in ("preparation_timeout_seconds", "evaluation_timeout_seconds"):
            if isinstance(raw, dict) and optional not in raw:
                names.remove(optional)
        data = mapping(raw, names, "session policy")
        if data["enable_prefix_caching"] is not False:
            raise ValueError("Paired experiments require prefix caching disabled")
        for name in set(data) - {"enable_prefix_caching"}:
            if (
                name in {"preparation_timeout_seconds", "evaluation_timeout_seconds"}
                and data[name] is None
            ):
                continue
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

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for optional in ("preparation_timeout_seconds", "evaluation_timeout_seconds"):
            if value[optional] is None:
                # Preserve immutable historical definitions.
                value.pop(optional)
        return value

    def pair_seconds(self, evaluation: dict[str, Any], verification_seconds: int) -> float:
        if self.evaluation_timeout_seconds is not None:
            return float(
                VARIANTS_PER_PAIR * self.evaluation_timeout_seconds + self.cleanup_reserve_seconds
            )
        return float(
            VARIANTS_PER_PAIR
            * (
                evaluation["max_time_minutes"] * SECONDS_PER_MINUTE
                + verification_seconds
                + self.per_evaluation_overhead_seconds
            )
            + self.cleanup_reserve_seconds
        )

    def completion(
        self, evaluation: dict[str, Any], runtime: dict[str, Any], count: int
    ) -> dict[str, Any]:
        pairs = count * (
            self.pair_seconds(evaluation, runtime["verification_timeout_seconds"])
            - self.cleanup_reserve_seconds
        )
        known = runtime["vllm"]["startup_timeout"] + pairs + self.cleanup_reserve_seconds
        preparation = self.preparation_timeout_seconds
        estimate = None if preparation is None else known + preparation
        total = estimate if self.evaluation_timeout_seconds is not None else None
        return {
            "cost_kind": "upper_bound"
            if self.evaluation_timeout_seconds is not None
            else "estimate",
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "pair_count": count,
            "preparation_seconds": preparation,
            "model_startup_seconds": runtime["vllm"]["startup_timeout"],
            "evaluation_seconds": pairs,
            "cleanup_seconds": self.cleanup_reserve_seconds,
            "known_upper_seconds": known if self.evaluation_timeout_seconds is not None else None,
            "known_estimated_seconds": known if self.evaluation_timeout_seconds is None else None,
            "estimated_total_seconds": estimate
            if self.evaluation_timeout_seconds is None
            else None,
            "total_upper_seconds": total,
            "job_timeout_seconds": self.job_timeout_seconds,
            "missing_seconds": None if total is None else max(0, total - self.job_timeout_seconds),
            "allowed": total is not None and total <= self.job_timeout_seconds,
        }

    def require_completion(
        self, evaluation: dict[str, Any], runtime: dict[str, Any], count: int
    ) -> None:
        bound = self.completion(evaluation, runtime, count)
        if not bound["allowed"]:
            raise ValueError(f"Complete experiment cannot fit the configured deadline: {bound}")
