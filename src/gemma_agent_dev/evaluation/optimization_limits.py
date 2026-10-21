"""Validate independent search limits before every evaluation and reflection."""

import time
from dataclasses import dataclass
from typing import Any

from gemma_agent_dev.common.config import mapping, positive
from gemma_agent_dev.common.events import event

SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class SearchLimits:
    max_metric_calls: int
    max_reflection_calls: int
    reflection_minibatch_size: int
    seed: int
    max_input_tokens: int
    max_output_tokens: int
    reflection_timeout_seconds: int
    cpu_timeout_seconds: int
    cleanup_reserve_seconds: int
    per_evaluation_overhead_seconds: int

    @classmethod
    def parse(cls, value: Any) -> "SearchLimits":
        data = mapping(value, set(cls.__dataclass_fields__), "optimization limits")
        for key, item in data.items():
            if key == "seed":
                if type(item) is not int or item < 0:
                    raise ValueError("Expected a nonnegative search seed")
            else:
                positive(item, key, integer=True)
        return cls(**data)

    def required_seconds(self, runtime: dict[str, Any], evaluation: dict[str, Any]) -> float:
        return float(
            runtime["vllm"]["startup_timeout"]
            + self.max_metric_calls * self.cell_seconds(runtime, evaluation)
            + self.max_reflection_calls * self.reflection_timeout_seconds
            + self.cpu_timeout_seconds
            + self.cleanup_reserve_seconds
        )

    def cell_seconds(self, runtime: dict[str, Any], evaluation: dict[str, Any]) -> float:
        return float(
            evaluation["max_time_minutes"] * SECONDS_PER_MINUTE
            + runtime["verification_timeout_seconds"]
            + self.per_evaluation_overhead_seconds
        )


@dataclass
class SearchBudget:
    limits: SearchLimits
    deadline: float
    cell_seconds: float
    metric_calls: int = 0
    reflection_calls: int = 0

    def check(self, *, cells: int = 0, reflection: bool = False) -> None:
        if type(cells) is not int or cells < 0:
            raise ValueError("Expected a nonnegative evaluation count")
        event(
            "optimization.budget.checked",
            level="DEBUG",
            count=cells,
            metric_calls=self.metric_calls,
            reflection_calls=self.reflection_calls,
            remaining_seconds=self.deadline - time.monotonic(),
        )
        if self.metric_calls + cells > self.limits.max_metric_calls:
            raise RuntimeError("Optimization evaluation limit reached")
        if reflection and self.reflection_calls >= self.limits.max_reflection_calls:
            raise RuntimeError("Optimization reflection limit reached")
        required = self.limits.cleanup_reserve_seconds + cells * self.cell_seconds
        if reflection:
            required += self.limits.reflection_timeout_seconds
        if self.deadline - time.monotonic() < required:
            raise TimeoutError("Insufficient optimization time before cleanup")

    def stop(self, state: Any) -> bool:
        return (
            self.metric_calls >= self.limits.max_metric_calls
            or self.reflection_calls >= self.limits.max_reflection_calls
            or self.deadline - time.monotonic() <= self.limits.cleanup_reserve_seconds
        )
