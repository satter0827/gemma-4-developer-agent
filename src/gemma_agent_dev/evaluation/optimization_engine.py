"""Run the actual GEPA library through explicit bounded, isolated callbacks."""

import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gemma_agent_dev.common.events import context, event
from gemma_agent_dev.evaluation.analysis import SCORABLE_TERMINATIONS
from gemma_agent_dev.evaluation.optimization_limits import SearchBudget

GEPA_VERSION = "0.1.4"
COMPONENT = "instruction"
REFLECTION_TEMPLATE = """Improve a general software-repair procedure.
Preserve embedded variables exactly. Do not include task IDs, repository names,
concrete patches, reference solutions, or issue-specific answers. Reason from
the observed behavior and improve investigation, implementation and verification.
Current procedure:
<curr_param>
Behavioral feedback:
<side_info>
Return only the improved procedure in one fenced code block.
"""


@dataclass(frozen=True)
class SearchOutcome:
    resolved: bool
    termination: str
    implementation_changed: bool | None
    tool_argument_errors: int | None

    @classmethod
    def parse(cls, value: Any) -> "SearchOutcome":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Invalid optimization evaluation outcome")
        if type(value["resolved"]) is not bool or value["termination"] not in SCORABLE_TERMINATIONS:
            raise RuntimeError("Infrastructure errors or incomplete evaluations cannot be scored")
        if (
            value["implementation_changed"] is not None
            and type(value["implementation_changed"]) is not bool
        ):
            raise ValueError("Unknown implementation changes must stay null")
        errors = value["tool_argument_errors"]
        if errors is not None and (type(errors) is not int or errors < 0):
            raise ValueError("Invalid tool error count")
        return cls(**value)

    def feedback(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def search(
    instruction: str,
    train_ids: list[str],
    selection_ids: list[str],
    budget: SearchBudget,
    freeze: Callable[[str], str],
    evaluate: Callable[[str, str], dict[str, Any]],
    reflect: Callable[[list[dict[str, str]]], str],
) -> dict[str, Any]:
    import gepa
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter

    if importlib.metadata.version("gepa") != GEPA_VERSION:
        raise ValueError("Optimization requires the pinned GEPA version")
    if (
        not train_ids
        or not selection_ids
        or len(set(train_ids + selection_ids)) != len(train_ids + selection_ids)
    ):
        raise ValueError("Search and selection sets must be nonempty and disjoint")
    candidates: dict[str, str] = {}
    outcomes: list[dict[str, Any]] = []

    def candidate_id(text: str) -> str:
        if text not in candidates:
            candidates[text] = freeze(text)
        return candidates[text]

    class Adapter(GEPAAdapter[str, dict[str, Any], dict[str, Any]]):  # type: ignore[misc]  # GEPA is isolated from the development environment.
        def evaluate(
            self, batch: list[str], candidate: dict[str, str], capture_traces: bool = False
        ) -> Any:
            if set(candidate) != {COMPONENT} or any(
                t not in train_ids + selection_ids for t in batch
            ):
                raise ValueError(
                    "Optimization cannot change components or evaluate confirmation tasks"
                )
            budget.check(cells=len(batch))
            key = candidate_id(candidate[COMPONENT])
            values = []
            for task in batch:
                budget.check(cells=1)
                budget.metric_calls += 1
                with context(task_id=task, variant=key):
                    event("optimization.evaluation.started", metric_calls=budget.metric_calls)
                    value = SearchOutcome.parse(evaluate(key, task)).feedback()
                    event("optimization.evaluation.completed", resolved=value["resolved"])
                values.append(value)
                outcomes.append({"candidate_id": key, "task_id": task, **value})
            return EvaluationBatch(
                outputs=values,
                scores=[float(v["resolved"]) for v in values],
                trajectories=values if capture_traces else None,
            )

        def make_reflective_dataset(
            self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]
        ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
            if components_to_update != [COMPONENT]:
                raise ValueError("Only the main instruction can be optimized")
            # No issue text, reference patch, task ID or model output reaches reflection.
            return {COMPONENT: [{"Feedback": value} for value in eval_batch.outputs]}

    def reflection(messages: list[dict[str, str]]) -> str:
        budget.check(reflection=True)
        budget.reflection_calls += 1
        event("optimization.reflection.started", reflection_calls=budget.reflection_calls)
        answer = reflect(messages)
        event("optimization.reflection.completed")
        return answer

    candidate_id(instruction)
    result = gepa.optimize(
        seed_candidate={COMPONENT: instruction},
        trainset=train_ids,
        valset=selection_ids,
        adapter=Adapter(),
        reflection_lm=reflection,
        reflection_prompt_template=REFLECTION_TEMPLATE,
        reflection_minibatch_size=budget.limits.reflection_minibatch_size,
        max_metric_calls=budget.limits.max_metric_calls,
        stop_callbacks=budget.stop,
        use_merge=False,
        seed=budget.limits.seed,
        use_mlflow=False,
        use_wandb=False,
        cache_evaluation=False,
        display_progress_bar=False,
        raise_on_exception=True,
        track_best_outputs=False,
    )
    best = candidate_id(result.best_candidate[COMPONENT])
    return {
        "status": "completed",
        "best_candidate_id": best,
        "candidate_ids": list(candidates.values()),
        "evaluations": outcomes,
        "metric_calls": budget.metric_calls,
        "reflection_calls": budget.reflection_calls,
        "improved": result.best_candidate[COMPONENT] != instruction,
        "adopted": False,
        "stop_reason": "evaluation_limit"
        if budget.metric_calls >= budget.limits.max_metric_calls
        else "reflection_limit"
        if budget.reflection_calls >= budget.limits.max_reflection_calls
        else "completed",
    }
