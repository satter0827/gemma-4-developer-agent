"""Validate bounded public observations without loading private execution evidence."""

import re
from typing import Any

from gemma_agent_dev.common.config import mapping
from gemma_agent_dev.evaluation.analysis import SCORABLE_TERMINATIONS
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR
from gemma_agent_dev.tasks.pools import identifier

RECORD_SCHEMA_VERSION = 2

EXTRA_FIELDS = frozenset(
    {
        "record_schema_version",
        "observations",
        "conditions",
        "quota_hours",
        "decision",
        "parent_candidates",
        "assessments",
    }
)
DECISIONS = frozenset({"unevaluated", "inconclusive", "provisional", "adopt", "reject"})
CATEGORIES = frozenset(
    {"expectation", "exploration", "tool", "repair", "verification", "budget", "environment"}
)
CONDITION_FIELDS = frozenset(
    {
        "model",
        "sdk_version",
        "max_time_minutes",
        "max_tool_calls",
        "max_turns",
        "verification_timeout_seconds",
        "max_model_len",
    }
)
OBSERVATION_FIELDS = frozenset(
    {
        "task_id",
        "variant",
        "run_id",
        "resolved",
        "duration_seconds",
        "status",
        "termination",
        "implementation_changed",
        "tool_argument_errors",
    }
)
PAIR_COUNTS = frozenset(
    {"both", "baseline_only", "candidate_only", "neither", "errors", "incomplete", "not_run"}
)


def finite(value: Any) -> bool:
    return type(value) in (int, float) and 0 <= value < float("inf")


def review(value: Any) -> None:
    if (
        value["decision"] not in DECISIONS
        or not isinstance(value["parent_candidates"], dict)
        or not value["parent_candidates"]
    ):
        raise ValueError("Expected an explicit decision and parent candidates")
    for name, parent in value["parent_candidates"].items():
        identifier(name)
        identifier(parent)
    if not isinstance(value["assessments"], list):
        raise ValueError("Expected reviewed cause assessments")
    for assessment in value["assessments"]:
        mapping(assessment, {"category", "certainty", "basis"}, "cause assessment")
        if assessment["category"] not in CATEGORIES or assessment["certainty"] not in {
            "confirmed",
            "hypothesis",
            "unknown",
        }:
            raise ValueError("Invalid cause classification")
        if (
            not isinstance(assessment["basis"], str)
            or not assessment["basis"].strip()
            or len(assessment["basis"].splitlines()) != 1
        ):
            raise ValueError("Cause assessment needs a short single-line basis")


def validate_extra(row: dict[str, Any]) -> None:
    if (
        row["record_schema_version"] != RECORD_SCHEMA_VERSION
        or set(row) & EXTRA_FIELDS != EXTRA_FIELDS
    ):
        raise ValueError("Invalid public observation schema")
    review(row)
    if set(row["parent_candidates"]) != set(row["candidate_hashes"]):
        raise ValueError("Every candidate needs an explicit parent")
    if row["quota_hours"] is not None and not finite(row["quota_hours"]):
        raise ValueError("Unknown quota must stay null")
    conditions = row["conditions"]
    if not isinstance(conditions, dict) or not conditions or set(conditions) - CONDITION_FIELDS:
        raise ValueError("Unexpected public conditions")
    for key, value in conditions.items():
        if key in {"model", "sdk_version"}:
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError("Invalid public model or SDK version")
        elif not finite(value):
            raise ValueError("Invalid public generation limit")
    if not isinstance(row["observations"], list):
        raise ValueError("Expected task observations")
    seen = set()
    for item in row["observations"]:
        mapping(item, set(OBSERVATION_FIELDS), "task observation")
        pair = item["task_id"], item["variant"]
        if pair in seen or pair[0] not in row["task_ids"] or pair[1] not in row["candidate_hashes"]:
            raise ValueError("Duplicate or foreign task observation")
        seen.add(pair)
        if item["run_id"] is not None:
            if not isinstance(item["run_id"], str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", item["run_id"]
            ):
                raise ValueError("Expected a stable run identifier")
        for field in ("resolved", "implementation_changed"):
            if item[field] is not None and type(item[field]) is not bool:
                raise ValueError("Unknown outcomes must stay null")
        for field in ("duration_seconds", "tool_argument_errors"):
            if item[field] is not None and not finite(item[field]):
                raise ValueError("Invalid observation measurement")
        if item["status"] not in {"completed", "incomplete", "not_run"} or item[
            "termination"
        ] not in {
            "completed",
            "unknown",
            "execution_error",
            "context_window_exceeded",
            "time_limit_reached",
            "tool_limit_reached",
            "turn_limit_reached",
            "verification_timeout",
        }:
            raise ValueError("Invalid observation state")
        if item["status"] != "completed" and item["resolved"] is not None:
            raise ValueError("Incomplete evaluations have no resolved outcome")
        if item["status"] == "completed" and (item["run_id"] is None or item["resolved"] is None):
            raise ValueError("Completed evaluations need a receipt and official outcome")
        if item["status"] == "not_run" and (
            item["run_id"] is not None or item["duration_seconds"] is not None
        ):
            raise ValueError("Unexecuted tasks have no run or measured duration")
    if row["status"] == "evaluated" and (
        seen != {(t, v) for t in row["task_ids"] for v in row["candidate_hashes"]}
        or any(r["status"] != "completed" for r in row["observations"])
    ):
        raise ValueError("Evaluated record requires complete observations")
    metrics = row["metrics"]
    if {"resolved", "total"} <= set(metrics):
        if (
            len(row["candidate_hashes"]) != 1
            or metrics["total"] != len(row["observations"])
            or metrics["resolved"] != sum(item["resolved"] is True for item in row["observations"])
        ):
            raise ValueError("Public outcome totals disagree with task observations")
    if set(metrics) & PAIR_COUNTS:
        variants = list(row["candidate_hashes"])
        if (
            len(variants) != VARIANTS_PER_PAIR
            or not PAIR_COUNTS <= set(metrics)
            or seen != {(t, v) for t in row["task_ids"] for v in variants}
        ):
            raise ValueError("Incomplete public pair aggregates")
        observed = {(item["task_id"], item["variant"]): item for item in row["observations"]}
        counts = dict.fromkeys(PAIR_COUNTS, 0)
        for task in row["task_ids"]:
            left, right = [observed[task, variant] for variant in variants]
            if all(item["status"] == "not_run" for item in (left, right)):
                key = "not_run"
            elif any(item["status"] != "completed" for item in (left, right)):
                key = "incomplete"
            elif any(item["termination"] not in SCORABLE_TERMINATIONS for item in (left, right)):
                key = "errors"
            else:
                key = (
                    "both"
                    if left["resolved"] and right["resolved"]
                    else "baseline_only"
                    if left["resolved"]
                    else "candidate_only"
                    if right["resolved"]
                    else "neither"
                )
            counts[key] += 1
        if any(metrics[key] != value for key, value in counts.items()):
            raise ValueError("Public pair aggregates disagree with task observations")
