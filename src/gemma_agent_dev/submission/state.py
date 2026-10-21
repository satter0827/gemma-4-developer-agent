"""Validate the durable submission journal and registered candidate list."""

import math
import re
from pathlib import Path
from typing import Any, TypedDict, cast

from gemma_agent_dev.candidates.evidence import Candidate, checked_candidate
from gemma_agent_dev.evaluation.results import read_object
from gemma_agent_dev.submission.config import timestamp
from gemma_agent_dev.tasks.catalog import checksum

SCHEMA_VERSION = 1


class SubmissionAttempt(TypedDict):
    candidate: str
    token: str
    started_at: str
    status: str
    submission_id: int | None


class SubmissionRecord(TypedDict):
    id: int
    date: str
    status: str
    score: str | None
    content_id: str
    manual: bool


class SubmissionState(TypedDict):
    schema_version: int
    competition: str
    candidates: list[Candidate]
    attempts: list[SubmissionAttempt]
    submissions: list[SubmissionRecord]
    incidents: list[str]
    last_sync: str | None


def empty_state(competition: str) -> SubmissionState:
    return {
        "schema_version": SCHEMA_VERSION,
        "competition": competition,
        "candidates": [],
        "attempts": [],
        "submissions": [],
        "incidents": [],
        "last_sync": None,
    }


def validate_state(value: Any, competition: str) -> SubmissionState:
    if (
        not isinstance(value, dict)
        or set(value) != set(empty_state(competition))
        or value["schema_version"] != SCHEMA_VERSION
        or value["competition"] != competition
    ):
        raise ValueError("Invalid submission state")
    if any(not isinstance(value[k], list) for k in ("candidates", "attempts", "submissions")):
        raise ValueError("Invalid state collections")
    if (
        not isinstance(value["incidents"], list)
        or any(not isinstance(item, str) or not item for item in value["incidents"])
        or len(value["incidents"]) != len(set(value["incidents"]))
    ):
        raise ValueError("Invalid incidents")
    if value["last_sync"] is not None:
        timestamp(value["last_sync"])
    ids = []
    for candidate in value["candidates"]:
        ids.append(checked_candidate(candidate)["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate")
    for attempt in value["attempts"]:
        if (
            not isinstance(attempt, dict)
            or set(attempt) != {"candidate", "token", "started_at", "status", "submission_id"}
            or attempt["candidate"] not in ids
            or attempt["status"] not in {"creating", "accepted", "not_accepted"}
            or not isinstance(attempt["token"], str)
            or not re.fullmatch(r"[a-f0-9]+", attempt["token"])
        ):
            raise ValueError("Invalid submission attempt")
        timestamp(attempt["started_at"])
        ref = attempt["submission_id"]
        if (attempt["status"] == "accepted" and (type(ref) is not int or ref <= 0)) or (
            attempt["status"] != "accepted" and ref is not None
        ):
            raise ValueError("Invalid attempt receipt")
    tokens = [a["token"] for a in value["attempts"]]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate attempt token")
    refs = []
    for row in value["submissions"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "date", "status", "score", "content_id", "manual"}
            or type(row["id"]) is not int
            or row["id"] <= 0
            or type(row["manual"]) is not bool
        ):
            raise ValueError("Invalid submission record")
        timestamp(row["date"])
        if not isinstance(row["status"], str) or (
            row["score"] is not None
            and (not isinstance(row["score"], str) or not math.isfinite(float(row["score"])))
        ):
            raise ValueError("Invalid submission score or status")
        checksum(row["content_id"])
        refs.append(row["id"])
    if len(refs) != len(set(refs)):
        raise ValueError("Duplicate submission receipt")
    return cast(SubmissionState, value)


def add_candidate(state: SubmissionState, path: Path) -> None:
    candidate = checked_candidate(read_object(path))
    if any(c["id"] == candidate["id"] for c in state["candidates"]):
        raise ValueError("Candidate is already registered")
    state["candidates"].append(candidate)
