"""Submit registered candidates once, preserving uncertainty across restarts."""

import copy
import math
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.evidence import Candidate, archive_files, content_id
from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.events import event, logged
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.submission.config import SubmissionSettings, timestamp
from gemma_agent_dev.submission.state import SubmissionAttempt, SubmissionRecord, SubmissionState
from gemma_agent_dev.submission.store import StateStore


def same_day(value: str, now: datetime) -> bool:
    return timestamp(value).astimezone(UTC).date() == now.astimezone(UTC).date()


def description(attempt: SubmissionAttempt) -> str:
    origin = "manual" if attempt.get("mode") == "manual" else "auto"
    return f"gemma-{origin}:{attempt['token']}:{attempt['candidate']}"


@logged("submission.sync")
def synchronize(
    state: SubmissionState, gateway: KaggleGateway, settings: SubmissionSettings, now: datetime
) -> dict[str, int]:
    history = gateway.call("history", page_size=settings.page_size, max_pages=settings.max_pages)
    limits = gateway.call("limits")
    if any(type(limits.get(k)) is not int or limits[k] < 0 for k in ("remaining", "today")):
        raise ValueError("Submission limits are unknown")
    previous = {row["id"]: row for row in state["submissions"]}
    rows: list[SubmissionRecord] = []
    seen: set[int] = set()
    for remote in history:
        ref = remote["id"]
        if type(ref) is not int or ref <= 0 or ref in seen:
            raise ValueError("Duplicate or invalid submission receipt")
        seen.add(ref)
        timestamp(remote["date"])
        score = remote["score"]
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("Invalid public score")
        attempt = next(
            (
                a
                for a in state["attempts"]
                if a["submission_id"] == ref or remote["description"] == description(a)
            ),
            None,
        )
        if ref in previous:
            identity = previous[ref]["content_id"]
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "submission.zip"
                gateway.call(
                    "download", id=ref, destination=str(path), max_bytes=settings.max_archive_bytes
                )
                identity = content_id(archive_files(path, settings.max_archive_bytes))
        if attempt:
            if attempt["candidate"] != identity:
                raise ValueError("Submission receipt disagrees with registered candidate")
            attempt["status"], attempt["submission_id"] = "accepted", ref
        rows.append(
            {
                "id": ref,
                "date": remote["date"],
                "status": remote["status"],
                "score": score,
                "content_id": identity,
                "manual": attempt is None or attempt.get("mode") == "manual",
            }
        )
    if not set(previous) <= seen:
        raise ValueError("Submission history is incomplete")
    state["submissions"], state["last_sync"] = rows, now.isoformat()
    return dict(limits)


@logged("submission.decide")
def decide(
    state: SubmissionState,
    settings: SubmissionSettings,
    limits: dict[str, int],
    now: datetime,
    *,
    candidate_id: str | None = None,
) -> tuple[Candidate | None, str]:
    if not settings.send_window(now, manual=candidate_id is not None):
        return None, "outside_window"
    if any(a["status"] == "creating" for a in state["attempts"]):
        return None, "acceptance_unknown"
    today = sum(same_day(row["date"], now) for row in state["submissions"])
    accepted_today = sum(
        a["status"] == "accepted" and same_day(a["started_at"], now) for a in state["attempts"]
    )
    if (
        limits["remaining"] <= 0
        or max(today, accepted_today, limits["today"]) >= settings.daily_limit
    ):
        return None, "daily_limit"
    if candidate_id is None and any(
        row["manual"] and now < timestamp(row["date"]) + timedelta(hours=settings.manual_hold_hours)
        for row in state["submissions"]
    ):
        return None, "manual_hold"
    submitted = {row["content_id"] for row in state["submissions"]} | {
        a["candidate"] for a in state["attempts"] if a["status"] == "accepted"
    }
    candidate = next(
        (
            c
            for c in state["candidates"]
            if c["id"] not in submitted and (candidate_id is None or c["id"] == candidate_id)
        ),
        None,
    )
    return candidate, "ready" if candidate else "no_candidate"


def top_submissions(state: SubmissionState, count: int) -> list[SubmissionRecord]:
    complete = [
        s for s in state["submissions"] if s["status"] == "COMPLETE" and s["score"] is not None
    ]
    return sorted(
        complete,
        key=lambda s: (float(s["score"] or "0"), timestamp(s["date"]), s["id"]),
        reverse=True,
    )[:count]


def update_incidents(state: SubmissionState, active: set[str]) -> bool:
    added = active - set(state["incidents"])
    state["incidents"] = sorted(active)
    return bool(added)


@logged("submission.run")
def execute_run(
    state: SubmissionState,
    store: StateStore,
    gateway: KaggleGateway,
    settings: SubmissionSettings,
    *,
    mode: str,
    execute: bool,
    clock: Callable[[], datetime],
    candidate_id: str | None = None,
) -> dict[str, Any]:
    if (mode == "manual") != (candidate_id is not None):
        raise ValueError("Manual submission requires an explicit candidate")
    if candidate_id is not None and not any(c["id"] == candidate_id for c in state["candidates"]):
        raise ValueError("Candidate is not registered")
    state = copy.deepcopy(state)
    now = clock()
    expired = now >= timestamp(settings.monitoring_end)
    after_deadline = now >= timestamp(settings.send_deadline)
    # Backup needs no API call if this UTC day's receipt is already durable.
    accepted_today = any(
        a["status"] == "accepted" and same_day(a["started_at"], now) for a in state["attempts"]
    )
    unresolved = any(a["status"] == "creating" for a in state["attempts"])
    if mode == "backup" and accepted_today and not expired and not unresolved:
        return result(state, settings, "already_accepted", False, False)
    if after_deadline and mode != "monitor" and not expired:
        return result(state, settings, "after_deadline", False, False)
    active = set(state["incidents"])
    stage = "sync_failed"
    try:
        limits = synchronize(state, gateway, settings, now)
        active.discard(stage)
        candidate, reason = decide(state, settings, limits, clock(), candidate_id=candidate_id)
        if mode == "monitor" or expired:
            candidate, reason = None, "monitoring_finished" if expired else "monitor_only"
        if candidate and execute:
            stage = f"candidate_failed:{candidate['id']}"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "submission.zip"
                gateway.call(
                    "kernel",
                    source=candidate["source"],
                    destination=str(path),
                    max_bytes=settings.max_archive_bytes,
                )
                if (
                    file_hash(path) != candidate["archive_sha256"]
                    or archive_files(path, settings.max_archive_bytes) != candidate["files"]
                ):
                    raise ValueError("Archived submission checksum changed")
                # Upload is separate from quota-consuming CreateSubmission.
                if not settings.send_window(clock(), manual=mode == "manual"):
                    reason = "outside_window"
                else:
                    uploaded = gateway.call("upload", execute=True, path=str(path))
                    active.discard(stage)
                    stage = "sync_failed"
                    limits = synchronize(state, gateway, settings, clock())
                    active.discard(stage)
                    current, reason = decide(
                        state, settings, limits, clock(), candidate_id=candidate_id
                    )
                    if current and current["id"] == candidate["id"]:
                        now = clock()
                        attempt: SubmissionAttempt = {
                            "candidate": candidate["id"],
                            "token": uuid.uuid4().hex,
                            "started_at": now.isoformat(),
                            "status": "creating",
                            "submission_id": None,
                        }
                        if mode == "manual":
                            attempt["mode"] = "manual"
                        stage = "acceptance_unknown"
                        state["attempts"].append(attempt)
                        store.save(state)  # A failed CAS prevents the remote call.
                        if not settings.send_window(clock(), manual=mode == "manual"):
                            attempt["status"], reason = "not_accepted", "outside_window"
                        else:
                            receipt = gateway.call(
                                "create",
                                execute=True,
                                token=uploaded["token"],
                                description=description(attempt),
                                before=settings.create_before(
                                    now, manual=mode == "manual"
                                ).isoformat(),
                                utc_day=now.astimezone(UTC).date().isoformat(),
                            )
                            if receipt.get("expired"):
                                attempt["status"], reason = "not_accepted", "outside_window"
                            else:
                                ref = receipt.get("id")
                                if type(ref) is not int or ref <= 0:
                                    raise ValueError("Submission receipt is unknown")
                                attempt["status"], attempt["submission_id"] = "accepted", ref
                                reason = "accepted"
    except (ValueError, RuntimeError, OSError) as error:
        # Keep SDK text out of public state. The operation name is enough for triage.
        reason = (
            "acceptance_unknown"
            if any(a["status"] == "creating" for a in state["attempts"])
            else "operation_failed"
        )
        active.add(stage)
        if not execute:
            raise RuntimeError("Read-only submission verification failed") from error
    submitted = {s["content_id"] for s in state["submissions"]}
    active.difference_update(f"candidate_failed:{identity}" for identity in submitted)
    if any(a["status"] == "creating" for a in state["attempts"]):
        active.add("acceptance_unknown")
        reason = "acceptance_unknown"
    else:
        active.discard("acceptance_unknown")
    notify = update_incidents(state, active)
    if execute:
        store.save(state)  # Failure here stays a real workflow failure.
    return result(state, settings, reason, notify, expired)


def result(
    state: SubmissionState, settings: SubmissionSettings, reason: str, notify: bool, stop: bool
) -> dict[str, Any]:
    event("submission.decision", reason_code=reason, count=len(state["attempts"]))
    return {
        "reason": reason,
        "notify": notify,
        "stop": stop,
        "top": [
            {"id": s["id"], "score": s["score"]}
            for s in top_submissions(state, settings.final_count)
        ],
        "attempts": [
            {
                "candidate": a["candidate"],
                "status": a["status"],
                "submission_id": a["submission_id"],
            }
            for a in state["attempts"]
        ],
        "last_sync": state["last_sync"],
        "incidents": state["incidents"],
        "final_selection": "unknown",
    }


def resolve_attempt(state: SubmissionState, token: str, submission_id: int | None) -> None:
    matches = [a for a in state["attempts"] if a["token"] == token and a["status"] == "creating"]
    if len(matches) != 1:
        raise ValueError("Expected one unresolved attempt")
    attempt = matches[0]
    if submission_id is not None:
        row = next((s for s in state["submissions"] if s["id"] == submission_id), None)
        if (
            row is None
            or row["content_id"] != attempt["candidate"]
            or timestamp(row["date"]) < timestamp(attempt["started_at"])
        ):
            raise ValueError("Receipt does not match the unresolved candidate")
        row["manual"] = attempt.get("mode") == "manual"
    attempt["status"] = "accepted" if submission_id is not None else "not_accepted"
    attempt["submission_id"] = submission_id
    state["incidents"] = [item for item in state["incidents"] if item != "acceptance_unknown"]
