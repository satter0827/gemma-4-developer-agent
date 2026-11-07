"""Validated submission policy; all scheduling values have explicit units."""

import re
from dataclasses import dataclass, fields
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from gemma_agent_dev.common.config import mapping, positive, project_path


def timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Expected an ISO timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result


@dataclass(frozen=True)
class ReadRetryPolicy:
    attempts: int
    initial_delay_seconds: int

    @classmethod
    def parse(cls, value: Any) -> "ReadRetryPolicy":
        data = mapping(value, {f.name for f in fields(cls)}, "read retry policy")
        for name, setting in data.items():
            positive(setting, name, integer=True)
        return cls(**data)


@dataclass(frozen=True)
class SubmissionSettings:
    competition: str
    state_branch: str
    timezone: str
    morning_start: str
    morning_cutoff: str
    competition_deadline: str
    send_deadline: str
    monitoring_end: str
    manual_hold_hours: int
    daily_limit: int
    final_count: int
    api_timeout_seconds: int
    transfer_timeout_seconds: int
    max_archive_bytes: int
    page_size: int
    max_pages: int
    seed_candidates: list[str]
    read_retry: ReadRetryPolicy | None = None

    @classmethod
    def load(cls, root: Path, path: Path) -> "SubmissionSettings":
        raw = yaml.safe_load(project_path(root, str(path)).read_text())
        names = {f.name for f in fields(cls)} - {"read_retry"}
        if isinstance(raw, dict) and "read_retry" in raw:
            names.add("read_retry")
        data = mapping(raw, names, "submission settings")
        if "read_retry" in data:
            data["read_retry"] = ReadRetryPolicy.parse(data["read_retry"])
        for key in ("competition", "state_branch"):
            if not isinstance(data[key], str) or not re.fullmatch(r"[\w/-]+", data[key]):
                raise ValueError(f"Invalid {key}")
        branch = data["state_branch"]
        if not branch.startswith("codex/") or ".." in branch or branch.endswith("/"):
            raise ValueError("Expected a dedicated codex state branch")
        ZoneInfo(data["timezone"])
        if time.fromisoformat(data["morning_start"]) >= time.fromisoformat(data["morning_cutoff"]):
            raise ValueError("Invalid submission window")
        if timestamp(data["competition_deadline"]) >= timestamp(data["monitoring_end"]):
            raise ValueError("Monitoring must end after the competition deadline")
        if timestamp(data["send_deadline"]) > timestamp(data["competition_deadline"]):
            raise ValueError("Sending must stop before the competition deadline")
        for key in (
            "manual_hold_hours",
            "daily_limit",
            "final_count",
            "api_timeout_seconds",
            "transfer_timeout_seconds",
            "max_archive_bytes",
            "page_size",
            "max_pages",
        ):
            positive(data[key], key, integer=True)
        seeds = data["seed_candidates"]
        if not isinstance(seeds, list) or any(not isinstance(s, str) for s in seeds):
            raise ValueError("Expected seed candidate paths")
        for seed in seeds:
            project_path(root, seed)
        return cls(**data)

    def send_window(self, now: datetime, *, manual: bool = False) -> bool:
        local = now.astimezone(ZoneInfo(self.timezone))
        return now < timestamp(self.send_deadline) and (
            manual
            or time.fromisoformat(self.morning_start)
            <= local.time()
            < time.fromisoformat(self.morning_cutoff)
        )

    def create_before(self, now: datetime, *, manual: bool = False) -> datetime:
        if manual:
            return timestamp(self.send_deadline)
        local = now.astimezone(ZoneInfo(self.timezone))
        return min(
            datetime.combine(local.date(), time.fromisoformat(self.morning_cutoff), local.tzinfo),
            timestamp(self.send_deadline),
        )
