"""Write bounded JSON Lines events without importing optional runtime dependencies."""

import json
import logging
import math
import os
import re
import sys
import time
import tomllib
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast

from gemma_agent_dev.common.artifacts import new_run_id
from gemma_agent_dev.common.config import mapping, positive

CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar("event_context", default=MappingProxyType({}))
ACTIVE: ContextVar[bool] = ContextVar("events_active", default=False)
FIELDS = frozenset(
    {
        "execution_id",
        "experiment_id",
        "session_id",
        "task_id",
        "variant",
        "candidate_sha256",
        "stage",
        "duration_seconds",
        "resolved",
        "count",
        "reason_code",
        "exit_code",
        "iteration",
        "remaining_seconds",
        "metric_calls",
        "reflection_calls",
        "exception_type",
        "tool_name",
        "result",
    }
)
LABEL = re.compile(r"[A-Za-z0-9_.:/-]+")
LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class LogSettings:
    file_level: str
    console_level: str
    max_bytes: int
    backup_count: int

    @classmethod
    def parse(cls, value: Any) -> "LogSettings":
        data = mapping(value, set(cls.__dataclass_fields__), "logging")
        if any(data[k] not in LEVELS for k in ("file_level", "console_level")):
            raise ValueError("Invalid logging level")
        for key in ("max_bytes", "backup_count"):
            positive(data[key], key, integer=True)
        return cls(**data)

    @classmethod
    def load(cls, root: Path) -> "LogSettings":
        data = tomllib.loads((root / "pyproject.toml").read_text())
        return cls.parse(data["tool"]["gemma_agent_dev"]["logging"])


def checked(fields: dict[str, Any]) -> dict[str, Any]:
    if set(fields) - FIELDS:
        raise ValueError("Unknown event fields")
    for value in fields.values():
        if value is None or type(value) is bool:
            continue
        if type(value) in (int, float) and math.isfinite(value):
            continue
        if isinstance(value, str) and LABEL.fullmatch(value):
            continue
        raise ValueError("Event values must be finite numbers or safe identifiers")
    return fields


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created, UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.__dict__["event_name"],
            "message": record.__dict__["event_name"].replace(".", " "),
            "process_id": record.process,
            **record.__dict__["event_context"],
            **record.__dict__["event_fields"],
        }
        if record.exc_info and record.exc_info[0] is not None:
            # Exception strings can contain HTTP bodies, commands, credentials or task text.
            data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": "Operation failed; inspect the private execution evidence",
                "stacktrace": [
                    {
                        "file": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(record.exc_info[2])
                ],
            }
        return json.dumps(data, ensure_ascii=False, allow_nan=False)


class EventHandler(RotatingFileHandler):
    failure: OSError | None = None

    def handleError(self, record: logging.LogRecord) -> None:
        if self.failure is None:
            self.failure = OSError("Cannot save execution events")
            print(
                "Cannot save execution events; results and cleanup remain active", file=sys.stderr
            )

    def emit(self, record: logging.LogRecord) -> None:
        if self.failure is None:
            super().emit(record)


class ConsoleHandler(logging.StreamHandler[Any]):
    def handleError(self, record: logging.LogRecord) -> None:
        print("Cannot format execution event; raw message omitted", file=sys.stderr)


def event(name: str, *, level: str = "INFO", exception: bool = False, **fields: Any) -> None:
    if not LABEL.fullmatch(name) or level not in LEVELS:
        raise ValueError("Invalid event name or level")
    if not ACTIVE.get():
        return
    logger = logging.getLogger("gemma_agent_dev." + name.partition(".")[0])
    logger.log(
        getattr(logging, level),
        name,
        extra={"event_name": name, "event_fields": checked(fields), "event_context": CONTEXT.get()},
        exc_info=exception,
    )


@contextmanager
def context(**fields: Any) -> Iterator[None]:
    token = CONTEXT.set({**CONTEXT.get(), **checked(fields)})
    try:
        yield
    finally:
        CONTEXT.reset(token)


@contextmanager
def session(directory: Path, settings: LogSettings, execution_id: str) -> Iterator[None]:
    checked({"execution_id": execution_id})
    logger = logging.getLogger("gemma_agent_dev")
    previous = logger.level, logger.propagate, list(logger.handlers)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler = EventHandler(
            directory / f"events-{os.getpid()}.jsonl",
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
    except OSError:
        print("Cannot initialize execution event storage", file=sys.stderr)
        raise
    handler.setLevel(settings.file_level)
    handler.setFormatter(JsonFormatter())
    console = ConsoleHandler(sys.stderr)
    console.setLevel(settings.console_level)
    console.setFormatter(JsonFormatter())
    logger.handlers = [handler, console]
    logger.setLevel(min(handler.level, console.level))
    logger.propagate = False
    failure: BaseException | None = None
    active_token = ACTIVE.set(True)
    try:
        with context(execution_id=execution_id):
            yield
    except BaseException as error:
        failure = error
        raise
    finally:
        ACTIVE.reset(active_token)
        logger.setLevel(previous[0])
        logger.propagate = previous[1]
        logger.handlers = previous[2]
        try:
            handler.close()
        except OSError:
            handler.failure = OSError("Cannot close execution events")
        console.close()
        if handler.failure is not None:
            if failure is None:
                raise handler.failure
            failure.add_note("Execution event storage also failed")


@contextmanager
def phase(name: str, **fields: Any) -> Iterator[None]:
    started = time.monotonic()
    with context(**fields):
        event(name + ".started")
        try:
            yield
        except BaseException as error:
            event(name + ".failed", level="ERROR", exception_type=type(error).__name__)
            raise
        else:
            event(name + ".completed", duration_seconds=time.monotonic() - started)


def logged(name: str) -> Callable[[F], F]:
    def decorate(function: F) -> F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with phase(name):
                return function(*args, **kwargs)

        return cast(F, wrapped)

    return decorate


@contextmanager
def bundle_session(root: Path, output: Path, provenance: dict[str, Any]) -> Iterator[None]:
    """Older archives remain readable; all new bundles carry an explicit logging policy."""
    policy = root / "logging.json"
    if not policy.is_file():
        yield
        return
    settings = LogSettings.parse(json.loads(policy.read_text()))
    execution_id = os.environ.get("GEMMA_EXECUTION_ID") or new_run_id()
    with session(output / "logs/events", settings, execution_id):
        with context(**{k: provenance[k] for k in ("experiment_id",) if k in provenance}):
            yield


def policy(root: Path) -> bytes:
    return json.dumps(asdict(LogSettings.load(root))).encode()


@contextmanager
def local_session(root: Path, output: Path) -> Iterator[None]:
    execution_id = (
        CONTEXT.get().get("execution_id") or os.environ.get("GEMMA_EXECUTION_ID") or new_run_id()
    )
    with session(output / "logs/events", LogSettings.load(root), execution_id):
        yield
