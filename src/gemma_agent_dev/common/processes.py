"""Keep local validation processes and their descendants in one cleanup scope."""

import os
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from gemma_agent_dev.common.events import CONTEXT, event
from gemma_agent_dev.common.ownership import stop_owned


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Interrupt synchronous and asynchronous preparation under one wall deadline."""
    if seconds <= 0:
        raise TimeoutError("Execution deadline exhausted")
    previous = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    timer = signal.getitimer(signal.ITIMER_REAL)

    def expired(signum: int, frame: object) -> None:
        error = TimeoutError("Execution deadline exhausted")
        registry = os.environ.get("GEMMA_PROCESS_REGISTRY")
        owner = os.environ.get("GEMMA_PROCESS_OWNER")
        if registry and owner:
            try:
                stop_owned(Path(registry), owner)
            except (OSError, ValueError) as cleanup_error:
                error.add_note(f"Cannot stop owned processes: {cleanup_error}")
        raise error

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, timer[0]) if timer[0] else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(
                    timer[0] - (time.monotonic() - started),
                    time.get_clock_info("monotonic").resolution,
                ),
                timer[1],
            )


@contextmanager
def _process_group(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: TextIO,
    stderr: TextIO | int = subprocess.STDOUT,
) -> Iterator[subprocess.Popen[str]]:
    with subprocess.Popen(
        args,
        cwd=cwd,
        env={
            **env,
            **(
                {"GEMMA_EXECUTION_ID": CONTEXT.get()["execution_id"]}
                if "execution_id" in CONTEXT.get()
                else {}
            ),
        },
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    ) as process:
        event("process.started")
        failure: BaseException | None = None
        try:
            yield process
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                if failure is None:
                    raise
                failure.add_note(f"Cannot stop validation process group: {error}")
            finally:
                process.wait()
                event("process.completed", exit_code=process.returncode)


@contextmanager
def process_group(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout: TextIO,
    stderr: TextIO | int = subprocess.STDOUT,
) -> Iterator[subprocess.Popen[str]]:
    if "GEMMA_PROCESS_REGISTRY" in env:
        with _process_group(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr) as process:
            yield process
        return
    with tempfile.TemporaryDirectory(prefix="gemma-processes-") as temporary:
        registry = Path(temporary) / "processes.jsonl"
        owner = uuid.uuid4().hex
        failure: BaseException | None = None
        try:
            with _process_group(
                args,
                cwd=cwd,
                env={**env, "GEMMA_PROCESS_REGISTRY": str(registry), "GEMMA_PROCESS_OWNER": owner},
                stdout=stdout,
                stderr=stderr,
            ) as process:
                yield process
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                stop_owned(registry, owner)
            except (OSError, ValueError) as error:
                if failure is None:
                    raise
                failure.add_note(f"Cannot stop owned subprocesses: {error}")


def command(args: list[str], *, cwd: Path, timeout: int, env: dict[str, str] | None = None) -> str:
    with (
        tempfile.TemporaryFile("w+", encoding="utf-8") as stdout,
        tempfile.TemporaryFile("w+", encoding="utf-8") as stderr,
    ):
        with process_group(
            args,
            cwd=cwd,
            env=dict(os.environ) if env is None else env,
            stdout=stdout,
            stderr=stderr,
        ) as process:
            code = process.wait(timeout=timeout)
        stdout.seek(0)
        output = stdout.read()
        stderr.seek(0)
        errors = stderr.read()
    if code:
        raise RuntimeError(f"Command failed ({code}): {args[0]}\n{output}\n{errors}")
    return output
