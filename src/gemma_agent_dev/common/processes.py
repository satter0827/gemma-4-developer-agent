"""Keep local validation processes and their descendants in one cleanup scope."""

import os
import signal
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from gemma_agent_dev.common.events import CONTEXT, event


@contextmanager
def process_group(
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
