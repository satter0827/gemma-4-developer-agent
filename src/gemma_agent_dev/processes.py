"""Keep local validation processes and their descendants in one cleanup scope."""

import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


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
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    ) as process:
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
