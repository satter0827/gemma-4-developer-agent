"""Record and stop subprocess groups created by one isolated notebook worker."""

import json
import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

OWNER_VARIABLE = "GEMMA_PROCESS_OWNER"
STOP_TIMEOUT_SECONDS = 10
PROC_START_FIELD = 19
ISOLATE_LAUNCH: ContextVar[bool] = ContextVar("isolate_model_launch", default=False)


@contextmanager
def isolated_launch() -> Iterator[None]:
    """Give a model launcher its own group without detaching SDK test commands."""
    token = ISOLATE_LAUNCH.set(True)
    try:
        yield
    finally:
        ISOLATE_LAUNCH.reset(token)


def _identity(pid: int) -> str | None:
    token = ISOLATE_LAUNCH.set(False)
    try:
        if Path("/proc").is_dir():
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[PROC_START_FIELD]
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            check=True,
            timeout=STOP_TIMEOUT_SECONDS,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None

    finally:
        ISOLATE_LAUNCH.reset(token)


def stop_owned(path: Path, owner: str) -> list[int]:
    """Use recorded identities, never a command-name match, to select groups."""
    stopped: list[int] = []
    if not path.exists():
        return stopped
    issues = []
    for line in reversed(path.read_text().splitlines()):
        try:
            record = json.loads(line)
            if (
                set(record) != {"owner", "pid", "identity"}
                or type(record["pid"]) is not int
                or record["pid"] <= 1
            ):
                raise ValueError("Invalid process identity")
        except (ValueError, TypeError):
            issues.append("Incomplete process ownership record")
            continue
        if record["owner"] != owner:
            issues.append("Process ownership record belongs to another job")
            continue
        pid = record["pid"]
        identity = _identity(pid)
        matches = identity is not None and identity == record["identity"]
        if identity is None and Path("/proc").is_dir():
            # A detached group's leader may have exited before its descendants.
            for member in Path("/proc").iterdir():
                if not member.name.isdigit():
                    continue
                try:
                    if os.getpgid(int(member.name)) == pid and (
                        f"{OWNER_VARIABLE}={owner}".encode()
                        in (member / "environ").read_bytes().split(b"\0")
                    ):
                        matches = True
                        break
                except (OSError, ProcessLookupError):
                    continue
        if not matches:
            continue
        try:
            if identity is not None and os.getpgid(pid) != pid:
                continue
            os.killpg(pid, signal.SIGKILL)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    if issues:
        raise ValueError("; ".join(issues))
    return stopped


@contextmanager
def own_subprocesses(path: Path, owner: str) -> Iterator[None]:
    """Instrument the existing dedicated worker; SDK command semantics stay intact."""
    original = subprocess.Popen
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    children: list[subprocess.Popen[Any]] = []
    registration_errors: list[BaseException] = []

    class OwnedPopen(subprocess.Popen[Any]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if ISOLATE_LAUNCH.get() and "start_new_session" not in kwargs:
                kwargs["start_new_session"] = True
            environment = kwargs.get("env")
            kwargs["env"] = {
                **(os.environ if environment is None else environment),
                OWNER_VARIABLE: owner,
            }
            super().__init__(*args, **kwargs)
            try:
                if kwargs.get("start_new_session"):
                    children.append(self)
                    identity = _identity(self.pid)
                    if identity is None and self.poll() is None:
                        raise RuntimeError("Cannot establish subprocess identity")
                    record = json.dumps({"pid": self.pid, "identity": identity, "owner": owner})
                    with path.open("a") as stream:
                        stream.write(record + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
            except BaseException as error:
                registration_errors.append(error)
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.wait()
                raise

    subprocess.Popen = OwnedPopen  # type: ignore[misc]  # Scoped to the isolated worker.
    failure: BaseException | None = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        subprocess.Popen = original  # type: ignore[misc]  # Restore the SDK constructor.
        try:
            # Direct Popen handles remain authoritative even if saving the journal failed.
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                stop_owned(path, owner)
            finally:
                for child in children:
                    child.wait(timeout=STOP_TIMEOUT_SECONDS)
            if registration_errors:
                raise RuntimeError("Cannot persist process ownership") from registration_errors[0]
        except BaseException as cleanup_error:
            if failure is None:
                raise
            failure.add_note(f"Cannot clean owned subprocesses: {cleanup_error}")
