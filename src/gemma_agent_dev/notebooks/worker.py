"""Bound the complete notebook job, including setup and model cleanup."""

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import new_run_id, write_json
from gemma_agent_dev.common.processes import process_group


def run_bounded(
    root: Path,
    provenance: dict[str, Any],
    *,
    verify_only: bool,
    diagnostic: bool,
    timeout: float,
    progress_interval: float,
    storage: Path,
) -> None:
    output = storage / "jobs" / new_run_id()
    output.mkdir(parents=True)
    start = time.monotonic()
    receipt: dict[str, Any] = {
        "status": "running",
        "provenance": provenance,
        "timeout_seconds": timeout,
    }
    write_json(output / "watchdog.json", receipt)
    failure: BaseException | None = None
    try:
        with (output / "worker.log").open("w") as log, (output / "worker.log").open() as stream:
            args = [
                sys.executable,
                "-m",
                "gemma_agent_dev.notebooks.worker",
                str(root),
                json.dumps(provenance),
                str(int(verify_only)),
                str(int(diagnostic)),
            ]
            with process_group(
                args, cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, stdout=log
            ) as process:
                while True:
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(args, timeout)
                    try:
                        code = process.wait(timeout=min(progress_interval, remaining))
                    except subprocess.TimeoutExpired:
                        print(stream.read(), end="", flush=True)
                        continue
                    print(stream.read(), end="", flush=True)
                    if code:
                        raise subprocess.CalledProcessError(code, args)
                    break
        receipt["status"] = "completed"
    except BaseException as error:
        failure = error
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            for path in (storage / "results").glob("*/manifest.json"):
                manifest = json.loads(path.read_text())
                if (
                    manifest.get("status") == "running"
                    and manifest.get("source_sha256") == provenance["source_sha256"]
                ):
                    manifest.update(status="failed", error=receipt["error"], phase="watchdog")
                    write_json(path, manifest)
        except (OSError, ValueError) as save_error:
            error.add_note(f"Cannot save stopped run: {save_error}")
    finally:
        receipt["duration_seconds"] = time.monotonic() - start
        try:
            write_json(output / "watchdog.json", receipt)
        except OSError as save_error:
            if failure is None:
                failure = save_error
            else:
                failure.add_note(f"Cannot save watchdog evidence: {save_error}")
    if failure is not None:
        raise failure


def main() -> None:
    root = Path(sys.argv[1])
    provenance = json.loads(sys.argv[2])
    if bool(int(sys.argv[-1])):
        importlib.import_module("gemma_agent_dev.diagnostics.remote").run(root, provenance)
    else:
        importlib.import_module("gemma_agent_dev.evaluation.runtime").run(
            root, provenance, verify_only=bool(int(sys.argv[-2]))
        )


if __name__ == "__main__":
    main()
