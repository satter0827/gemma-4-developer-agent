"""Verify a frozen experiment with local paths and no model initialization."""

import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.common.ownership import own_subprocesses
from gemma_agent_dev.evaluation.session import run


def deny_external(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect" or (
        event == "import" and str(args[0]).split(".")[0] in {"torch", "torchvision", "vllm"}
    ):
        raise RuntimeError("Network and GPU imports are forbidden during CPU verification")


def main() -> None:
    registry = os.environ.get("GEMMA_PROCESS_REGISTRY")
    if registry is None:
        raise ValueError("CPU verification requires an owned process scope")
    scope = own_subprocesses(Path(registry), os.environ["GEMMA_PROCESS_OWNER"])
    manifest: Path | None = None
    try:
        with scope:
            manifest = execute()
    except BaseException as error:
        if manifest is not None:
            try:
                report = json.loads(manifest.read_text())
                report.update(
                    status="failed",
                    phase="process_cleanup",
                    error=f"{type(error).__name__}: {error}",
                )
                write_json(manifest, report)
            except (OSError, ValueError) as save_error:
                error.add_note(f"Cannot save CPU cleanup failure: {save_error}")
        raise
    report = json.loads(manifest.read_text())
    report["cpu_worker_completed"] = True
    write_json(manifest, report)
    print(
        json.dumps({"kind": "experiment-verification-receipt", "manifest": str(manifest)}),
        flush=True,
    )


def execute() -> Path:
    sys.addaudithook(deny_external)
    with redirect_stdout(sys.stderr):
        return run(
            Path(sys.argv[1]),
            {**json.loads(sys.argv[2]), "cpu_worker_completion_required": True},
            verify_only=True,
            bindings=json.loads(sys.argv[3]),
        )


if __name__ == "__main__":
    main()
