"""Verify a frozen experiment with local paths and no model initialization."""

import json
import sys
from pathlib import Path
from typing import Any

from gemma_agent_dev.evaluation.session import run


def deny_external(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect" or (
        event == "import" and str(args[0]).split(".")[0] in {"torch", "torchvision", "vllm"}
    ):
        raise RuntimeError("Network and GPU imports are forbidden during CPU verification")


def main() -> None:
    sys.addaudithook(deny_external)
    print(
        run(
            Path(sys.argv[1]),
            json.loads(sys.argv[2]),
            verify_only=True,
            bindings=json.loads(sys.argv[3]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
