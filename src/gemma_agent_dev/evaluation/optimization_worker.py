"""Verify actual GEPA search with synthetic outcomes and no network or GPU imports."""

import json
import sys
from pathlib import Path

from gemma_agent_dev.evaluation.optimization_runtime import run
from gemma_agent_dev.evaluation.session_worker import deny_external


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
