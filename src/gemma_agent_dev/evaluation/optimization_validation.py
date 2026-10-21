"""Compile a generated candidate in the pinned SDK CPU environment."""

import json
import sys
from pathlib import Path

from gemma_agent_dev.evaluation.official import (
    create_server,
    expected_model,
    make_validator,
    stop_model,
)
from gemma_agent_dev.evaluation.session import evaluation_config
from gemma_agent_dev.evaluation.session_worker import deny_external


def main() -> None:
    sys.addaudithook(deny_external)
    root, candidate = Path(sys.argv[1]), Path(sys.argv[2])
    runtime = json.loads((root / "optimization.json").read_text())["definition"]["runtime"]
    server = create_server(candidate, runtime)
    try:
        configuration = evaluation_config(runtime, server, candidate, candidate.parent, [])
        make_validator(configuration, expected_model(runtime))(candidate)
    finally:
        stop_model(server)


if __name__ == "__main__":
    main()
