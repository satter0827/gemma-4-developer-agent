"""Observe the modules actually imported by the selected pytest process."""

import json
import os
import sys
from pathlib import Path


def pytest_sessionfinish() -> None:
    prefixes = json.loads(os.environ["ISSUE_ORIGIN_MODULES"])
    origins = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            origin = getattr(module, "__file__", None)
            if origin:
                origins[name] = str(Path(origin).resolve())
    Path(os.environ["ISSUE_ORIGIN_OUTPUT"]).write_text(json.dumps(origins, sort_keys=True))
