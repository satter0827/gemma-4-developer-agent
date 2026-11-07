"""Observe the modules actually imported by the selected pytest process."""

import json
import os
import sys
from pathlib import Path
from typing import Any


def pytest_sessionfinish(session: Any) -> None:
    prefixes = json.loads(os.environ["ISSUE_ORIGIN_MODULES"])
    origins = {}
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            origin = getattr(module, "__file__", None)
            if origin:
                origins[name] = str(Path(origin).resolve())
    Path(os.environ["ISSUE_ORIGIN_OUTPUT"]).write_text(
        json.dumps(
            {
                "origins": origins,
                "test_paths": sorted({str(Path(item.path).resolve()) for item in session.items}),
                "test_patterns": session.config.getini("python_files"),
            },
            sort_keys=True,
        )
    )
