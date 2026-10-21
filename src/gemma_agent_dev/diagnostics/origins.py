"""Check source origins inside the pytest process, including failed collection."""

import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PLUGIN_NAME = "gemma_source_guard"
IMPORT_ERRORS: dict[str, str] = {}


def test_environment(
    workspace: Path, roots: list[str], modules: list[str], directory: Path
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), directory / f"{PLUGIN_NAME}.py")
    return {
        "PYTHONPATH": os.pathsep.join(str(p) for p in [*(workspace / r for r in roots), directory]),
        "PYTEST_PLUGINS": PLUGIN_NAME,
        "GEMMA_SOURCE_WORKSPACE": str(workspace),
        "GEMMA_SOURCE_MODULES": json.dumps(modules),
        "GEMMA_SOURCE_EVIDENCE": str(directory / "origins.json"),
    }


def _origins() -> dict[str, Any]:
    workspace = Path(os.environ["GEMMA_SOURCE_WORKSPACE"]).resolve()
    required = json.loads(os.environ["GEMMA_SOURCE_MODULES"])
    origins = {
        name: str(Path(str(module.__file__)).resolve())
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(prefix + ".") for prefix in required)
        and getattr(module, "__file__", None)
        and getattr(module, "__name__", name) == name
    }
    for name in required:
        if name not in origins:
            spec = importlib.util.find_spec(name)
            if spec is not None and spec.origin is not None:
                origins[name] = str(Path(spec.origin).resolve())
    valid = all(name in origins for name in required) and all(
        Path(path).is_relative_to(workspace) for path in origins.values()
    )
    return {
        "workspace": str(workspace),
        "modules": origins,
        "valid": valid,
        "import_errors": dict(IMPORT_ERRORS),
    }


def pytest_sessionstart(session: Any) -> None:
    for name in json.loads(os.environ["GEMMA_SOURCE_MODULES"]):
        try:
            importlib.import_module(name)
        except ImportError as error:
            # Let pytest report the original collection error and preserve its exit code.
            IMPORT_ERRORS[name] = str(error)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    evidence = _origins()
    Path(os.environ["GEMMA_SOURCE_EVIDENCE"]).write_text(json.dumps(evidence, sort_keys=True))
    if not evidence["valid"]:
        session.exitstatus = 1


def validate_origins(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = json.loads(path.read_text())
    workspace = Path(evidence["workspace"])
    if (
        evidence.get("valid") is not True
        or not evidence["modules"]
        or any(
            not Path(origin).is_relative_to(workspace) for origin in evidence["modules"].values()
        )
    ):
        raise ValueError("Tests did not import the workspace source")
    return evidence
