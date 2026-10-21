import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.diagnostics.config import DiagnosticSettings

ROOT = Path(__file__).resolve().parents[2]


def test_diagnostic_order_survives_sorted_mapping_serialization(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch
) -> None:

    from gemma_agent_dev.diagnostics import remote as diagnostic_kaggle
    from gemma_agent_dev.notebooks import environment

    settings = diagnostic_project
    root = settings.root
    runtime = settings.runtime.as_dict()
    runtime["paths"]["output"] = str(root / "artifacts/remote-results")
    ids = ["fixture_z", "fixture_a"]
    spec = {
        "runtime": runtime,
        "task_ids": ids,
        "fingerprint": "fixture",
        "tasks_sha256": settings.tasks_sha256,
        "snapshots": {key: "0" * 64 for key in reversed(ids)},
        "timeout_seconds": settings.timeout_seconds,
        "lock_sha256": "0" * 64,
        "cases": [{"task_id": key, "case": "reference"} for key in ids],
    }
    (root / "diagnostics.json").write_text(json.dumps(spec, sort_keys=True))
    monkeypatch.setattr(environment, "install_wheelhouse", lambda *args: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: settings.runtime.packages[name])
    captured = []

    async def verify(runtime: Any, spec: Any, output: Any, report: Any) -> None:
        captured.append(runtime.task_ids)
        assert [case["task_id"] for case in spec["cases"]] == runtime.task_ids

    monkeypatch.setattr(diagnostic_kaggle, "verify_cases", verify)
    diagnostic_kaggle.run(root, {"source_sha256": "fixture"})
    assert captured == [ids]
