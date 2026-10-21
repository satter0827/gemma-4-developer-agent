"""Bind native task controls to a frozen bundle without mixing SDK and task dependencies."""

import json
import shutil
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.common.junit import validate_test_evidence
from gemma_agent_dev.diagnostics.origins import validate_origins

CONTROL_LABELS = ("unmodified", "reference")


def copy_native_controls(path: Path, expected: str, ids: list[str], output: Path) -> None:
    if file_hash(path) != expected:
        raise ValueError("Frozen native control report changed")
    report = json.loads(path.read_text())
    cases = {(c["task_id"], c["case"]): c for c in report["cases"] if c["case"] in CONTROL_LABELS}
    if (
        report["status"] != "completed"
        or len(cases) != len([c for c in report["cases"] if c["case"] in CONTROL_LABELS])
        or set(cases) != {(task, label) for task in ids for label in CONTROL_LABELS}
    ):
        raise ValueError("Frozen native control cases are incomplete or duplicated")
    records = []
    for task in ids:
        outcomes: dict[str, Any] = {}
        for label in CONTROL_LABELS:
            case = cases[(task, label)]
            folder = path.parent / task / label
            files = inventory(folder)
            files.pop("case.json", None)
            if (
                files != case["artifacts"]
                or json.loads((folder / "case.json").read_text()) != case
                or case.get("error")
                or not case["matched"]
                or case["passed"] is not (label == "reference")
            ):
                raise ValueError("Native control evidence changed or failed")
            validate_test_evidence(case["exit_code"], folder / "junit.xml")
            validate_origins(folder / "source-guard/origins.json")
            destination = output / task / label
            destination.mkdir(parents=True)
            for source, name in (
                (folder / "junit.xml", "junit.xml"),
                (folder / "source-guard/origins.json", "origins.json"),
            ):
                shutil.copyfile(source, destination / name)
            outcomes[label] = {
                "resolved": case["passed"],
                "exit_code": case["exit_code"],
                "error": None,
                "duration_seconds": case["duration_seconds"],
            }
        records.append({"instance_id": task, **outcomes})
    write_json(output / "controls.json", records)
