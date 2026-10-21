from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.tasks.catalog import TaskCatalog

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_roots", ["../outside"]),
        ("import_modules", ["bad/module"]),
        ("source_roots", []),
        ("test_paths", ["bad.txt"]),
        ("snapshot_sha256", "invalid"),
        ("instance_id", "../bad"),
        ("sdk_dependency_wheels", []),
        ("sdk_dependency_wheels", {"../outside.whl": "0" * 64}),
        ("sdk_dependency_wheels", {"helper-1-py3-none-any.whl": "invalid"}),
    ],
)
def test_shared_catalog_rejects_bad_contracts(
    diagnostic_project: DiagnosticSettings, key: str, value: Any
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    data["tasks"][0][key] = value
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        TaskCatalog.load(settings.root, settings.catalog_path)


@pytest.mark.parametrize("damage", ["duplicate", "test_paths", "tasks_data", "missing_data"])
def test_catalog_detects_duplicates_drift_and_missing_tasks(
    diagnostic_project: DiagnosticSettings, damage: str
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    if damage == "duplicate":
        data["tasks"] *= 2
    elif damage == "test_paths":
        data["tasks"][0]["test_paths"] = ["tests/another.py"]
    elif damage == "tasks_data":
        settings.tasks_file.write_text("{}\n")
    else:
        data["tasks"][0]["instance_id"] = "missing"
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        TaskCatalog.load(settings.root, settings.catalog_path)


def test_catalog_cannot_pin_two_hashes_for_the_same_wheel(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    first = data["tasks"][0]
    first["dependency_wheels"] = {"helper-1-py3-none-any.whl": "0" * 64}
    data["tasks"].append(
        {
            **first,
            "instance_id": "fixture_2",
            "dependency_wheels": {"helper-1-py3-none-any.whl": "1" * 64},
        }
    )
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Conflicting"):
        TaskCatalog.load(settings.root, settings.catalog_path)


def test_sdk_wheels_are_hashed_and_do_not_change_native_preparation(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    before = settings.preparation_fingerprint()
    data = yaml.safe_load(settings.catalog_path.read_text())
    wheel = settings.root / "artifacts/reference/task-wheels/helper-1-linux.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    wheel.write_bytes(b"sdk-only-platform-wheel")
    data["tasks"][0]["sdk_dependency_wheels"] = {wheel.name: file_hash(wheel)}
    settings.catalog_path.write_text(yaml.safe_dump(data))
    updated = DiagnosticSettings.load(settings.root, settings.config_path)
    assert updated.preparation_fingerprint() == before
    task = updated.tasks[0]
    assert task.dependency_wheels == {}
    assert task.sdk_source()["dependency_wheels"] == {wheel.name: file_hash(wheel)}
    catalog = TaskCatalog.load(settings.root, settings.catalog_path)
    assert catalog.wheels([task.instance_id]) == {wheel: file_hash(wheel)}
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        catalog.wheels([task.instance_id])


def test_shared_and_sdk_wheels_cannot_disagree(diagnostic_project: DiagnosticSettings) -> None:
    settings = diagnostic_project
    data = yaml.safe_load(settings.catalog_path.read_text())
    task = data["tasks"][0]
    task["dependency_wheels"] = {"helper-1-py3-none-any.whl": "0" * 64}
    task["sdk_dependency_wheels"] = {"helper-1-py3-none-any.whl": "1" * 64}
    settings.catalog_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Conflicting"):
        TaskCatalog.load(settings.root, settings.catalog_path)
