import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import inventory, write_json
from gemma_agent_dev.diagnostics import qualification
from gemma_agent_dev.diagnostics.config import DiagnosticSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["valid", "reserve", "insufficient", "environment", "wrong-repo"])
def test_qualification_freezes_controls_and_uses_reserves_before_model_results(
    diagnostic_project: DiagnosticSettings, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    settings = diagnostic_project
    config_path = settings.root / "qualification.yaml"
    tasks = [
        replace(
            settings.tasks[0],
            instance_id=f"fixture_{i}",
            generated_patch=None,
            generated_sha256=None,
            probe_replacements=[],
        )
        for i in (1, 2, 3)
    ]
    settings = replace(settings, tasks=tasks)
    records = {t.instance_id: {"repo": "fixture/repo", "patch": "reference"} for t in tasks}
    if mode == "wrong-repo":
        records["fixture_1"]["repo"] = "other/other"
    monkeypatch.setattr(DiagnosticSettings, "load", lambda *args: settings)
    monkeypatch.setattr(DiagnosticSettings, "read_tasks", lambda self: records)
    monkeypatch.setattr(qualification, "check_prepared", lambda s: {})
    config_path.write_text(
        yaml.safe_dump(
            {
                "diagnostic_config": "diagnostics.yaml",
                "fixed_selection": None,
                "per_repository": 1,
                "candidates": {"repo": [t.instance_id for t in tasks]},
            }
        )
    )
    calls = []

    def case(s: Any, task: Any, data: Any, name: str, patch: str, output: Path) -> dict[str, Any]:
        calls.append((task.instance_id, name))
        output.mkdir(parents=True)
        passed = name == "reference"
        if mode == "reserve" and task.instance_id == "fixture_1":
            passed = True
        if mode == "insufficient":
            passed = False
        (output / "junit.xml").write_text(
            "<testsuite><testcase>" + ("" if passed else "<failure/>") + "</testcase></testsuite>"
        )
        result = {
            "exit_code": 0 if passed else 1,
            "task_id": task.instance_id,
            "case": name,
            "passed": passed,
            "source_origins": {"valid": True},
            "artifacts": inventory(output),
        }
        if mode == "environment":
            result["error"] = "collection error"
        write_json(output / "case.json", result)
        return result

    monkeypatch.setattr(qualification, "run_case", case)
    if mode in {"insufficient", "environment", "wrong-repo"}:
        with pytest.raises((ValueError, RuntimeError)):
            qualification.qualify(settings.root, config_path)
        assert not (settings.output_dir / "qualified.json").exists()
    else:
        pointer = qualification.qualify(settings.root, config_path)
        report = qualification.load_selection(settings.root, pointer)
        assert report["task_ids"] == ["fixture_2" if mode == "reserve" else "fixture_1"]
        assert len(calls) == (4 if mode == "reserve" else 2)
        with pytest.raises(FileExistsError):
            qualification.qualify(settings.root, config_path)
        original_pointer = pointer.read_bytes()
        data = yaml.safe_load(config_path.read_text())
        data["fixed_selection"] = str(pointer.relative_to(settings.root))
        config_path.write_text(yaml.safe_dump(data))
        calls.clear()
        current = qualification.qualify(settings.root, config_path, fixed_selection=True)
        assert (
            qualification.load_selection(settings.root, current)["task_ids"] == report["task_ids"]
        )
        assert calls == [(report["task_ids"][0], label) for label in ("reference", "unmodified")]
        assert pointer.read_bytes() == original_pointer
        previous_current = current.read_bytes()
        calls.clear()
        mode = "insufficient"
        with pytest.raises(RuntimeError, match="Not enough"):
            qualification.qualify(settings.root, config_path, fixed_selection=True)
        assert len(calls) == 2 and current.read_bytes() == previous_current
        case_path = Path(json.loads(pointer.read_text())["report"])
        case_folder = settings.root / case_path.parent / report["task_ids"][0] / "reference"
        (case_folder / "case.json").write_text("{}")
        with pytest.raises(ValueError, match="artifacts"):
            qualification.load_selection(settings.root, pointer, current=False)
