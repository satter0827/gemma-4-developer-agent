"""Check complete inventory accounting, unknowns, evidence joins, and reproducible output."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest
import yaml

from gemma_agent_dev import task_metrics, task_report
from gemma_agent_dev.artifacts import file_hash


@pytest.fixture
def analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path
    (root / "docs/analysis").mkdir(parents=True)
    tasks = root / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": f"task_{i}",
                    "repo": "repo",
                    "patch": "patch",
                    "test_patch": "test",
                    "problem_statement": "synthetic",
                }
            )
            for i in range(2)
        )
    )
    monkeypatch.setattr(
        "gemma_agent_dev.task_report.TaskCatalog.load",
        lambda *a: NS(tasks_file=tasks, tasks_sha256=file_hash(tasks), tasks={"task_0": None}),
    )
    monkeypatch.setattr(
        "gemma_agent_dev.task_report.RuntimeSettings.load",
        lambda *a: NS(
            packages={"swegemma": "0.2.11"},
            wheelhouse_hashes={"swegemma-0.2.11-py3-none-any.whl": "a" * 64},
        ),
    )
    qualified = root / "qualified.json"
    qualified.write_text(json.dumps({"task_ids": ["task_0"]}))
    (root / "pointer.json").write_text(
        json.dumps({"report": "qualified.json", "sha256": file_hash(qualified)})
    )
    (root / "diagnostic.yaml").write_text("tasks:\n- instance_id: task_1\n")
    run = root / "run"
    run.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": "run",
        "status": "completed",
        "task_ids": ["task_0"],
        "tasks_sha256": file_hash(tasks),
        "timings": {"model_startup_seconds": 100, "evaluation_seconds": 100, "total_seconds": 220},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(json.dumps({"total_tasks": 1, "resolved": 1}))
    (run / "task_results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task_0",
                "resolved": True,
                "duration_seconds": 100,
                "agent_patch_size": 1,
                "test_exit_code": 0,
            }
        )
    )
    (root / "before.json").write_text('{"used_hours":1,"refresh_at":"date"}')
    (root / "after.json").write_text('{"used_hours":2,"refresh_at":"date"}')
    config = {
        "catalog": "catalog",
        "runtime": "runtime",
        "cpu_environment": "cpu",
        "wheel_cache": "wheels",
        "timeout_seconds": 10,
        "task_count": 2,
        "qualified": "pointer.json",
        "diagnostics": "diagnostic.yaml",
        "scenario_counts": [1, 6],
        "annotations": {},
        "runs": [
            {
                "manifest": "run/manifest.json",
                "quota_before": "before.json",
                "quota_after": "after.json",
            }
        ],
    }
    path = root / "analysis.yaml"
    path.write_text(yaml.safe_dump(config))

    def metrics(args: list[str], **kwargs: Any) -> Any:
        rows = [
            {
                "instance_id": f"task_{i}",
                "repo": "repo",
                "num_files": 1,
                "total_churn": 2,
                "test_files": 1,
                "composite_score": 20,
                "difficulty_tier": "Easy",
            }
            for i in range(2)
        ]
        Path(args[-1]).write_text(json.dumps(rows))
        return NS(returncode=0)

    monkeypatch.setattr("gemma_agent_dev.task_report.subprocess.run", metrics)
    return root, path


def test_report_generation(analysis: tuple[Path, Path]) -> None:
    root, config = analysis
    report = task_report.build_report(root, config)
    assert len(report["tasks"]) == 2 and report["tasks"][1]["annotation"] is None
    assert report["tasks"][0]["observations"][0]["resolved"] is True
    assert not report["tasks"][1]["observations"]
    assert report["costs"][0]["generation_seconds"] is None
    assert report["costs"][0]["scenarios"]["1"]["wall_seconds"] == 220
    output = task_report.generate(root, config, write_docs=True)
    assert (output / "tasks.md").read_text() == (root / "docs/analysis/tasks.md").read_text()
    assert (output / "tasks.md").read_text() == task_report.render(
        task_report.build_report(root, config)
    )
    assert "未評価" in task_report.render(report)
    report["tasks"][0]["annotation"] = {
        "target": "explicit",
        "behavior": "absent",
        "reproduction": "unknown",
        "fix": "unknown",
        "basis": "確認した根拠",
    }
    assert "確認した根拠" in task_report.render(report)
    task_report.generate(root, config, write_docs=False)


@pytest.mark.parametrize(
    "change",
    [
        {"task_count": 3},
        {"annotations": {"unknown": {}}},
        {"scenario_counts": []},
        {"scenario_counts": [0]},
        {"timeout_seconds": 0},
    ],
)
def test_report_rejects_inconsistent_inputs(
    analysis: tuple[Path, Path], change: dict[str, Any]
) -> None:
    root, path = analysis
    config = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**config, **change}))
    with pytest.raises(ValueError):
        task_report.build_report(root, path)


def test_inventory_and_annotations(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "task",
        "repo": "repo",
        "patch": "p",
        "test_patch": "t",
        "problem_statement": "s",
    }
    path.write_text(json.dumps(row))
    assert task_report.read_inventory(path, file_hash(path), 1)["task"] == row
    with pytest.raises(ValueError, match="checksum"):
        task_report.read_inventory(path, "wrong", 1)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="duplicate"):
        task_report.read_inventory(path, file_hash(path), 2)
    valid = {
        "target": "explicit",
        "behavior": "absent",
        "reproduction": "unknown",
        "fix": "unknown",
        "basis": "reviewed",
    }
    assert task_report.annotations({"task": valid}, {"task"})
    for value in ({}, {**valid, "target": "guess"}, {**valid, "basis": ""}):
        with pytest.raises(ValueError):
            task_report.annotations({"task": value}, {"task"})
    for event in ("socket.connect", "subprocess.Popen", "import"):
        with pytest.raises(RuntimeError):
            task_metrics.audit(event, ("torch",))
    task_metrics.audit("open", ())


def test_official_worker_checks_installed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import zipfile
    from types import ModuleType

    wheel = tmp_path / "sdk.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("swegemma/difficulty.py", "verified SDK source")
    installed = tmp_path / "swegemma"
    installed.mkdir()
    module = installed / "difficulty.py"
    module.write_text("verified SDK source")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"instance_id":"task"}')
    monkeypatch.setattr(
        "gemma_agent_dev.task_metrics.importlib.metadata.version", lambda _: "version"
    )
    monkeypatch.setattr(
        "gemma_agent_dev.task_metrics.importlib.metadata.distribution",
        lambda _: NS(locate_file=lambda _: module),
    )
    hooks: list[Any] = []
    monkeypatch.setattr(sys, "addaudithook", hooks.append)
    fake: Any = ModuleType("swegemma.difficulty")
    fake.analyze_task = lambda row: NS(
        to_dict=lambda: {
            "instance_id": row["instance_id"],
            "repo": "repo",
            "num_files": 1,
            "total_churn": 1,
            "test_files": 1,
            "composite_score": 20,
            "difficulty_tier": "Easy",
        }
    )
    monkeypatch.setitem(sys.modules, "swegemma", ModuleType("swegemma"))
    monkeypatch.setitem(sys.modules, "swegemma.difficulty", fake)
    rows = task_metrics.calculate(tasks, wheel, file_hash(wheel), "version")
    assert rows[0]["instance_id"] == "task" and len(hooks) == 1
    with pytest.raises(ValueError, match="changed"):
        task_metrics.calculate(tasks, wheel, "wrong", "version")
    module.write_text("modified SDK source")
    with pytest.raises(ValueError, match="differs"):
        task_metrics.calculate(tasks, wheel, file_hash(wheel), "version")
