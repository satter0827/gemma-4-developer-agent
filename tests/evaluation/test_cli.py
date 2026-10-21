import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from gemma_agent_dev.cli import main


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / "src/gemma_agent_dev").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[tool.gemma_agent_dev.results]\nmax_artifact_bytes=2000000\n"
        '[tool.gemma_agent_dev.logging]\nfile_level="INFO"\nconsole_level="WARNING"\nmax_bytes=10485760\nbackup_count=3\n'
    )
    monkeypatch.chdir(root)
    return root


def import_run(project: Path, path: Path) -> Path:
    destination = project / "artifacts/results" / path.name
    shutil.copytree(path, destination)
    return destination


def test_empty_results_are_explicit(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["results"]) == 0
    assert "評価結果はまだありません" in capsys.readouterr().out
    assert (project / "artifacts/logs").exists()


def test_results_keep_incomplete_runs_separate(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    import_run(project, make_run())
    import_run(project, make_run("partial", completed=1, status="running"))
    assert main(["results"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert next(line for line in lines if line.startswith("run-a\t")).split("\t")[6] == "50.0%"
    assert next(line for line in lines if line.startswith("partial\t")).split("\t")[6] == "—"


@pytest.mark.parametrize("damage", ["manifest", "missing", "duplicate"])
def test_broken_results_are_reported_and_fail(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    directory = import_run(project, make_run())
    if damage == "manifest":
        (directory / "manifest.json").write_text("{broken")
    elif damage == "missing":
        (directory / "task_results.jsonl").unlink()
    else:
        records = directory / "task_results.jsonl"
        records.write_text(records.read_text() + records.read_text().splitlines()[0] + "\n")
    assert main(["results"]) == 1
    assert capsys.readouterr().err


def test_show_task_and_missing_artifact(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = import_run(project, make_run())
    assert main(["show", "run-a"]) == 0
    assert '"run_id": "run-a"' in capsys.readouterr().out
    arguments = ["show", "run-a", "--task", "task_a", "--artifact", "patch"]
    assert main(arguments) == 1
    assert capsys.readouterr().err
    (directory / "patches").mkdir()
    (directory / "patches/task_a.patch").write_text("diff --git a/file b/file\n")
    assert main(arguments) == 0
    assert "diff --git" in capsys.readouterr().out


def test_comparison_explains_different_conditions(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    import_run(project, make_run())
    other = import_run(project, make_run("run-b"))
    assert main(["compare", "run-a", "run-b"]) == 0
    assert "同じ課題と実行条件" in capsys.readouterr().out
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["evaluation"]["max_time_minutes"] = 5
    (other / "manifest.json").write_text(json.dumps(manifest))
    assert main(["compare", "run-a", "run-b"]) == 0
    output = capsys.readouterr().out
    assert "実行上限" in output and "task_a" in output


def test_duplicate_run_ids_are_not_selectable(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = import_run(project, make_run())
    shutil.copytree(directory, directory.with_name("duplicate"))
    assert main(["show", "run-a"]) == 1
    output = capsys.readouterr()
    assert "重複" in output.err
    assert '"run_id"' not in output.out


@pytest.mark.parametrize("name", ["../escape", "nested/name", ".", ""])
def test_prepare_rejects_path_names(project: Path, name: str) -> None:
    with pytest.raises(SystemExit) as error:
        main(["prepare-kaggle", "--name", name])
    assert error.value.code == 2
    assert (project / "artifacts/logs").exists()


def test_artifact_requires_task(project: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["show", "run-a", "--artifact", "trace"])
    assert error.value.code == 2


def test_prepare_uses_artifacts_and_prevents_overwrite(
    preflight_project: Path,
    preflight_report: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(preflight_project)
    args = [
        "prepare-kaggle",
        "--name",
        "baseline-01",
        "--preflight-report",
        str(preflight_report),
        "--verify-only",
    ]
    assert main(args) == 0
    assert (preflight_project / "artifacts/kaggle/baseline-01/source-manifest.json").is_file()
    assert main(args) == 1
    assert capsys.readouterr().err


def test_prepare_rejects_output_symlink(project: Path, tmp_path: Path) -> None:
    (project / "artifacts").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "artifacts/kaggle").symlink_to(outside, target_is_directory=True)
    assert main(["prepare-kaggle", "--name", "baseline"]) == 1
    assert not list(outside.iterdir())


def test_failed_run_is_displayed_without_a_read_error(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    path = import_run(project, make_run(status="failed", completed=0))
    (path / "task_results.jsonl").unlink()
    (path / "summary.json").unlink()
    assert main(["results"]) == 0
    output = capsys.readouterr()
    assert "失敗" in output.out and not output.err


def test_prepare_requires_completed_preflight(project: Path) -> None:
    assert main(["prepare-kaggle", "--name", "missing"]) == 1
    assert main(["prepare-kaggle", "--name", "missing", "--preflight-report", "missing.json"]) == 1


def test_comparison_reports_a_changed_agent_even_with_equal_budgets(
    project: Path, make_run: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    import_run(project, make_run())
    other = import_run(project, make_run("run-b"))
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["submission_files"]["prompts/system.md"] = "changed"
    (other / "manifest.json").write_text(json.dumps(manifest))
    assert main(["compare", "run-a", "run-b"]) == 0
    output = capsys.readouterr().out
    assert "エージェント・提出ファイル" in output and "prompts/system.md" in output
    assert "agent.yaml" not in output
    assert "同じ課題と実行条件" not in output
