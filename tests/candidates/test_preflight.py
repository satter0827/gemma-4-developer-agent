import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.candidates import preflight
from gemma_agent_dev.candidates.preflight import PreflightSettings, load_report
from gemma_agent_dev.common.artifacts import file_hash


def test_preflight_freezes_current_agent_with_only_saved_adapters(
    preflight_project: Path,
    preflight_report: Path,
) -> None:
    settings, report = load_report(preflight_project, preflight_report)
    assert report["status"] == "completed"
    assert report["candidate_sha256"] == file_hash(preflight_report.parent / "submission.zip")
    assert report["files"]["agent.yaml"] == file_hash(settings.agent_dir / "agent.yaml")
    assert not any(name.endswith(".patch") for name in report["files"])
    assert len(json.loads(Path(report["diagnostics"]["path"]).read_text())["cases"]) == 4
    assert preflight.artifact_limit(preflight_project) > 0


@pytest.mark.parametrize(
    "damage", ["input", "archive", "log", "status", "kind", "diagnostic", "case", "patch"]
)
def test_stale_and_tampered_evidence_is_rejected(
    preflight_project: Path,
    preflight_report: Path,
    damage: str,
) -> None:
    report = json.loads(preflight_report.read_text())
    diagnostic = Path(report["diagnostics"]["path"])
    if damage == "input":
        (preflight_project / "agent/prompts/system.md").write_text("changed")
    elif damage == "archive":
        (preflight_report.parent / "submission.zip").write_bytes(b"changed")
    elif damage == "log":
        (preflight_report.parent / "quality.log").unlink()
    elif damage in {"status", "kind"}:
        report[damage] = "invalid"
        preflight_report.write_text(json.dumps(report))
    elif damage == "diagnostic":
        diagnostic.write_text("{}")
    else:
        path = (
            diagnostic.parent
            / "fixture_1/reference"
            / ("case.json" if damage == "case" else "candidate.patch")
        )
        path.write_text("changed")
    with pytest.raises((ValueError, KeyError)):
        load_report(preflight_project, preflight_report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("output_dir", "artifacts/results"),
        ("quality_timeout_seconds", 0),
        ("max_unpacked_bytes", True),
        ("adapter_sha256", "bad"),
        ("agent_dir", "missing"),
        ("output_dir", "artifacts/preflight/agent"),
    ],
)
def test_preflight_rejects_invalid_settings(preflight_project: Path, key: str, value: Any) -> None:
    path = preflight_project / "preflight.yaml"
    data = yaml.safe_load(path.read_text())
    data[key] = value
    if key == "output_dir" and value.endswith("/agent"):
        data["agent_dir"] = "artifacts/preflight"
        (preflight_project / data["agent_dir"]).mkdir(parents=True)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        PreflightSettings.load(preflight_project, path, "smoke")


def test_preflight_requires_matching_diagnostic_runtime(preflight_project: Path) -> None:
    data = json.loads((preflight_project / "runtime.json").read_text())
    data["task_ids"] = ["other"]
    (preflight_project / "other.json").write_text(json.dumps(data))
    path = preflight_project / "preflight.yaml"
    config = yaml.safe_load(path.read_text())
    config["runtime_config"] = "other.json"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="runtime settings differ"):
        PreflightSettings.load(preflight_project, path, "smoke")


def test_stale_installation_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "src/gemma_agent_dev").mkdir(parents=True)
    with pytest.raises(ValueError, match="stale"):
        preflight.check_installation(tmp_path)


def test_failed_quality_does_not_create_candidate(
    preflight_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        assert kwargs["env"]["UV_OFFLINE"] == "1"
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(preflight, "process_group", fail)
    with pytest.raises(subprocess.CalledProcessError):
        preflight.run(preflight_project, Path("preflight.yaml"), "smoke")
    report = next((preflight_project / "artifacts/preflight").glob("*/report.json"))
    assert json.loads(report.read_text())["status"] == "failed"
    assert not (report.parent / "submission.zip").exists()


def test_quality_timeout_stops_make_and_its_descendants(preflight_project: Path) -> None:
    path = preflight_project / "preflight.yaml"
    config = yaml.safe_load(path.read_text())
    config["quality_timeout_seconds"] = 1
    path.write_text(yaml.safe_dump(config))
    marker = preflight_project / "leaked-quality-child"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
    script = preflight_project / "quality.py"
    script.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "print('quality started', flush=True)\n"
        "time.sleep(3)\n"
    )
    (preflight_project / "Makefile").write_text(
        f"verify:\n\t{shlex.quote(sys.executable)} {shlex.quote(str(script))}\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        preflight.run(preflight_project, Path("preflight.yaml"), "smoke")
    report_path = next((preflight_project / "artifacts/preflight").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and "TimeoutExpired" in report["error"]
    assert "quality started" in (report_path.parent / "quality.log").read_text()
    assert not (report_path.parent / "submission.zip").exists()
    time.sleep(1.5)
    assert not marker.exists()


def test_report_cannot_escape_project(preflight_project: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_report(preflight_project, tmp_path / "outside.json")
    with pytest.raises(ValueError):
        load_report(preflight_project, preflight_project / "report.json")


def test_candidate_does_not_copy_old_agent(preflight_project: Path, preflight_report: Path) -> None:
    import zipfile

    with zipfile.ZipFile(preflight_report.parent / "submission.zip") as archive:
        assert archive.read("agent.yaml") == (preflight_project / "agent/agent.yaml").read_bytes()
        assert (
            archive.read("eval_config.yaml")
            == (preflight_project / "agent/eval_config.yaml").read_bytes()
        )
    copy = preflight_project / "artifacts/preflight/copied"
    shutil.copytree(preflight_report.parent, copy)
    assert load_report(preflight_project, copy / "report.json")[1]["candidate_sha256"]
