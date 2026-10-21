import json
import subprocess
import sys
from pathlib import Path

import pytest

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.notebooks.evaluation import prepare as bundle
from gemma_agent_dev.notebooks.worker import run_bounded

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "damage", ["none", "missing", "hash", "status", "controls", "source", "no-evidence", "artifact"]
)
def test_gpu_bundle_requires_matching_immutable_cpu_evidence(
    preflight_project: Path, preflight_report: Path, cpu_evidence: Path, damage: str
) -> None:
    path = cpu_evidence
    manifest = json.loads(path.read_text())
    if damage == "hash":
        manifest["candidate_sha256"] = "bad"
    elif damage == "status":
        manifest["status"] = "failed"
    elif damage == "controls":
        (path.parent / "controls.json").unlink()
    elif damage == "source":
        manifest["source_sha256"] = "bad"
    elif damage == "no-evidence":
        manifest.pop("control_artifacts")
    elif damage == "artifact":
        (path.parent / "origins.json").write_text("changed")
    write_json(path, manifest)
    destination = preflight_project / "gpu"
    if damage == "none":
        assert bundle(preflight_project, destination, report_path=preflight_report, cpu_report=path)
    else:
        with pytest.raises((ValueError, OSError)):
            bundle(
                preflight_project,
                destination,
                report_path=preflight_report,
                cpu_report=None if damage == "missing" else path,
            )
        assert not destination.exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_notebook_watchdog_stops_worker_descendants_and_preserves_failure(
    tmp_path: Path, timeout: bool
) -> None:
    source = tmp_path / "source"
    pkg = source / "src/gemma_agent_dev"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    marker = tmp_path / "leaked"
    child = f"import time;from pathlib import Path;time.sleep(2);Path({str(marker)!r}).touch()"
    (pkg / "notebooks").mkdir(parents=True)
    (pkg / "notebooks/__init__.py").touch()
    (pkg / "notebooks/worker.py").write_text(
        "import subprocess,sys,time,json\nfrom pathlib import Path\n"
        + f"folder=Path({str(tmp_path / 'artifacts/results/current')!r})\n"
        + "folder.mkdir(parents=True)\n"
        + "for name in ('manifest.json','session.json','search.json'):\n"
        + "    value={'status':'running',**json.loads(sys.argv[2])}\n"
        + "    (folder/name).write_text(json.dumps(value))\n"
        + "done=folder/'done';done.mkdir()\n"
        + "(done/'manifest.json').write_text(json.dumps({**value,'status':'completed'}))\n"
        + f"diagnostic=Path({str(tmp_path / 'artifacts/diagnostics/current')!r})\n"
        + "diagnostic.mkdir(parents=True)\n"
        + "(diagnostic/'report.json').write_text(json.dumps(value))\n"
        + f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'
        + ("time.sleep(30)\n" if timeout else 'print("finished",flush=True)\n')
    )
    storage = tmp_path / "artifacts"
    results = storage / "results/run"
    results.mkdir(parents=True)
    for name in ("manifest.json", "session.json", "search.json"):
        write_json(
            results / name, {"status": "running", "source_sha256": "fixture", "job_id": "previous"}
        )
    unchanged = {p: p.read_bytes() for p in results.iterdir()}
    write_json(results / "completed.json", {"status": "completed", "source_sha256": "fixture"})
    (results / "corrupt").mkdir()
    (results / "corrupt/manifest.json").write_text("invalid json")
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            run_bounded(
                source,
                {"source_sha256": "fixture"},
                verify_only=False,
                diagnostic=False,
                timeout=0.2,
                progress_interval=0.05,
                storage=storage,
            )
        assert all(
            json.loads((storage / "results/current" / name).read_text())["phase"] == "watchdog"
            for name in ("manifest.json", "session.json", "search.json")
        )
        assert (
            json.loads((storage / "diagnostics/current/report.json").read_text())["phase"]
            == "watchdog"
        )
    else:
        run_bounded(
            source,
            {"source_sha256": "fixture"},
            verify_only=True,
            diagnostic=False,
            timeout=5,
            progress_interval=0.05,
            storage=storage,
        )
    assert all(p.read_bytes() == value for p, value in unchanged.items())
    assert (
        json.loads((storage / "results/current/done/manifest.json").read_text())["status"]
        == "completed"
    )
    import time

    time.sleep(2.1)
    assert not marker.exists()


@pytest.mark.parametrize("diagnostic", [False, True])
def test_worker_dispatches_only_the_selected_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, diagnostic: bool
) -> None:
    import types

    from gemma_agent_dev.notebooks import worker

    module_name = (
        "gemma_agent_dev.diagnostics.remote" if diagnostic else "gemma_agent_dev.evaluation.runtime"
    )
    module = types.ModuleType(module_name)
    calls = []
    module.run = lambda *args, **kwargs: calls.append((args, kwargs))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), "{}", "1", str(int(diagnostic))])
    worker.main()
    assert calls and (calls[0][1] == {} if diagnostic else calls[0][1] == {"verify_only": True})
