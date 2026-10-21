import ast
import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from gemma_agent_dev.kaggle_bundle import prepare


def source_files(directory: Path) -> dict[str, bytes]:
    module = ast.parse((directory / "run.py").read_text())
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(ast.literal_eval(statement.value.args[0])))
    ) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_cpu_gpu_share_frozen_candidate_and_isolated_runtime_imports(
    preflight_project: Path,
    preflight_report: Path,
    tmp_path: Path,
    cpu_evidence: Path,
) -> None:
    root = preflight_project
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    assert prepare(root, cpu, report_path=preflight_report, verify_only=True) == prepare(
        root, gpu, report_path=preflight_report, cpu_report=cpu_evidence
    )
    files = source_files(cpu)
    assert files == source_files(gpu)
    assert files["candidate.zip"] == (preflight_report.parent / "submission.zip").read_bytes()
    assert not any(
        "diagnostic" in n or n.endswith(("cli.py", "preflight.py", ".patch")) for n in files
    )
    assert not any(n.startswith(("agent/", "artifacts/", "tests/")) for n in files)
    for directory, enabled in ((cpu, False), (gpu, True)):
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        assert metadata["is_private"] and not metadata["enable_internet"]
        assert metadata["enable_gpu"] is enabled
        assert bool(metadata["model_sources"]) is enabled
    extracted = tmp_path / "isolated"
    for name, data in files.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    script = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "import gemma_agent_dev.runtime as m;assert m.__file__.startswith(sys.argv[1]);"
        "assert 'torch' not in sys.modules and 'swegemma' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(extracted / "src")], check=True)
    with pytest.raises(FileExistsError):
        prepare(root, cpu, report_path=preflight_report)


def test_bundle_rejects_changed_preflight(preflight_project: Path, preflight_report: Path) -> None:
    (preflight_report.parent / "submission.zip").write_bytes(b"changed")
    destination = preflight_project / "artifacts/kaggle/bad"
    with pytest.raises(ValueError, match="artifacts changed"):
        prepare(preflight_project, destination, report_path=preflight_report)
    assert not destination.exists()
