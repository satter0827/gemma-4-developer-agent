import ast
import base64
import io
import json
import lzma
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from gemma_agent_dev.notebooks.evaluation import prepare


def source_files(directory: Path) -> dict[str, bytes]:
    module = ast.parse((directory / "run.py").read_text())
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    with zipfile.ZipFile(
        io.BytesIO(
            lzma.decompress(base64.b64decode(ast.literal_eval(statement.value.args[0])))
            if ast.literal_eval(statement.value.args[0]) is not None
            else (directory / "source-bundle.zip").read_bytes()
        )
    ) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.mark.parametrize("external", [False, True])
def test_cpu_gpu_share_frozen_candidate_and_isolated_runtime_imports(
    preflight_project: Path,
    preflight_report: Path,
    tmp_path: Path,
    cpu_evidence: Path,
    external: bool,
) -> None:
    root = preflight_project
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    assert prepare(
        root,
        cpu,
        report_path=preflight_report,
        verify_only=True,
        bundle_dataset="owner/private-source" if external else None,
    ) == prepare(
        root,
        gpu,
        report_path=preflight_report,
        cpu_report=cpu_evidence,
        bundle_dataset="owner/private-source" if external else None,
    )
    files = source_files(cpu)
    assert files == source_files(gpu)
    assert files["candidate.zip"] == (preflight_report.parent / "submission.zip").read_bytes()
    assert not any(
        n.endswith(
            (
                "diagnostics/remote.py",
                "diagnostics/local.py",
                "diagnostics/config.py",
                "cli.py",
                "preflight.py",
                ".patch",
            )
        )
        for n in files
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
        "import gemma_agent_dev.evaluation.runtime as m;assert m.__file__.startswith(sys.argv[1]);"
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


def test_source_limit_is_checked_before_writing_and_external_assets_keep_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from gemma_agent_dev.common.archives import zip_content
    from gemma_agent_dev.evaluation.config import RuntimeSettings
    from gemma_agent_dev.notebooks import bundle

    root = Path(__file__).resolve().parents[2]
    content = zip_content({"src/gemma_agent_dev/notebooks/entry.py": b"# entry\n"})
    runtime = RuntimeSettings.load(root / "configs/evaluation/runtime.yaml").as_dict()
    provenance = {"source_sha256": hashlib.sha256(content).hexdigest()}
    bundle.write_notebook(
        tmp_path / "inline",
        content,
        provenance,
        runtime,
        runtime["verification_kernel_id"],
        verify_only=True,
        diagnostic=False,
    )
    size = (tmp_path / "inline/run.py").stat().st_size
    monkeypatch.setattr(bundle, "KAGGLE_SOURCE_MAX_BYTES", size)
    with pytest.raises(ValueError, match="upload limit"):
        bundle.write_notebook(
            tmp_path / "blocked",
            content,
            provenance,
            runtime,
            runtime["verification_kernel_id"],
            verify_only=True,
            diagnostic=False,
        )
    assert not (tmp_path / "blocked").exists()
    monkeypatch.setattr(bundle, "KAGGLE_SOURCE_MAX_BYTES", size + 1)
    bundle.write_notebook(
        tmp_path / "boundary",
        content,
        provenance,
        runtime,
        runtime["verification_kernel_id"],
        verify_only=True,
        diagnostic=False,
    )
    monkeypatch.setattr(bundle, "KAGGLE_SOURCE_MAX_BYTES", 1000000)
    assert (
        bundle.write_notebook(
            tmp_path / "external",
            content,
            provenance,
            runtime,
            runtime["verification_kernel_id"],
            verify_only=True,
            diagnostic=False,
            bundle_dataset="owner/private-source",
        )
        == provenance["source_sha256"]
    )
    script = (tmp_path / "external/run.py").read_text()
    assert (
        "main(None" in script
        and "/kaggle/input/datasets/owner/private-source/source-bundle.bin" in script
    )
    assert (
        json.loads((tmp_path / "external/kernel-metadata.json").read_text())["dataset_sources"][-1]
        == "owner/private-source"
    )
    with pytest.raises(ValueError, match="owner/dataset"):
        bundle.write_notebook(
            tmp_path / "bad",
            content,
            provenance,
            runtime,
            runtime["verification_kernel_id"],
            verify_only=True,
            diagnostic=False,
            bundle_dataset="../outside",
        )
