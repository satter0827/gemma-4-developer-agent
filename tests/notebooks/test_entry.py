import base64
import hashlib
import io
import lzma
import runpy
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_entry_extracts_and_cleans_sources_inside_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: bool, external: bool
) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "src/gemma_agent_dev/notebooks/entry.py")
    )
    entry = namespace["main"]
    artifacts = tmp_path / "artifacts"
    monkeypatch.setitem(entry.__globals__, "ARTIFACTS_ROOT", artifacts)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def bounded(root: Path, provenance: dict[str, Any], **kwargs: Any) -> None:
        assert isinstance(kwargs["started_at"], float)
        run(root, provenance, verify_only=kwargs["verify_only"])

    monkeypatch.setattr("gemma_agent_dev.notebooks.worker.run_bounded", bounded)
    extracted = []
    module_name = (
        "gemma_agent_dev.diagnostics.remote" if diagnostic else "gemma_agent_dev.evaluation.runtime"
    )
    module = types.ModuleType(module_name)

    def run(root: Path, provenance: dict[str, Any], *, verify_only: bool = False) -> None:
        assert verify_only is not diagnostic
        assert sys.dont_write_bytecode
        extracted.append(root)
        assert root.is_relative_to(artifacts / "tmp")
        assert (root / "src/example.py").read_text() == "pass\n"
        assert Path(sys.pycache_prefix or "").is_relative_to(artifacts)

    module.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, module)
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("src/example.py", "pass\n")
    payload = content.getvalue()
    source = None
    if external:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        source = inputs / "source-bundle.zip"
        source.write_bytes(payload)
        monkeypatch.setitem(entry.__globals__, "INPUTS_ROOT", inputs)
    entry(
        None if external else base64.b64encode(lzma.compress(payload)).decode(),
        {"source_sha256": hashlib.sha256(payload).hexdigest()},
        timeout_seconds=60,
        progress_interval_seconds=1,
        verify_only=not diagnostic,
        diagnostic=diagnostic,
        payload_path=str(source) if source else None,
    )
    assert extracted and not extracted[0].exists()
    assert {path.name for path in tmp_path.iterdir()} == (
        {"artifacts", "inputs"} if external else {"artifacts"}
    )


@pytest.mark.parametrize("damage", ["missing", "duplicate", "outside", "hash"])
def test_entry_rejects_invalid_external_payload_before_execution(
    tmp_path: Path, damage: str
) -> None:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "src/gemma_agent_dev/notebooks/entry.py")
    )
    entry = namespace["main"]
    source = tmp_path / "source-bundle.zip"
    source.write_bytes(b"changed")
    entry.__globals__["INPUTS_ROOT"] = tmp_path
    provenance = {"source_sha256": "0" * 64}
    if damage == "outside":
        entry.__globals__["INPUTS_ROOT"] = tmp_path / "other"
    with pytest.raises(ValueError):
        entry(
            "encoded" if damage == "duplicate" else None,
            provenance,
            payload_path=None if damage == "missing" else str(source),
            timeout_seconds=1,
            progress_interval_seconds=1,
        )
