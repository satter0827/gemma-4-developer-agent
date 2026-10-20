import base64
import hashlib
import io
import runpy
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("diagnostic", [False, True])
def test_entry_extracts_and_cleans_sources_inside_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: bool
) -> None:
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "kaggle/entry.py"))
    entry = namespace["main"]
    artifacts = tmp_path / "artifacts"
    monkeypatch.setitem(entry.__globals__, "ARTIFACTS_ROOT", artifacts)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "pycache_prefix", sys.pycache_prefix)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    extracted = []
    module_name = "gemma_agent_dev.diagnostic_kaggle" if diagnostic else "gemma_agent_dev.runtime"
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
    entry(
        base64.b64encode(payload).decode(),
        {"source_sha256": hashlib.sha256(payload).hexdigest()},
        verify_only=not diagnostic,
        diagnostic=diagnostic,
    )
    assert extracted and not extracted[0].exists()
    assert {path.name for path in tmp_path.iterdir()} == {"artifacts"}
