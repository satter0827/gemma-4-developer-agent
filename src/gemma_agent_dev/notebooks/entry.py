# The bundle builder appends a call with the source payload and provenance.
import base64
import hashlib
import io
import lzma
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ARTIFACTS_ROOT = Path("/kaggle/working/artifacts")
INPUTS_ROOT = Path("/kaggle/input")


def main(
    source_b64: str | None,
    provenance: dict[str, Any],
    *,
    timeout_seconds: int,
    progress_interval_seconds: int,
    verify_only: bool = False,
    diagnostic: bool = False,
    payload_path: str | None = None,
) -> None:
    start = time.monotonic()
    if (source_b64 is None) == (payload_path is None):
        raise ValueError("Expected exactly one source payload")
    if payload_path is not None:
        source = Path(payload_path).resolve()
        if not source.is_relative_to(INPUTS_ROOT.resolve()):
            raise ValueError("Source payload must belong to the mounted inputs")
        payload = source.read_bytes()
    else:
        assert source_b64 is not None
        payload = lzma.decompress(base64.b64decode(source_b64, validate=True))
    if hashlib.sha256(payload).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Source bundle checksum mismatch")
    temporary = ARTIFACTS_ROOT / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(temporary)
    sys.pycache_prefix = str(ARTIFACTS_ROOT / "cache/python")
    with tempfile.TemporaryDirectory(prefix="gemma-source-", dir=temporary) as directory:
        root = Path(directory)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for entry in archive.infolist():
                if not (root / entry.filename).resolve().is_relative_to(root):
                    raise ValueError("Invalid source bundle path")
            archive.extractall(root)
        sys.path.insert(0, str(root / "src"))
        from gemma_agent_dev.notebooks.worker import run_bounded

        remaining = timeout_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("Source extraction exceeded the notebook deadline")
        run_bounded(
            root,
            provenance,
            verify_only=verify_only,
            diagnostic=diagnostic,
            timeout=remaining,
            progress_interval=progress_interval_seconds,
            storage=ARTIFACTS_ROOT,
            started_at=start,
        )
