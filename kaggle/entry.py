# The bundle builder appends a call with the source payload and provenance.
import base64
import hashlib
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ARTIFACTS_ROOT = Path("/kaggle/working/artifacts")


def main(source_b64: str, provenance: dict[str, Any]) -> None:
    payload = base64.b64decode(source_b64, validate=True)
    if hashlib.sha256(payload).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Source bundle checksum mismatch")
    temporary = ARTIFACTS_ROOT / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
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
        from gemma_agent_dev.runtime import run

        run(root, provenance)
