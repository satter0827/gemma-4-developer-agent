# The bundle builder appends a call with the source payload and provenance.
import base64
import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def main(source_b64: str, provenance: dict[str, Any]) -> None:
    payload = base64.b64decode(source_b64, validate=True)
    if hashlib.sha256(payload).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Source bundle checksum mismatch")
    root = Path(tempfile.mkdtemp(prefix="gemma-source-"))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            if not (root / entry.filename).resolve().is_relative_to(root):
                raise ValueError("Invalid source bundle path")
        archive.extractall(root)
    sys.path.insert(0, str(root / "src"))
    from gemma_agent_dev.runtime import run

    run(root, provenance)
