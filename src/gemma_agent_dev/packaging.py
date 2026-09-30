"""Package only an explicitly prepared submission and verify its round trip."""

import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from gemma_agent_dev.artifacts import file_hash, inventory, write_json


def package_submission(
    source: Path,
    destination: Path,
    validate: Callable[[Path], None],
) -> dict[str, str]:
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Archive must be outside the submission directory")
    if destination.exists():
        raise FileExistsError(destination)
    if not (source / "agent.yaml").is_file():
        raise ValueError("agent.yaml must be at the submission root")
    before = inventory(source)
    validate(source)
    temporary = destination.with_suffix(".zip.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in before:
                archive.write(source / relative, relative)
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory)
            with zipfile.ZipFile(temporary) as archive:
                archive.extractall(extracted)
            if inventory(extracted) != before or inventory(source) != before:
                raise ValueError("Submission changed during packaging")
            validate(extracted)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(
        destination.with_suffix(".manifest.json"),
        {
            "files": before,
            "archive_sha256": file_hash(destination),
            "validated": True,
        },
    )
    return before
