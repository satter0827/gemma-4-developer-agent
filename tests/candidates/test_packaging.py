import os
import zipfile
from pathlib import Path

import pytest

from gemma_agent_dev.candidates.packaging import package_submission


def test_archive_is_validated_after_extraction(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "agent.yaml").write_text("instruction: !include prompt.md\n")
    (source / "prompt.md").write_text("Inspect and repair the issue.\n")
    calls = []

    def validate(path: Path) -> None:
        calls.append(path)
        assert (path / "prompt.md").read_text() == "Inspect and repair the issue.\n"

    destination = tmp_path / "submission.zip"
    package_submission(source, destination, validate, validation_scope="structural")
    assert len(calls) == 2 and calls[0] != calls[1]
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"agent.yaml", "prompt.md"}
    with pytest.raises(FileExistsError):
        package_submission(source, destination, validate, validation_scope="structural")


def test_validation_failure_does_not_publish_archive(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "agent.yaml").write_text("instruction: !include missing.md\n")

    def reject(path: Path) -> None:
        raise ValueError("Missing included prompt")

    destination = tmp_path / "submission.zip"
    with pytest.raises(ValueError, match="Missing included"):
        package_submission(source, destination, reject, validation_scope="structural")
    assert not destination.exists()


def test_symlink_cannot_export_external_file(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "agent.yaml").write_text("name: example\n")
    (source / "secret.txt").symlink_to(tmp_path / "external.txt")
    with pytest.raises(ValueError, match="Symlink"):
        package_submission(
            source, tmp_path / "submission.zip", lambda path: None, validation_scope="structural"
        )


def test_archive_hash_ignores_creation_order_and_modification_times(tmp_path: Path) -> None:
    source = tmp_path / "agent"
    source.mkdir()
    (source / "prompt.md").write_text("Inspect and repair.")
    (source / "agent.yaml").write_text("instruction: !include prompt.md\n")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    package_submission(source, first, lambda path: None, validation_scope="structural")
    for path in source.iterdir():
        os.utime(path, (1700000000, 1700000000))
    package_submission(source, second, lambda path: None, validation_scope="structural")
    assert first.read_bytes() == second.read_bytes()
