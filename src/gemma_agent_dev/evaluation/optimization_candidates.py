"""Freeze each generated instruction while preserving all other seed bytes."""

import hashlib
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gemma_agent_dev.candidates.archive import validate_references
from gemma_agent_dev.candidates.packaging import package_submission
from gemma_agent_dev.common.artifacts import inventory
from gemma_agent_dev.common.events import event

VARIABLE = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")


@dataclass(frozen=True)
class FrozenSearchCandidate:
    candidate_id: str
    directory: Path
    sha256: str
    files: dict[str, str]


def freeze(
    seed: Path, output: Path, prompt_path: str, instruction: str, validate: Callable[[Path], object]
) -> FrozenSearchCandidate:
    if not instruction.strip() or set(VARIABLE.findall(instruction)) != set(
        VARIABLE.findall((seed / prompt_path).read_text())
    ):
        raise ValueError("Generated instructions must preserve embedded variables")
    key = hashlib.sha256(instruction.encode()).hexdigest()
    folder = output / key
    candidate = folder / "submission"
    folder.mkdir(parents=True)
    shutil.copytree(seed, candidate)
    (candidate / prompt_path).write_text(instruction, encoding="utf-8")
    before, after = inventory(seed), inventory(candidate)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if changed - {prompt_path}:
        raise ValueError("Optimization changed a fixed candidate file")
    validate_references(candidate)
    archive = folder / "submission.zip"
    files = package_submission(
        candidate, archive, validate, validation_scope="optimization-sdk-compile"
    )
    from gemma_agent_dev.common.artifacts import file_hash

    event("optimization.candidate.frozen", candidate_sha256=file_hash(archive))
    return FrozenSearchCandidate(key, candidate, file_hash(archive), files)
