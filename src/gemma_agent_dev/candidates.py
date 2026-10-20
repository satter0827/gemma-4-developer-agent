"""Validate and extract immutable submission inputs without inference dependencies."""

import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from gemma_agent_dev.artifacts import inventory

ZIP_MODE_SHIFT = 16

ROOT_CONFIGS = frozenset({"agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"})
ALLOWED_SUFFIXES = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})


def extract_archive(source: Path, destination: Path, max_bytes: int) -> None:
    with zipfile.ZipFile(source) as archive:
        names: set[str] = set()
        total = 0
        for entry in archive.infolist():
            name = entry.filename.rstrip("/")
            path = PurePosixPath(name)
            if (
                not name
                or name in names
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or str(path) != name
                or stat.S_ISLNK(entry.external_attr >> ZIP_MODE_SHIFT)
                or not (destination / name).resolve().is_relative_to(destination.resolve())
            ):
                raise ValueError(f"Invalid archive member: {entry.filename}")
            names.add(name)
            total += entry.file_size
            if total > max_bytes:
                raise ValueError("Archive exceeds the configured unpacked size")
        if not names:
            raise ValueError("Empty submission archive")
        archive.extractall(destination)


def validate_references(root: Path) -> dict[str, str]:
    files = inventory(root)
    if set(files) & ROOT_CONFIGS != {"agent.yaml"}:
        raise ValueError("Exactly agent.yaml must exist at the submission root")
    if any(
        Path(name).suffix not in ALLOWED_SUFFIXES
        or any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
        or (Path(name).suffix == ".py" and not name.startswith("skills/"))
        for name in files
    ):
        raise ValueError("Unexpected file in submission")
    active: set[Path] = set()
    complete: set[Path] = set()

    def load(path: Path) -> Any:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or external submission reference: {path.name}")
        if path in active:
            raise ValueError(f"Cyclic submission reference: {path.name}")
        active.add(path)
        try:

            class Loader(yaml.SafeLoader):
                pass

            def include(loader: Loader, node: yaml.ScalarNode) -> Any:
                referenced = path.parent / loader.construct_scalar(node)
                if referenced.suffix in {".yaml", ".yml"}:
                    return load(referenced)
                referenced = referenced.resolve()
                if not referenced.is_relative_to(root.resolve()) or not referenced.is_file():
                    raise ValueError("Missing or external include")
                return referenced.read_text(encoding="utf-8")

            Loader.add_constructor("!include", include)
            value = yaml.load(path.read_text(), Loader=Loader)

            def visit(item: Any) -> None:
                if isinstance(item, dict):
                    for key, child in item.items():
                        if key == "config_path":
                            if not isinstance(child, str):
                                raise ValueError("Expected agent config path")
                            load(path.parent / child)
                        else:
                            visit(child)
                elif isinstance(item, list):
                    for child in item:
                        visit(child)

            visit(value)
            complete.add(path)
            return value
        finally:
            active.remove(path)

    for name in files:
        path = root / name
        if path.suffix in {".yaml", ".yml"} and path.resolve() not in complete:
            load(path)
    return files
