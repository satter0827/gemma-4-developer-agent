"""Validate and extract immutable submission inputs without inference dependencies."""

import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from gemma_agent_dev.common.artifacts import file_hash, inventory

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
    referenced_files(root, root / "agent.yaml")
    return files


def resolve_agent(root: Path, entry: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Resolve only the selected agent's transitive inputs inside its source tree."""
    root = root.resolve()
    active: set[Path] = set()
    complete: set[Path] = set()
    references: set[Path] = set()
    agents: list[dict[str, Any]] = []

    def checked(path: Path) -> Path:
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external submission reference: {path.name}")
        references.add(path.resolve())
        return path.resolve()

    def load(path: Path) -> Any:
        path = checked(path)
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
                referenced = checked(referenced)
                return referenced.read_text(encoding="utf-8")

            Loader.add_constructor("!include", include)
            value = yaml.load(path.read_text(), Loader=Loader)

            def visit(item: Any) -> None:
                if isinstance(item, dict):
                    if "model" in item:
                        agents.append(item)
                    for key, child in item.items():
                        if key == "config_path":
                            if not isinstance(child, str):
                                raise ValueError("Expected agent config path")
                            load(path.parent / child)
                        elif key == "skills":
                            if (
                                not isinstance(child, list)
                                or not child
                                or any(not isinstance(p, str) for p in child)
                            ):
                                raise ValueError("Expected declared skill directories")
                            for name in child:
                                relative = PurePosixPath(name)
                                skill = root / name
                                if (
                                    relative.is_absolute()
                                    or ".." in relative.parts
                                    or "\\" in name
                                    or str(relative) != name
                                    or not name.startswith("skills/")
                                    or skill.is_symlink()
                                    or not skill.is_dir()
                                ):
                                    raise ValueError("Invalid skill directory")
                                checked(skill / "SKILL.md")
                                for asset in skill.rglob("*"):
                                    if asset.is_symlink() or any(
                                        p.startswith(".") or p == "__pycache__"
                                        for p in asset.relative_to(root).parts
                                    ):
                                        raise ValueError("Linked or hidden skill resource")
                                    if asset.is_file():
                                        if (
                                            asset.suffix not in ALLOWED_SUFFIXES
                                            or asset.suffix == ".safetensors"
                                        ):
                                            raise ValueError("Unexpected skill resource")
                                        checked(asset)
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

    load(entry)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(references)}, agents


def referenced_files(root: Path, entry: Path) -> dict[str, str]:
    return resolve_agent(root, entry)[0]


def generation_contract(root: Path, entry: Path | None = None) -> list[dict[str, Any]]:
    """Resolve generation only for LLM nodes reachable from the selected root."""
    import json

    contracts = {}
    for agent in resolve_agent(root, entry or root / "agent.yaml")[1]:
        value = {"model": agent["model"], "generation": agent.get("generate_content_config")}
        if not isinstance(value["generation"], dict):
            raise ValueError("Every selected LLM must declare generation settings")
        contracts[json.dumps(value, sort_keys=True)] = value
    if not contracts:
        raise ValueError("No selected LLM agents")
    return [contracts[key] for key in sorted(contracts)]


def declared_skills(root: Path) -> list[Path]:
    return sorted(
        {
            root / skill
            for agent in resolve_agent(root, root / "agent.yaml")[1]
            for skill in agent.get("skills", [])
        }
    )
