"""Resolve project and private artifact locations for command entry points."""

from pathlib import Path


def project_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").is_file() and (path / "src/gemma_agent_dev").is_dir():
            return path
    raise ValueError("プロジェクトのフォルダー内で実行してください。")


def artifact_directory(root: Path, category: str) -> Path:
    directory = (root / "artifacts" / category).resolve()
    if not directory.is_relative_to(root / "artifacts"):
        raise ValueError("生成物の保存先がartifactsフォルダーの外を指しています。")
    return directory
