"""Declare and verify the exact source inputs used by isolated execution paths."""

import ast
from pathlib import Path

BASE_MODULES = (
    "common.artifacts",
    "common.config",
    "common.junit",
    "common.processes",
    "evaluation.config",
    "evaluation.official",
    "diagnostics.origins",
    "diagnostics.sandbox",
    "notebooks.environment",
    "notebooks.worker",
    "notebooks.entry",
)
EVALUATION_MODULES = (
    *BASE_MODULES,
    "candidates.archive",
    "evaluation.results",
    "evaluation.runtime",
)
DIAGNOSTIC_MODULES = (*BASE_MODULES, "diagnostics.remote")
LOCAL_DIAGNOSTIC_MODULES = (
    "common.sources",
    "common.artifacts",
    "common.config",
    "common.junit",
    "common.processes",
    "diagnostics.config",
    "diagnostics.local",
    "diagnostics.origins",
    "evaluation.config",
    "tasks.catalog",
)


def source_paths(root: Path, modules: tuple[str, ...]) -> list[Path]:
    package = root / "src/gemma_agent_dev"
    paths = {package / (name.replace(".", "/") + ".py") for name in modules}
    for path in list(paths):
        for parent in (path.parent, *path.parent.parents):
            if not parent.is_relative_to(package):
                break
            paths.add(parent / "__init__.py")
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(package):
            raise ValueError(f"Missing or linked source input: {path}")
    return sorted(paths)


def validate_bundle_sources(root: Path, paths: list[Path]) -> None:
    package = root / "src"
    modules = set()
    for path in paths:
        parts = path.relative_to(package).with_suffix("").parts
        modules.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            dependencies = []
            if isinstance(node, ast.Import):
                dependencies = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(f"Relative bundle import is unsupported: {path}")
                base = node.module or ""
                dependencies = [base]
                for alias in node.names:
                    child = f"{base}.{alias.name}"
                    location = package / child.replace(".", "/")
                    if location.with_suffix(".py").is_file() or location.is_dir():
                        dependencies.append(child)
            for dependency in dependencies:
                if dependency.startswith("gemma_agent_dev") and dependency not in modules:
                    raise ValueError(f"Bundle dependency is missing: {path.name}: {dependency}")
