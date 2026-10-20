"""Check static dependency boundaries and operational numeric literals."""

import ast
import graphlib
import sys
import tomllib
from pathlib import Path
from typing import Any


def imports(tree: ast.AST, module: str, known: set[str]) -> set[str]:
    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = module.split(".")[: -node.level]
                base = ".".join([*parts, *([base] if base else [])])
            edges.add(base)
            edges.update(f"{base}.{a.name}" for a in node.names if f"{base}.{a.name}" in known)
    return edges


def numeric_violations(tree: ast.AST, protected: set[str]) -> list[int]:
    def literal(node: ast.AST | None) -> bool:
        if isinstance(node, ast.UnaryOp):
            node = node.operand
        return isinstance(node, ast.Constant) and type(node.value) in (int, float)

    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            lines.extend(
                k.value.lineno for k in node.keywords if k.arg in protected and literal(k.value)
            )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                and node.args
                and literal(node.args[0])
            ):
                lines.append(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
            lines.extend(
                node.lineno
                for a, v in zip(args, defaults, strict=True)
                if a.arg in protected and literal(v)
            )
            lines.extend(
                node.lineno
                for a, v in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                if a.arg in protected and literal(v)
            )
    return lines


def check(root: Path, policy: dict[str, Any]) -> list[str]:
    modules: dict[str, Path] = {}
    for directory, prefix in policy["roots"].items():
        for path in (root / directory).rglob("*.py"):
            relative = path.relative_to(root / directory).with_suffix("").parts
            parts = [*prefix.split("."), *relative]
            if parts[-1] == "__init__":
                parts.pop()
            modules[".".join(parts)] = path
    roles = {name: role for role, names in policy["modules"].items() for name in names}
    errors = []
    graph = {}
    known = set(modules)
    for name, path in modules.items():
        if name not in roles:
            errors.append(f"Unclassified module: {name}")
            continue
        tree = ast.parse(path.read_text())
        dependencies = imports(tree, name, known)
        graph[name] = dependencies & known
        for dependency in dependencies:
            if name.startswith("gemma_agent_dev") and dependency.split(".")[0] in {
                "tests",
                "scripts",
                "kaggle",
            }:
                errors.append(f"Production imports development entrypoint: {name} -> {dependency}")
            if roles[name] in policy["gpu_free_roles"] and any(
                dependency == blocked or dependency.startswith(blocked + ".")
                for blocked in policy["inference_imports"]
            ):
                errors.append(f"Inference dependency in local code: {name} -> {dependency}")
            if dependency in roles and roles[dependency] not in policy["allowed"][roles[name]]:
                errors.append(f"Dependency boundary: {name} -> {dependency}")
        errors.extend(
            f"Operational numeric literal: {path.relative_to(root)}:{line}"
            for line in numeric_violations(tree, set(policy["numeric_parameters"]))
        )
    try:
        tuple(graphlib.TopologicalSorter(graph).static_order())
    except graphlib.CycleError as error:
        errors.append(f"Import cycle: {error.args[1]}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    policy = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["architecture"]
    errors = check(root, policy)
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print("Architecture checks passed")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
