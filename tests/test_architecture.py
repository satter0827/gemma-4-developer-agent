import ast
import json
import runpy
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.candidates import extract_archive, validate_references

ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "scripts/check_architecture.py"))


def test_project_dependencies_and_operational_values() -> None:
    policy = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["architecture"]
    assert CHECKER["check"](ROOT, policy) == []
    assert CHECKER["main"]() == 0


@pytest.mark.parametrize(
    "sources,expected",
    [
        ({"a": "from . import b", "b": "from . import a"}, "Import cycle"),
        ({"a": "def run():\n import pkg.a"}, "Import cycle"),
        ({"a": "if False:\n import pkg.b", "b": "from pkg import a as other"}, "Import cycle"),
        ({"a": "import tests.helper", "b": "pass"}, "development entrypoint"),
        ({"a": "from pkg import b", "b": "pass"}, "Dependency boundary"),
        ({"a": "def run():\n import torch", "b": "pass"}, "Inference dependency"),
    ],
)
def test_checker_rejects_bad_structures(
    tmp_path: Path, sources: dict[str, str], expected: str
) -> None:
    directory = tmp_path / "src"
    directory.mkdir()
    for name, code in sources.items():
        (directory / f"{name}.py").write_text(code)
    policy: dict[str, Any] = {
        "roots": {"src": "pkg"},
        "modules": {"one": ["pkg.a"], "two": ["pkg.b"] if "b" in sources else []},
        "allowed": {"one": ["one"], "two": ["one", "two"]},
        "numeric_parameters": ["timeout"],
        "gpu_free_roles": ["one"],
        "inference_imports": ["torch"],
    }
    if expected == "development entrypoint":
        policy["roots"] = {"src": "gemma_agent_dev"}
        policy["modules"] = {"one": ["gemma_agent_dev.a", "gemma_agent_dev.b"]}
        policy["allowed"] = {"one": ["one"]}
    assert any(expected in error for error in CHECKER["check"](tmp_path, policy))
    (directory / "new.py").write_text("pass")
    assert any("Unclassified" in error for error in CHECKER["check"](tmp_path, policy))


@pytest.mark.parametrize(
    "code,bad",
    [
        ("run(timeout=1)", True),
        ("time.sleep(3)", True),
        ("def run(*, timeout=300): pass", True),
        ("def run(timeout=-1): pass", True),
        ("def run(value, *, max_bytes=200): pass", True),
        ("run(timeout=settings.timeout_seconds)", False),
        ("items[2]; print(value, indent=2)", False),
    ],
)
def test_numeric_policy_is_targeted(code: str, bad: bool) -> None:
    assert bool(CHECKER["numeric_violations"](ast.parse(code), {"timeout", "max_bytes"})) is bad


def test_all_imports_are_inert() -> None:
    script = """
import importlib, pathlib, sys
sys.path.insert(0, sys.argv[1])
def audit(event,args):
    if event in {
        'subprocess.Popen','os.system','socket.connect','socket.bind','os.chdir',
        'os.mkdir','os.remove','os.rename','os.rmdir','os.putenv'
    }:
        raise AssertionError(event)
    if event == 'open' and isinstance(args[1],str) and any(x in args[1] for x in 'wax+'):
        raise AssertionError(event)
sys.addaudithook(audit)
for path in pathlib.Path(sys.argv[1], 'gemma_agent_dev').glob('*.py'):
    importlib.import_module('gemma_agent_dev.' + path.stem)
assert not {'torch','litellm','swegemma'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")], check=True)


@pytest.mark.parametrize("mode", ["missing", "escape", "cycle", "root", "suffix", "symlink"])
def test_invalid_submission_references(tmp_path: Path, mode: str) -> None:
    (tmp_path / "agent.yaml").write_text("instruction: !include prompt.md")
    (tmp_path / "prompt.md").write_text("hello")
    if mode == "missing":
        (tmp_path / "prompt.md").unlink()
    elif mode == "escape":
        (tmp_path / "agent.yaml").write_text("instruction: !include ../outside.md")
    elif mode == "cycle":
        (tmp_path / "agent.yaml").write_text("instruction: !include agent.yaml")
    elif mode == "root":
        (tmp_path / "agent.yml").write_text("{}")
    elif mode == "suffix":
        (tmp_path / "data.bin").write_text("bad")
    else:
        (tmp_path / "link.md").symlink_to(tmp_path / "prompt.md")
    with pytest.raises(ValueError):
        validate_references(tmp_path)


def test_shared_includes_and_parent_references_within_root_are_valid(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "agent.yaml").write_text(
        "tools: [{agent_tool: {config_path: sub/agent.yaml}}]\ninstruction: !include prompt.md"
    )
    (tmp_path / "sub/agent.yaml").write_text(
        "instruction: !include ../prompt.md\noptions: !include ../options.yaml"
    )
    (tmp_path / "options.yaml").write_text("value: 1")
    (tmp_path / "prompt.md").write_text("hello")
    assert len(validate_references(tmp_path)) == 4
    (tmp_path / "sub/agent.yaml").write_text("sub_agents: [{config_path: ../agent.yaml}]")
    with pytest.raises(ValueError, match="Cyclic"):
        validate_references(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\b", "./agent.yaml"])
def test_archive_traversal_is_rejected(tmp_path: Path, name: str) -> None:
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError):
        extract_archive(source, tmp_path / "out", 100)


def test_archive_size_empty_and_duplicate_checks(tmp_path: Path) -> None:
    source = tmp_path / "a.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("agent.yaml", json.dumps({"data": "x"}))
    with pytest.raises(ValueError, match="size"):
        extract_archive(source, tmp_path / "out", 1)
    with zipfile.ZipFile(source, "w"):
        pass
    with pytest.raises(ValueError, match="Empty"):
        extract_archive(source, tmp_path / "out", 100)


@pytest.mark.parametrize("cyclic", [False, True])
def test_package_initializer_imports_are_resolved(tmp_path: Path, cyclic: bool) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "__init__.py").write_text("from . import child")
    (source / "child.py").write_text("import pkg" if cyclic else "pass")
    policy = {
        "roots": {"src": "pkg"},
        "modules": {"core": ["pkg", "pkg.child"]},
        "allowed": {"core": ["core"]},
        "numeric_parameters": [],
        "gpu_free_roles": [],
        "inference_imports": [],
    }
    errors = CHECKER["check"](tmp_path, policy)
    assert bool(errors) is cyclic
    assert not errors or "Import cycle" in errors[0]


@pytest.mark.parametrize("damage", ["duplicate", "missing", "allowed", "role", "root", "empty"])
def test_architecture_policy_cannot_silently_override_or_skip_modules(
    tmp_path: Path, damage: str
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text("pass")
    policy: dict[str, Any] = {
        "roots": {"src": "pkg"},
        "modules": {"core": ["pkg.a"]},
        "allowed": {"core": ["core"]},
        "numeric_parameters": [],
        "gpu_free_roles": ["core"],
        "inference_imports": [],
    }
    if damage == "duplicate":
        policy["modules"]["other"] = ["pkg.a"]
        policy["allowed"]["other"] = ["other"]
    elif damage == "missing":
        policy["modules"]["core"].append("pkg.missing")
    elif damage == "allowed":
        policy["allowed"]["core"].append("unknown")
    elif damage == "role":
        policy["allowed"] = {}
    elif damage == "root":
        second = tmp_path / "other"
        second.mkdir()
        (second / "a.py").write_text("pass")
        policy["roots"]["other"] = "pkg"
    else:
        policy["roots"] = {}
        policy["modules"]["core"] = []
    assert CHECKER["check"](tmp_path, policy)
