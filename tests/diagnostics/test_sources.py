import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.diagnostics.local import source_inventory
from gemma_agent_dev.diagnostics.origins import test_environment as source_environment
from gemma_agent_dev.diagnostics.origins import validate_origins
from gemma_agent_dev.diagnostics.sandbox import observe_sources

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("layout,correct", [(".", True), ("src", True), ("src", False)])
def test_real_pytest_rejects_installed_copy_even_when_its_tests_pass(
    tmp_path: Path, layout: str, correct: bool
) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / layout
    source.mkdir(parents=True)
    (source / "example.py").write_text("value = 1\n")
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "example.py").write_text("value = 1\n")
    (workspace / "test_example.py").write_text(
        "from example import value\ndef test_value():\n assert value == 1\n"
    )
    directory = tmp_path / "guard"
    env = {
        **os.environ,
        **source_environment(workspace, [layout] if correct else ["."], ["example"], directory),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    code = (
        f"import sys;sys.path.append({str(installed)!r});import pytest;"
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "-q", str(workspace / "test_example.py")],
        env=env,
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert "1 passed" in result.stdout
    if correct:
        assert result.returncode == 0 and validate_origins(directory / "origins.json")["valid"]
    else:
        assert result.returncode == 1
        with pytest.raises(ValueError, match="workspace"):
            validate_origins(directory / "origins.json")


def test_snapshot_links_are_preserved_but_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "original").write_text("content")
    (root / "alias").symlink_to("original")
    assert source_inventory(root)["alias"] == "symlink:original"
    with pytest.raises(ValueError):
        inventory(root)
    (root / "alias").unlink()
    (root / "alias").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Unsafe"):
        source_inventory(root)


@pytest.mark.parametrize("mode", ["valid", "missing", "ambiguous", "empty-workspace"])
def test_official_subprocess_source_guard_records_actual_test_process(
    tmp_path: Path, mode: str
) -> None:
    root = tmp_path / "sandbox"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    if mode != "empty-workspace":
        (workspace / "example.py").write_text("value = 1\n")

    class Sandbox:
        sandboxes = {"one": {"root": root, "workspace": workspace}}

        def exec(self, sandbox_id: str, command: str, **kwargs: Any) -> Any:
            if command.startswith("cat "):
                return SimpleNamespace(stdout="<testsuite><testcase/></testsuite>")
            assert ("GEMMA_SOURCE_WORKSPACE" in command) is (mode != "empty-workspace")
            if mode == "valid":
                write_json(
                    root / "source-guard/origins.json",
                    {
                        "valid": True,
                        "workspace": str(workspace),
                        "modules": {"example": str(workspace / "example.py")},
                    },
                )
            return SimpleNamespace(stdout="tests")

    specs = {"task": {"source_roots": ["."], "import_modules": ["example"]}}
    if mode == "ambiguous":
        specs["other"] = {"source_roots": [".", "src"], "import_modules": ["example"]}
    sandbox = Sandbox()
    check = observe_sources(sandbox, specs, tmp_path / "out")
    if mode == "ambiguous":
        with pytest.raises(ValueError, match="Ambiguous"):
            sandbox.exec("one", "python3 -m pytest --junitxml=/tmp/_swegemma_junit_abc123.xml")
    else:
        sandbox.exec("one", "python3 -m pytest --junitxml=/tmp/_swegemma_junit_abc123.xml")
        if mode == "valid":
            check()
            assert (tmp_path / "out/source-checks/one/origins.json").is_file()
        else:
            with pytest.raises(ValueError, match="source verification"):
                check()


def test_source_plugin_records_import_errors_and_excludes_external_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    from gemma_agent_dev.diagnostics import origins as provenance

    (tmp_path / "example.py").write_text("value=1")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("GEMMA_SOURCE_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GEMMA_SOURCE_MODULES", '["example"]')
    monkeypatch.setenv("GEMMA_SOURCE_EVIDENCE", str(tmp_path / "origins.json"))
    monkeypatch.setattr(provenance, "IMPORT_ERRORS", {})
    real_import = importlib.import_module

    def fail(name: str) -> Any:
        if name == "example":
            raise ImportError("missing dependency")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    provenance.pytest_sessionstart(None)
    external = types.ModuleType("third_party")
    external.__file__ = "/outside/third_party.py"
    monkeypatch.setitem(sys.modules, "example.alias", external)
    session = SimpleNamespace(exitstatus=2)
    provenance.pytest_sessionfinish(session, 2)
    evidence = validate_origins(tmp_path / "origins.json")
    assert (
        evidence["import_errors"]["example"] == "missing dependency"
        and "example.alias" not in evidence["modules"]
    )
    assert session.exitstatus == 2


@pytest.mark.parametrize("fault", ["none", "checksum", "install", "stale"])
def test_offline_sandbox_dependencies_and_stale_source_evidence_are_guarded(
    tmp_path: Path, fault: str
) -> None:
    root = tmp_path / "sandbox"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "example.py").write_text("value=1")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "helper-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    installs = []

    class Sandbox:
        sandboxes = {"one": {"root": root, "workspace": workspace, "venv": root / "venv"}}

        def exec(self, sid: str, command: str, **kwargs: Any) -> Any:
            if "pip install" in command:
                installs.append(command)
                return SimpleNamespace(exit_code=1 if fault == "install" else 0)
            if command.startswith("cat "):
                return SimpleNamespace(stdout="<testsuite><testcase/></testsuite>")
            if fault != "stale":
                write_json(
                    root / "source-guard/origins.json",
                    {
                        "valid": True,
                        "workspace": str(workspace),
                        "modules": {"example": str(workspace / "example.py")},
                    },
                )
            return SimpleNamespace(stdout="done")

    spec = {
        "source_roots": ["."],
        "import_modules": ["example"],
        "dependency_wheels": {wheel.name: "0" * 64 if fault == "checksum" else file_hash(wheel)},
    }
    sandbox = Sandbox()
    check = observe_sources(sandbox, {"one": spec}, tmp_path / "out", wheels)
    if fault in {"checksum", "install"}:
        with pytest.raises((ValueError, RuntimeError)):
            sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
    elif fault == "stale":
        (root / "source-guard").mkdir()
        write_json(
            root / "source-guard/origins.json",
            {
                "valid": True,
                "workspace": str(workspace),
                "modules": {"example": str(workspace / "example.py")},
            },
        )
        sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
        with pytest.raises(ValueError):
            check()
    else:
        sandbox.exec("one", "python -m pytest --junitxml=/tmp/_swegemma_junit_abc.xml")
        check()
        (workspace / "example.py").unlink()
        sandbox.exec("one", "echo done")
        assert len(installs) == 1
