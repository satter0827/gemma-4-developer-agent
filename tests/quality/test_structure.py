"""Exercise recursive source closure and feature-scoped validation inputs."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gemma_agent_dev.candidates.preflight import PreflightSettings, check_installation
from gemma_agent_dev.common.sources import source_paths, validate_bundle_sources
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import check_prepared
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.notebooks.bundle import payload, write_notebook

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("damage", ["missing", "changed", "extra"])
def test_installation_checks_nested_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    import gemma_agent_dev

    expected = tmp_path / "project/src/gemma_agent_dev"
    actual = tmp_path / "installed"
    shutil.copytree(ROOT / "src/gemma_agent_dev", expected)
    shutil.copytree(expected, actual)
    monkeypatch.setattr(gemma_agent_dev, "__file__", str(actual / "__init__.py"))
    check_installation(tmp_path / "project")
    changed = actual / "common/config.py"
    if damage == "missing":
        changed.unlink()
    elif damage == "changed":
        changed.write_text("# changed")
    else:
        (actual / "common/extra.py").write_text("pass")
    with pytest.raises(ValueError, match="stale"):
        check_installation(tmp_path / "project")


@pytest.mark.parametrize(
    "statement",
    [
        "import gemma_agent_dev.common.b",
        "from gemma_agent_dev.common import b",
        "from gemma_agent_dev.common.b import value",
        "from . import b",
    ],
)
def test_bundle_rejects_missing_internal_dependencies(tmp_path: Path, statement: str) -> None:
    folder = tmp_path / "src/gemma_agent_dev/common"
    folder.mkdir(parents=True)
    (folder.parent / "__init__.py").touch()
    (folder / "__init__.py").touch()
    (folder / "a.py").write_text(statement)
    (folder / "b.py").write_text("value = 1")
    paths = source_paths(tmp_path, ("common.a",))
    with pytest.raises(ValueError, match="import|dependency"):
        validate_bundle_sources(tmp_path, paths)
    (folder / "__init__.py").unlink()
    with pytest.raises(ValueError, match="Missing"):
        source_paths(tmp_path, ("common.a",))


def test_bundle_hash_covers_launcher_and_is_reproducible(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    modules = (
        "notebooks.entry",
        "notebooks.worker",
        "common.processes",
        "common.ownership",
        "common.artifacts",
        "common.events",
        "common.config",
    )
    first, _ = payload(tmp_path, modules, {})
    assert payload(tmp_path, modules, {})[0] == first
    launcher = tmp_path / "src/gemma_agent_dev/notebooks/entry.py"
    launcher.write_text(launcher.read_text() + "\n# changed\n")
    assert payload(tmp_path, modules, {})[0] != first


def test_notebook_uses_the_launcher_already_hashed_in_its_payload(tmp_path: Path) -> None:
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    folder = tmp_path / "src/gemma_agent_dev/notebooks"
    folder.mkdir(parents=True)
    (folder.parent / "__init__.py").touch()
    (folder / "__init__.py").touch()
    launcher = folder / "entry.py"
    original = "def main(*args, **kwargs): pass\n"
    launcher.write_text(original)
    content, _ = payload(tmp_path, ("notebooks.entry",), {})
    launcher.write_text("raise RuntimeError('changed after bundling')")
    output = tmp_path / "bundle"
    runtime = RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml").as_dict()
    write_notebook(
        output, content, {}, runtime, runtime["kernel_id"], verify_only=True, diagnostic=False
    )
    assert (output / "run.py").read_text().startswith(original)


@pytest.mark.parametrize(
    "kind", ["policy", "annotations", "unselected", "selected", "code", "lock", "inference"]
)
def test_preflight_binds_only_relevant_settings(preflight_project: Path, kind: str) -> None:
    root = preflight_project
    before = PreflightSettings.load(root, Path("preflight.yaml"), "smoke").inputs()
    irrelevant = {
        "policy": "configs/submission/policy.yaml",
        "annotations": "configs/tasks/annotations.yaml",
    }
    if kind in irrelevant:
        path = root / irrelevant[kind]
        path.write_text(path.read_text() + "\n# unrelated change\n")
    elif kind in {"unselected", "selected"}:
        path = root / "configs/evaluation/profiles.yaml"
        profiles = yaml.safe_load(path.read_text())
        if kind == "unselected":
            profiles["workflow"]["agent_config"] = "not-selected.yaml"
        else:
            profiles["smoke"]["agent_config"] = "agent/workflow-v1.yaml"
        path.write_text(yaml.safe_dump(profiles))
    elif kind == "inference":
        path = root / "runtime.json"
        data = json.loads(path.read_text())
        data["verification_timeout_seconds"] += 1
        path.write_text(json.dumps(data))
    else:
        path = root / ("src/gemma_agent_dev/common/config.py" if kind == "code" else "uv.lock")
        path.write_text(path.read_text() + "\n# changed\n")
    after = PreflightSettings.load(root, Path("preflight.yaml"), "smoke").inputs()
    assert (before == after) is (kind in {"policy", "annotations", "unselected"})


def test_runtime_changes_do_not_require_diagnostic_environment_recreation(
    diagnostic_project: DiagnosticSettings,
) -> None:
    settings = diagnostic_project
    preparation, execution = settings.preparation_fingerprint(), settings.fingerprint()
    runtime = json.loads(settings.runtime_path.read_text())
    runtime["verification_timeout_seconds"] += 1
    settings.runtime_path.write_text(json.dumps(runtime))
    changed = replace(settings, timeout_seconds=settings.timeout_seconds + 1)
    assert changed.preparation_fingerprint() == preparation
    assert changed.fingerprint() != execution
    check_prepared(changed)


def test_runtime_fragments_are_closed_and_resolved(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").touch()
    config = tmp_path / "configs/evaluation/runtime.yaml"
    assert (
        RuntimeSettings.load(config).as_dict()
        == RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml").as_dict()
    )
    task_path = tmp_path / "configs/tasks/smoke.yaml"
    data: dict[str, Any] = yaml.safe_load(task_path.read_text())
    data["model_source"] = "duplicate"
    task_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="distinct"):
        RuntimeSettings.load(config)
