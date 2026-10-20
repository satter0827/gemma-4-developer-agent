import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from gemma_agent_dev.kaggle_bundle import prepare


def test_kaggle_bundle_is_private_and_contains_only_runtime_sources(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "kernel"
    digest = prepare(root, destination)
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    module = ast.parse((destination / "run.py").read_text())
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    payload = ast.literal_eval(statement.value.args[0])
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        names = archive.namelist()
        assert "agent/agent.yaml" in names
        assert "src/gemma_agent_dev/runtime.py" in names
        assert not any(name.endswith(("app.py", "cli.py", ".safetensors")) for name in names)
        assert all(name.startswith(("src/", "agent/", "configs/", "kaggle/")) for name in names)
    assert json.loads((destination / "source-manifest.json").read_text())["source_sha256"] == digest


def test_verification_bundle_shares_sources_without_gpu_inputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    evaluation = tmp_path / "evaluation"
    verification = tmp_path / "verification"
    assert prepare(root, evaluation) == prepare(root, verification, verify_only=True)
    cpu = json.loads((verification / "kernel-metadata.json").read_text())
    gpu = json.loads((evaluation / "kernel-metadata.json").read_text())
    assert cpu["id"] != gpu["id"]
    for metadata in (cpu, gpu):
        assert metadata["title"].lower().replace(" ", "-") == metadata["id"].split("/")[-1]
    assert cpu["is_private"] and not cpu["enable_gpu"] and not cpu["enable_internet"]
    assert not cpu["model_sources"] and "machine_shape" not in cpu
    assert gpu["enable_gpu"] and gpu["model_sources"]


def source_files(directory: Path) -> dict[str, bytes]:
    module = ast.parse((directory / "run.py").read_text())
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    with zipfile.ZipFile(
        io.BytesIO(base64.b64decode(ast.literal_eval(statement.value.args[0])))
    ) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_selected_budget_changes_only_embedded_evaluation_configuration(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "agent/eval_config.yaml").read_bytes()
    baseline, candidate, verification = (tmp_path / name for name in ("one", "five", "cpu"))
    one_digest = prepare(root, baseline)
    five_digest = prepare(root, candidate, eval_config=Path("configs/evaluation/5min.yaml"))
    assert one_digest != five_digest
    assert five_digest == prepare(
        root, verification, verify_only=True, eval_config=Path("configs/evaluation/5min.yaml")
    )
    one, five = source_files(baseline), source_files(candidate)
    assert one.keys() == five.keys()
    assert [name for name in one if one[name] != five[name]] == ["agent/eval_config.yaml"]
    assert five["agent/eval_config.yaml"] == (root / "configs/evaluation/5min.yaml").read_bytes()
    assert "configs/evaluation/5min.yaml" not in five
    assert (root / "agent/eval_config.yaml").read_bytes() == original
    manifest = json.loads((candidate / "source-manifest.json").read_text())
    assert manifest["evaluation_config"]["path"] == "configs/evaluation/5min.yaml"
    assert manifest["evaluation_config"]["values"]["max_time_minutes"] == 5
    assert (
        manifest["evaluation_config"]["sha256"]
        == hashlib.sha256(five["agent/eval_config.yaml"]).hexdigest()
    )


@pytest.mark.parametrize("symlink", [False, True])
def test_selected_configuration_cannot_escape_project(tmp_path: Path, symlink: bool) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("evaluation: {}")
    selected = outside
    if symlink:
        selected = root / "selected.yaml"
        selected.symlink_to(outside)
    destination = root / "artifacts/kaggle/run"
    with pytest.raises(ValueError, match="inside the project"):
        prepare(root, destination, eval_config=selected)
    assert not destination.exists()
