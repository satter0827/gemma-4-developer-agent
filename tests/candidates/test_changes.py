from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.candidates.changes import ChangeScope, agent_fields, validate_change
from gemma_agent_dev.common.artifacts import inventory


def test_declared_scope_checks_files_and_root_fields(tmp_path: Path) -> None:
    folders = {name: tmp_path / name for name in ("base", "new")}
    for folder in folders.values():
        folder.mkdir()
        (folder / "prompt.md").write_text("instruction")
        (folder / "agent.yaml").write_text(
            "instruction: !include prompt.md\nname: base\nmodel: fixed\n"
            "generate_content_config: {temperature: 0.2}\n"
        )
    (folders["new"] / "agent.yaml").write_text(
        "instruction: !include prompt.md\nname: new\nmodel: fixed\n"
        "generate_content_config: {temperature: 0.2}\n"
    )
    scope = ChangeScope.parse({"files": ["agent.yaml"], "agent_fields": ["name"]})
    assert validate_change(scope, folders, {k: inventory(v) for k, v in folders.items()}) == {
        "files": ["agent.yaml"],
        "agent_fields": ["name"],
    }
    assert scope.as_dict()["files"] == ["agent.yaml"]
    with pytest.raises(ValueError):
        validate_change(
            ChangeScope(["agent.yaml"], ["tools"]),
            folders,
            {k: inventory(v) for k, v in folders.items()},
        )
    (folders["new"] / "extra.md").write_text("extra")
    with pytest.raises(ValueError):
        validate_change(scope, folders, {k: inventory(v) for k, v in folders.items()})
    with pytest.raises(ValueError):
        validate_change(scope, {}, {})
    (folders["new"] / "agent.yaml").write_text("[]")
    with pytest.raises(ValueError):
        agent_fields(folders["new"] / "agent.yaml")


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"files": [], "agent_fields": ["name"]},
        {"files": ["agent.yaml", "agent.yaml"], "agent_fields": ["name"]},
        {"files": [None], "agent_fields": ["name"]},
        {"files": ["../x"], "agent_fields": ["name"]},
        {"files": ["adapters/main.safetensors"], "agent_fields": ["name"]},
        {"files": ["configs/sampling.yaml"], "agent_fields": ["name"]},
        {"files": ["x"], "agent_fields": ["name"]},
        {"files": ["agent.yaml"], "agent_fields": ["bad.field"]},
    ],
)
def test_invalid_change_declarations(raw: Any) -> None:
    with pytest.raises(ValueError):
        ChangeScope.parse(raw)


@pytest.mark.parametrize("field", ["max_output_tokens", "temperature", "model"])
def test_effective_generation_allows_only_declared_changes(tmp_path: Path, field: str) -> None:
    from gemma_agent_dev.candidates.changes import validate_generation

    folders = {name: tmp_path / name for name in ("base", "new")}
    for folder in folders.values():
        folder.mkdir()
        (folder / "agent.yaml").write_text(
            "model: fixed\ngenerate_content_config: !include actual.yaml\n"
        )
        (folder / "actual.yaml").write_text("temperature: 0.2\nmax_output_tokens: 16384\n")
        (folder / "sampling.yaml").write_text("unused: true\n")
    if field == "model":
        (folders["new"] / "agent.yaml").write_text(
            "model: other\ngenerate_content_config: !include actual.yaml\n"
        )
    else:
        (folders["new"] / "actual.yaml").write_text(
            "temperature: "
            + ("0.3" if field == "temperature" else "0.2")
            + "\nmax_output_tokens: "
            + ("8192" if field == "max_output_tokens" else "16384")
            + "\n"
        )
    if field == "max_output_tokens":
        validate_generation(folders, ["max_output_tokens"])
    else:
        with pytest.raises(ValueError, match="Undeclared"):
            validate_generation(folders, ["max_output_tokens"])
    with pytest.raises(ValueError):
        validate_generation(folders, [])
