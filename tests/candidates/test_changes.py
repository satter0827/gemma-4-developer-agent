from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.candidates.changes import ChangeScope, agent_fields, validate_change
from gemma_agent_dev.common.artifacts import inventory


def test_declared_scope_checks_files_and_root_fields(tmp_path: Path) -> None:
    folders = {name: tmp_path / name for name in ("base", "new")}
    for folder in folders.values():
        folder.mkdir()
        (folder / "agent.yaml").write_text("instruction: !include prompt.md\nname: base\n")
    (folders["new"] / "agent.yaml").write_text("instruction: !include prompt.md\nname: new\n")
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
