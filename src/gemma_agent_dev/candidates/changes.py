"""Check declared candidate differences while protecting fixed evaluation inputs."""

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from gemma_agent_dev.common.config import mapping

FIXED_FILES = frozenset({"configs/sampling.yaml", "eval_config.yaml"})
COMPARISON_SIZE = 2


@dataclass(frozen=True)
class ChangeScope:
    files: list[str]
    agent_fields: list[str]

    @classmethod
    def parse(cls, raw: Any) -> "ChangeScope":
        data = mapping(raw, {"files", "agent_fields"}, "change scope")
        for key, values in data.items():
            if (
                not isinstance(values, list)
                or not values
                or len(values) != len(set(map(str, values)))
            ):
                raise ValueError(f"Expected unique change scope: {key}")
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError("Invalid change scope entry")
        for value in data["files"]:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in value
                or str(path) != value
                or value.startswith("adapters/")
                or value in FIXED_FILES
            ):
                raise ValueError("Change scope contains protected or external files")
        if "agent.yaml" not in data["files"] or any(
            not value.replace("_", "").isalnum() for value in data["agent_fields"]
        ):
            raise ValueError("Declare root agent changes explicitly")
        return cls(**data)

    def as_dict(self) -> dict[str, list[str]]:
        return asdict(self)


def agent_fields(path: Path) -> dict[str, Any]:
    class Loader(yaml.SafeLoader):
        pass

    def include(loader: yaml.SafeLoader, node: yaml.Node) -> dict[str, str]:
        if not isinstance(node, yaml.ScalarNode):
            raise ValueError("Expected scalar include")
        return {"include": loader.construct_scalar(node)}

    Loader.add_constructor("!include", include)
    value = yaml.load(path.read_text(), Loader=Loader)
    if not isinstance(value, dict):
        raise ValueError("Expected root agent mapping")
    return dict(value)


def validate_change(
    scope: ChangeScope, folders: dict[str, Path], files: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    if len(folders) != COMPARISON_SIZE or set(folders) != set(files):
        raise ValueError("Change checking requires the exact two candidates")
    left, right = files.values()
    changed = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
    if not changed or set(changed) - set(scope.files):
        raise ValueError("Candidate changes exceed the declared file scope")
    a, b = (agent_fields(folder / "agent.yaml") for folder in folders.values())
    fields = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if set(fields) - set(scope.agent_fields):
        raise ValueError("Candidate changes exceed the declared root settings")
    return {"files": changed, "agent_fields": fields}
