"""Check declared candidate differences while protecting fixed evaluation inputs."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from gemma_agent_dev.candidates.archive import generation_contract
from gemma_agent_dev.common.config import mapping

FIXED_FILES = frozenset({"eval_config.yaml"})
GENERATION_FIELDS = frozenset({"max_output_tokens"})
COMPARISON_SIZE = 2


@dataclass(frozen=True)
class ChangeScope:
    files: list[str]
    agent_fields: list[str]
    generation_fields: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: Any) -> "ChangeScope":
        keys = {"files", "agent_fields"}
        if isinstance(raw, dict) and "generation_fields" in raw:
            keys.add("generation_fields")
        data = mapping(raw, keys, "change scope")
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
        if set(data.get("generation_fields", [])) - GENERATION_FIELDS:
            raise ValueError("Unsupported generation change")
        if set(data["agent_fields"]) & {"model", "adapter"}:
            raise ValueError("Model and LoRA must remain fixed")
        if not data.get("generation_fields") and "configs/sampling.yaml" in data["files"]:
            raise ValueError("Generation changes must be declared")
        return cls(**data)

    def as_dict(self) -> dict[str, list[str]]:
        result = {"files": self.files, "agent_fields": self.agent_fields}
        if self.generation_fields:
            result["generation_fields"] = self.generation_fields
        return result


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
    validate_generation(folders, scope.generation_fields)
    return {
        "files": changed,
        "agent_fields": fields,
        **({"generation_fields": scope.generation_fields} if scope.generation_fields else {}),
    }


def validate_generation(folders: dict[str, Path], allowed: list[str]) -> None:
    """Permit only explicitly declared differences in effective LLM generation."""
    import json

    contracts = []
    for folder in folders.values():
        values = set()
        for contract in generation_contract(folder):
            fixed = {k: v for k, v in contract["generation"].items() if k not in allowed}
            values.add(
                json.dumps({"model": contract["model"], "generation": fixed}, sort_keys=True)
            )
        contracts.append(values)
    if any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("Undeclared model or generation differences")
