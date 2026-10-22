"""Validate immutable experiments and deterministic paired execution plans."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.candidates.archive import referenced_files
from gemma_agent_dev.candidates.changes import ChangeScope
from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.config import mapping, project_path
from gemma_agent_dev.evaluation.config import EvaluationSettings, RuntimeSettings
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR, SessionPolicy
from gemma_agent_dev.tasks.pools import TaskPool, identifier


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    hypothesis: str
    pool: Path
    diagnostic_config: Path
    preflight_config: Path
    runtime_config: Path
    evaluation_config: Path
    variants: dict[str, Path]
    session: SessionPolicy
    task_ids: list[str]
    change: ChangeScope | None = None

    @classmethod
    def load(cls, root: Path, path: Path) -> "Experiment":
        raw = yaml.safe_load(project_path(root, str(path)).read_text())
        names = set(cls.__dataclass_fields__) - {"task_ids", "change"}
        if isinstance(raw, dict) and "change" in raw:
            names.add("change")
        data = mapping(
            raw,
            names,
            "experiment",
        )
        identifier(data["experiment_id"])
        identifier(data["hypothesis"])
        for name in (
            "pool",
            "diagnostic_config",
            "preflight_config",
            "runtime_config",
            "evaluation_config",
        ):
            data[name] = project_path(root, data[name])
            if not data[name].is_file():
                raise ValueError(f"Missing experiment input: {name}")
        variants = data["variants"]
        if not isinstance(variants, dict) or len(variants) != VARIANTS_PER_PAIR:
            raise ValueError("A paired experiment requires exactly two variants")
        resolved = {}
        for name, path_value in variants.items():
            identifier(name)
            entry = project_path(root, path_value)
            if entry.parent != root / "agent":
                raise ValueError("Variant roots must belong to agent")
            referenced_files(entry.parent, entry)
            resolved[name] = entry
        if len(set(resolved.values())) != VARIANTS_PER_PAIR:
            raise ValueError("Variants must use distinct agent definitions")
        pool = TaskPool.load(root, data["pool"])
        data["task_ids"] = pool.selection(data["experiment_id"])
        data["variants"] = resolved
        data["session"] = SessionPolicy.parse(data["session"])
        if "change" in data:
            data["change"] = ChangeScope.parse(data["change"])
        evaluation = EvaluationSettings.load(data["evaluation_config"])
        runtime = RuntimeSettings.load(data["runtime_config"])
        if (
            data["session"].pair_seconds(evaluation.as_dict(), runtime.verification_timeout_seconds)
            >= data["session"].job_timeout_seconds
        ):
            raise ValueError("One pair cannot fit inside the session deadline")
        return cls(**data)

    def plan(self) -> list[dict[str, str]]:
        variants = list(self.variants)
        return [
            {"task_id": task, "variant": variant}
            for index, task in enumerate(self.task_ids)
            for variant in (variants if index % VARIANTS_PER_PAIR == 0 else variants[::-1])
        ]

    def definition(self, root: Path) -> dict[str, Any]:
        pool = TaskPool.load(root, self.pool)
        diagnostics = yaml.safe_load(self.diagnostic_config.read_text())
        dependencies = [
            project_path(root, diagnostics[key])
            for key in ("catalog_path", "requirements", "lock_file")
        ]
        runtime = RuntimeSettings.load(self.runtime_config).as_dict()
        runtime.update(task_ids=self.task_ids, job_timeout_seconds=self.session.job_timeout_seconds)
        definition = {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "task_ids": self.task_ids,
            "planned_order": self.plan(),
            "variants": {
                name: referenced_files(path.parent, path) for name, path in self.variants.items()
            },
            "pool": pool.identity(self.experiment_id),
            "inputs": {
                p.relative_to(root).as_posix(): file_hash(p)
                for p in (self.diagnostic_config, self.preflight_config, *dependencies)
            },
            "runtime": runtime,
            "evaluation": EvaluationSettings.load(self.evaluation_config).as_dict(),
            "session": self.session.as_dict(),
        }
        if self.change is not None:
            definition["change"] = self.change.as_dict()
        return definition
