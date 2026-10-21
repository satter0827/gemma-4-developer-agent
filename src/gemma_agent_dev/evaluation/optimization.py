"""Resolve a search definition separately from paired candidate comparisons."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from gemma_agent_dev.candidates.archive import referenced_files
from gemma_agent_dev.candidates.preflight import PreflightSettings
from gemma_agent_dev.common.artifacts import file_hash
from gemma_agent_dev.common.config import mapping, project_path
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.evaluation.budget import GpuBudget
from gemma_agent_dev.evaluation.config import EvaluationSettings, RuntimeSettings
from gemma_agent_dev.evaluation.optimization_limits import SearchLimits
from gemma_agent_dev.tasks.pools import TaskPool, identifier, task_ids


@dataclass(frozen=True)
class Optimization:
    optimization_id: str
    hypothesis: str
    pool: Path
    diagnostic_config: Path
    preflight_config: Path
    runtime_config: Path
    evaluation_config: Path
    seed_agent: Path
    prompt_path: str
    search_task_ids: list[str]
    selection_task_ids: list[str]
    environment: Path
    limits: SearchLimits
    gpu_budget: GpuBudget | None

    @classmethod
    def load(cls, root: Path, config: Path) -> "Optimization":
        data = mapping(
            yaml.safe_load(project_path(root, str(config)).read_text()),
            set(cls.__dataclass_fields__),
            "optimization",
        )
        identifier(data["optimization_id"])
        identifier(data["hypothesis"])
        for key in (
            "pool",
            "diagnostic_config",
            "preflight_config",
            "runtime_config",
            "evaluation_config",
            "seed_agent",
            "environment",
        ):
            data[key] = project_path(root, data[key])
            if not data[key].is_file():
                raise ValueError("Missing optimization input")
        for name in ("search_task_ids", "selection_task_ids"):
            task_ids(data[name])
        selected = data["search_task_ids"] + data["selection_task_ids"]
        if len(set(selected)) != len(selected):
            raise ValueError("Search and selection tasks must be disjoint")
        pool = TaskPool.load(root, data["pool"])
        if set(pool.selection(data["optimization_id"])) != set(selected) or not set(
            selected
        ) <= set(pool.tuning):
            raise ValueError("Optimization must use its frozen tuning selection")
        relative = PurePosixPath(data["prompt_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != data["prompt_path"]
            or not data["prompt_path"].startswith("prompts/")
            or relative.suffix != ".md"
        ):
            raise ValueError("Expected one authored prompt inside the candidate")
        data["limits"] = SearchLimits.parse(data["limits"])
        data["gpu_budget"] = (
            GpuBudget.parse(data["gpu_budget"]) if data["gpu_budget"] is not None else None
        )
        result = cls(**data)
        if data["seed_agent"].parent != root / "agent":
            raise ValueError("Seed agent must belong to agent")
        if result.prompt_path not in referenced_files(root / "agent", result.seed_agent):
            raise ValueError("Mutable instruction must belong to the seed candidate")
        diagnostic = DiagnosticSettings.load(root, result.diagnostic_config)
        if (
            set(t.instance_id for t in diagnostic.tasks) != set(selected)
            or pool.tasks_sha256 != diagnostic.tasks_sha256
            or pool.sdk_version != diagnostic.runtime.packages["swegemma"]
        ):
            raise ValueError("Optimization CPU controls must cover the exact task set and SDK")
        if result.limits.reflection_minibatch_size > len(result.search_task_ids):
            raise ValueError("Reflection minibatch exceeds the search set")
        if (
            result.gpu_budget is not None
            and result.gpu_budget.job_timeout_seconds < result.required_seconds()
        ):
            raise ValueError("GPU deadline is below the full optimization estimate")
        return result

    def preflight(self, root: Path) -> PreflightSettings:
        base = PreflightSettings.load(root, self.preflight_config, "smoke")
        return replace(
            base,
            profile=self.optimization_id,
            purpose="optimization",
            agent_config=self.seed_agent,
            evaluation_config=self.evaluation_config,
            diagnostic_config=self.diagnostic_config,
            runtime_config=self.runtime_config,
            task_selection=None,
        )

    def required_seconds(self) -> float:
        return self.limits.required_seconds(
            RuntimeSettings.load(self.runtime_config).as_dict(),
            EvaluationSettings.load(self.evaluation_config).as_dict(),
        )

    def definition(self, root: Path) -> dict[str, Any]:
        environment = yaml.safe_load(self.environment.read_text())
        inputs = [self.diagnostic_config, self.environment, self.pool]
        inputs.extend(project_path(root, environment[key]) for key in ("requirements", "lock_file"))
        return {
            "optimization_id": self.optimization_id,
            "hypothesis": self.hypothesis,
            "search_task_ids": self.search_task_ids,
            "selection_task_ids": self.selection_task_ids,
            "prompt_path": self.prompt_path,
            "runtime": RuntimeSettings.load(self.runtime_config).as_dict(),
            "evaluation": EvaluationSettings.load(self.evaluation_config).as_dict(),
            "limits": asdict(self.limits),
            "gpu_budget": asdict(self.gpu_budget) if self.gpu_budget is not None else None,
            "required_seconds": self.required_seconds(),
            "inputs": {p.relative_to(root).as_posix(): file_hash(p) for p in inputs},
        }
