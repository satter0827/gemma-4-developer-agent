"""Load the evaluation contract without implicit CLI defaults."""

import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.config import mapping, positive, project_path

MODEL_SOURCE_PARTS = 5
MAX_TCP_PORT = 65535


@dataclass(frozen=True)
class RuntimePaths:
    data: str
    model: str
    wheelhouse: str
    output: str


@dataclass(frozen=True)
class InferenceSettings:
    port: int
    host: str
    tool_call_parser: str
    reasoning_parser: str
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float
    enable_auto_tool_choice: bool
    enable_lora: bool
    max_loras: int
    max_lora_rank: int
    tensor_parallel_size: int
    startup_timeout: int
    scheduler_reserve_full_isl: bool


@dataclass(frozen=True)
class CacheSettings:
    min_tokens: int
    ttl_seconds: int
    cache_intervals: int


@dataclass(frozen=True)
class CompactionSettings:
    compaction_interval: int
    overlap_size: int
    token_threshold: int
    event_retention_size: int


@dataclass(frozen=True)
class RuntimeSettings:
    competition: str
    kernel_id: str
    verification_kernel_id: str
    model_source: str
    wheelhouse_source: str
    docker_image: str
    machine_shape: str
    task_ids: list[str]
    paths: RuntimePaths
    packages: dict[str, str]
    wheelhouse_hashes: dict[str, str]
    job_timeout_seconds: int
    progress_interval_seconds: int
    verification_timeout_seconds: int
    concurrency: int
    vllm: InferenceSettings
    context_cache: CacheSettings
    compaction: CompactionSettings

    @classmethod
    def load(cls, path: Path) -> "RuntimeSettings":
        document = yaml.safe_load(path.read_text())
        if isinstance(document, dict) and set(document) == {
            "environment",
            "inference",
            "sdk",
            "tasks",
        }:
            root = next(
                (p for p in path.resolve().parents if (p / "pyproject.toml").is_file()), None
            )
            if root is None:
                raise ValueError("Runtime configuration must belong to a project")
            resolved: dict[str, Any] = {}
            for reference in document.values():
                fragment = yaml.safe_load(project_path(root, reference).read_text())
                if not isinstance(fragment, dict) or set(fragment) & set(resolved):
                    raise ValueError("Runtime fragments must have distinct mapping keys")
                resolved.update(fragment)
            document = resolved
        value = mapping(document, {f.name for f in fields(cls)}, "runtime")
        for name in ("competition", "model_source", "docker_image", "machine_shape"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise ValueError(f"Invalid runtime string: {name}")
        for name in ("kernel_id", "verification_kernel_id", "wheelhouse_source"):
            if not isinstance(value[name], str) or not re.fullmatch(r"[\w-]+/[\w-]+", value[name]):
                raise ValueError(f"Invalid Kaggle resource: {name}")
        if value["kernel_id"] == value["verification_kernel_id"]:
            raise ValueError("Verification and GPU evaluation require separate notebooks")
        ids = value["task_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) or not re.fullmatch(r"[\w-]+", i) for i in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("Invalid or duplicate task IDs")
        packages = value["packages"]
        if (
            not isinstance(packages, dict)
            or not packages
            or any(
                not isinstance(k, str)
                or not isinstance(v, str)
                or not re.fullmatch(r"\d+(?:\.\d+)+(?:[\w.]+)?", v)
                for k, v in packages.items()
            )
        ):
            raise ValueError("Expected pinned official package versions")
        positive(value["concurrency"], "concurrency", integer=True)
        positive(value["job_timeout_seconds"], "job timeout", integer=True)
        positive(value["progress_interval_seconds"], "progress interval", integer=True)
        if value["progress_interval_seconds"] >= value["job_timeout_seconds"]:
            raise ValueError("Progress interval must be below the job deadline")
        hashes = value["wheelhouse_hashes"]
        if (
            not isinstance(hashes, dict)
            or not hashes
            or any(
                not isinstance(k, str)
                or Path(k).name != k
                or not k.endswith(".whl")
                or not isinstance(v, str)
                or not re.fullmatch(r"[a-f0-9]{64}", v)
                for k, v in hashes.items()
            )
        ):
            raise ValueError("Expected official wheel hashes")
        if any(
            f"{name.replace('-', '_')}-{version}-py3-none-any.whl" not in hashes
            for name, version in packages.items()
        ):
            raise ValueError("Official packages are missing expected wheel hashes")
        positive(value["verification_timeout_seconds"], "verification timeout", integer=True)
        nested: dict[str, Any] = {}
        for name, kind in (
            ("paths", RuntimePaths),
            ("vllm", InferenceSettings),
            ("context_cache", CacheSettings),
            ("compaction", CompactionSettings),
        ):
            data = mapping(value[name], {f.name for f in fields(kind)}, name)
            for f in fields(kind):
                item = data[f.name]
                if f.type is bool:
                    if type(item) is not bool:
                        raise ValueError(f"Expected boolean: {name}.{f.name}")
                elif f.type is str:
                    if not isinstance(item, str) or not item:
                        raise ValueError(f"Expected string: {name}.{f.name}")
                else:
                    positive(item, f"{name}.{f.name}", integer=f.type is int)
            nested[name] = kind(**data)
        paths = nested["paths"]
        if any(not Path(p).is_absolute() for p in asdict(paths).values()):
            raise ValueError("Runtime paths must be absolute")
        source = value["model_source"].split("/")
        if (
            len(source) != MODEL_SOURCE_PARTS
            or any(not re.fullmatch(r"[\w.-]+", part) for part in source)
            or not source[-1].isdigit()
            or int(source[-1]) < 1
            or tuple(Path(paths.model).parts[-MODEL_SOURCE_PARTS:])
            != (source[0], source[1], source[2].lower(), source[3], source[4])
        ):
            raise ValueError("Model source and mounted model path disagree")
        inference = nested["vllm"]
        if inference.port > MAX_TCP_PORT or inference.gpu_memory_utilization > 1:
            raise ValueError("Invalid inference port or memory utilization")
        return cls(**{**value, **nested})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSettings:
    timeout_seconds: int
    max_tool_calls: int
    max_time_minutes: float
    max_turns: int

    @classmethod
    def load(cls, path: Path) -> "EvaluationSettings":
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid evaluation YAML: {path.name}") from error
        if not isinstance(document, dict) or not isinstance(document.get("evaluation"), dict):
            raise ValueError("Expected an evaluation mapping")
        values = document["evaluation"]
        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            raise ValueError(f"Expected evaluation keys: {sorted(expected)}")
        for name, value in values.items():
            valid_type = (
                type(value) in (int, float) if name == "max_time_minutes" else type(value) is int
            )
            if not valid_type or not 0 < value < float("inf"):
                raise ValueError(f"Invalid positive evaluation value: {name}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int
    thinking_config: dict[str, Any]

    @classmethod
    def load(cls, path: Path, runtime: RuntimeSettings) -> "SamplingSettings":
        return cls.parse(yaml.safe_load(path.read_text()), runtime)

    @classmethod
    def parse(cls, raw: Any, runtime: RuntimeSettings) -> "SamplingSettings":
        data = mapping(raw, {f.name for f in fields(cls)}, "sampling")
        positive(data["top_p"], "top_p")
        temperature = data["temperature"]
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("Invalid temperature")
        if data["top_p"] > 1:
            raise ValueError("top_p must not exceed one")
        positive(data["max_output_tokens"], "max output tokens", integer=True)
        if type(data["seed"]) is not int or data["seed"] < 0:
            raise ValueError("Seed must be a nonnegative integer")
        thinking = mapping(
            data["thinking_config"], {"thinking_budget", "include_thoughts"}, "thinking"
        )
        if (
            type(thinking["thinking_budget"]) is not int
            or thinking["thinking_budget"] < 0
            or type(thinking["include_thoughts"]) is not bool
        ):
            raise ValueError("Invalid thinking configuration")
        if thinking["thinking_budget"] > data["max_output_tokens"] or (
            runtime.compaction.token_threshold + data["max_output_tokens"]
            >= runtime.vllm.max_model_len
        ):
            raise ValueError("Generation and compaction budgets exceed the context window")
        return cls(**data)
