import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gemma_agent_dev.evaluation.official import (
    create_server,
    expected_model,
    make_validator,
    validate_model,
)


def test_model_source_and_mount_must_agree() -> None:
    with pytest.raises(ValueError, match="disagree"):
        expected_model(
            dict(model_source="owner/model/Other/expected/2", paths=dict(model="/wrong/2"))
        )


def test_current_compiler_registers_and_resolves_both_loras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def module(name: str, **attributes: Any) -> Any:
        value = types.ModuleType(name)
        for key, item in attributes.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    name = "gemma-4-31b-it-qat-w4a16-ct"
    adapters = SimpleNamespace(
        adapters={key: SimpleNamespace(name=key) for key in ("main_lora", "tool_lora")}
    )
    models = SimpleNamespace(get={key: f"openai/{key}" for key in adapters.adapters}.__getitem__)
    compiled = []

    def compile_submission(**kwargs: Any) -> None:
        assert kwargs["model_registry"] is models
        resolver = kwargs["adapter_resolver_fn"]
        compiled.append(
            [resolver("base", info) for info in kwargs["adapter_manifest"].adapters.values()]
        )
        with pytest.raises(KeyError):
            resolver("base", SimpleNamespace(name="unknown"))

    adk = module(
        "adk_submission",
        discover_declared_models=lambda _: {name},
        discover_adapters=lambda *_, **kw: adapters,
        VllmConfig=lambda **kw: SimpleNamespace(**kw),
        VllmServer=lambda config, **kw: SimpleNamespace(config=config, **kw),
        compile_submission=compile_submission,
        validate_directory=lambda *_, **kw: None,
    )
    module("swegemma")
    module("swegemma.config", ALLOWED_ADAPTER_EXTENSIONS={".safetensors"})
    module(
        "swegemma.context", SwegemmaContext=lambda **kw: SimpleNamespace(create_tools=lambda: {})
    )
    runtime = dict(
        model_source=f"google/gemma-4/Other/{name}/2",
        paths=dict(model=f"/models/{name}/2"),
        vllm=dict(port=8000),
    )
    server = create_server(tmp_path, runtime)
    assert server.adapter_manifest is adapters and server.config.port == 8000
    config = SimpleNamespace(
        budget=None, harness=None, models=models, limits=None, generation_constraints=None
    )
    make_validator(config, name)(tmp_path)
    assert compiled == [["openai/main_lora", "openai/tool_lora"]]
    monkeypatch.setattr(adk, "discover_declared_models", lambda _: {name, "other"})
    with pytest.raises(ValueError, match="Expected model"):
        validate_model(tmp_path, name)


@pytest.mark.parametrize("fault", [None, "implicit", "empty"])
def test_generation_contract_checks_every_llm_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    import types

    from gemma_agent_dev.evaluation.official import generation_contract

    for name in ["agent.yaml", "sub.yaml", "sampling.yaml"]:
        (tmp_path / name).touch()
    data: dict[str, Any] = {
        "agent.yaml": {"agent_class": "SequentialAgent"},
        "sub.yaml": {"model": "model", "generate_content_config": {"temperature": 0.2}},
        "sampling.yaml": {"temperature": 0.2},
    }
    if fault == "implicit":
        del data["sub.yaml"]["generate_content_config"]
    elif fault == "empty":
        data["sub.yaml"] = {}
    module: Any = types.ModuleType("adk_submission.yaml_loader")
    module.load_yaml = lambda path, root: data[path.name]
    monkeypatch.setitem(sys.modules, "adk_submission.yaml_loader", module)
    if fault:
        with pytest.raises(ValueError):
            generation_contract(tmp_path)
    else:
        assert generation_contract(tmp_path) == [
            {"model": "model", "generation": {"temperature": 0.2}}
        ]


def test_controls_bind_task_specific_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from gemma_agent_dev.evaluation.official import verify_controls

    calls = []

    def module(name: str, **attributes: Any) -> None:
        item = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(item, key, value)
        monkeypatch.setitem(sys.modules, name, item)

    module("swegemma.evaluate", Evaluator=lambda c: SimpleNamespace(sandbox=object()))
    module(
        "swegemma.deduplication",
        resolve_task_snapshot_paths=lambda *a: (tmp_path / "snapshot", None, None),
    )

    async def verify(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            resolved=bool(kwargs["agent_patch"]),
            test_exit_code=0 if kwargs["agent_patch"] else 1,
            error=None,
            duration_seconds=1,
            test_output="synthetic",
        )

    module("swegemma.harness.verification", verify_task=verify)

    def observe(sandbox: Any, sources: Any, *args: Any) -> Any:
        calls.append(sources)
        return lambda: None

    monkeypatch.setattr("gemma_agent_dev.diagnostics.sandbox.observe_sources", observe)
    tasks = [
        SimpleNamespace(instance_id=id, repo="rich", patch="reference", test_patch="tests")
        for id in ["rich_one", "rich_two"]
    ]
    sources = {
        "rich_one": {"dependency_wheels": {"pygments.whl": "one"}},
        "rich_two": {"dependency_wheels": {}},
    }
    assert asyncio.run(
        verify_controls(
            SimpleNamespace(task_ids=["rich_two", "rich_one"], snapshots_dir=tmp_path),
            tasks,
            tmp_path,
            sources,
        )
    )
    assert calls == [{"rich_two": sources["rich_two"]}, {"rich_one": sources["rich_one"]}]
    assert [
        r["instance_id"] for r in __import__("json").loads((tmp_path / "controls.json").read_text())
    ] == ["rich_two", "rich_one"]
