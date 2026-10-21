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
