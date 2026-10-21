import asyncio
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.evaluation import sdk_checks as runtime_checks
from gemma_agent_dev.evaluation.config import RuntimeSettings

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_checks_prepare_is_separate_and_normal_check_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    (root / "configs/evaluation").mkdir(parents=True)
    write_json(
        root / "configs/evaluation/runtime.yaml",
        RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml").as_dict(),
    )
    for name in ("requirements.in", "requirements.lock", "sampling.yaml"):
        (root / name).write_text("fixture")
    config = {
        "runtime_config": "configs/evaluation/runtime.yaml",
        "requirements": "requirements.in",
        "lock_file": "requirements.lock",
        "environment": "artifacts/sdk",
        "wheel_cache": "artifacts/wheels",
        "output_dir": "artifacts/checks",
        "timeout_seconds": 10,
        "python_version": "3.12",
        "sampling": "sampling.yaml",
    }
    shutil.copytree(ROOT / "src", root / "src")
    path = root / "checks.yaml"
    path.write_text(yaml.safe_dump(config))
    runtime = json.loads((root / "configs/evaluation/runtime.yaml").read_text())
    runtime["wheelhouse_hashes"] = {
        f"{name.replace('-', '_')}-{v}-py3-none-any.whl": __import__("hashlib")
        .sha256(b"wheel")
        .hexdigest()
        for name, v in runtime["packages"].items()
    }
    write_json(root / "configs/evaluation/runtime.yaml", runtime)
    calls = []

    def command(args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        if "-c" in args:
            return json.dumps({"python": "3.12.1", "packages": {"fixture": "1"}})
        if args[0] == "kaggle":
            (Path(args[-2]) / args[5]).write_bytes(b"wheel")
        if args[:2] == ["uv", "venv"]:
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        if "-m" in args:
            write_json(Path(args[-1]) / "report.json", {"status": "completed", "gpu_modules": []})
        return "ok"

    monkeypatch.setattr(runtime_checks, "command", command)
    receipt = runtime_checks.prepare(root, path)
    assert receipt.is_file()
    first_downloads = sum(c[0] == "kaggle" for c in calls)
    runtime_checks.prepare(root, path)
    assert first_downloads == 3 and sum(c[0] == "kaggle" for c in calls) == 3
    calls.clear()
    result = runtime_checks.check(root, path)
    assert (result / "check.log").read_text() == "ok" and all(c[0] != "kaggle" for c in calls)
    prepared_bytes = receipt.read_bytes()
    (root / "sampling.yaml").write_text("different sampling")
    runtime["verification_timeout_seconds"] += 1
    write_json(root / "configs/evaluation/runtime.yaml", runtime)
    calls.clear()
    assert runtime_checks.check(root, path).is_dir()
    assert receipt.read_bytes() == prepared_bytes
    assert not any(c[0] in {"uv", "kaggle"} for c in calls)
    (root / "requirements.lock").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runtime_checks.check(root, path)


@pytest.mark.parametrize("fault", ["none", "seed", "budget", "retention", "disabled"])
def test_cpu_request_contract_rejects_changed_wire_behavior(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import types

    from gemma_agent_dev.evaluation.sdk_worker import request_contract

    captured: dict[str, Any] = {}

    class Client:
        pass

    class Model:
        def __init__(self, **kwargs: Any) -> None:
            self.client = kwargs["llm_client"]

        async def generate_content_async(self, request: Any) -> Any:
            config = request.config.__dict__
            thinking = config["thinking_config"]
            kwargs = {
                "seed": 1 if fault == "seed" else config["seed"],
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": thinking["thinking_budget"] > 0},
                    "thinking_token_budget": 1
                    if fault == "budget"
                    else thinking["thinking_budget"],
                },
                "messages": [{"reasoning": "retained thought"}]
                if thinking["include_thoughts"] and fault != "retention"
                else [],
            }
            captured.update(kwargs)
            await self.client.acompletion(**kwargs)
            yield {"synthetic": True}

    def module(name: str, **members: Any) -> None:
        value = types.ModuleType(name)
        for key, item in members.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)

    module(
        "adk_submission.resolvers.generation",
        apply_thinking_config_to_model=lambda model, cfg: model,
    )
    module("google.adk.models.lite_llm", LiteLlm=Model, LiteLLMClient=Client)
    module("google.adk.models.llm_request", LlmRequest=lambda **kw: SimpleNamespace(**kw))
    module(
        "google.genai",
        types=SimpleNamespace(
            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw),
            Content=lambda **kw: SimpleNamespace(**kw),
            Part=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    module("litellm", ModelResponse=lambda **kw: kw)
    sampling = {
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 0,
        "max_output_tokens": 16384,
        "thinking_config": {
            "thinking_budget": 0 if fault == "disabled" else 4096,
            "include_thoughts": False if fault == "disabled" else True,
        },
    }
    if fault in {"seed", "budget", "retention"}:
        with pytest.raises(ValueError):
            asyncio.run(request_contract(sampling))
    else:
        report = asyncio.run(request_contract(sampling))
        assert report["seed"] == 0 and bool(report["retained_thoughts"]) is (fault != "disabled")
    assert captured


@pytest.mark.parametrize("gpu_imported", [False, True])
def test_cpu_compilation_checks_all_aliases_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu_imported: bool
) -> None:
    import types

    from gemma_agent_dev.evaluation import sdk_worker as official_cpu

    module = types.ModuleType("swegemma.config")
    module.EvalConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swegemma.config", module)
    runtime = RuntimeSettings.load(ROOT / "configs/evaluation/runtime.yaml")
    aliases = []
    stopped = []
    server = SimpleNamespace(
        create_model_registry=lambda **kw: SimpleNamespace(get=lambda name: aliases.append(name)),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setattr(official_cpu, "create_server", lambda *args: server)
    monkeypatch.setattr(official_cpu, "make_validator", lambda *args: lambda source: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: runtime.packages[name])

    async def request(*args: Any) -> dict[str, Any]:
        return {"seed": 0}

    monkeypatch.setattr(official_cpu, "request_contract", request)
    if gpu_imported:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        with pytest.raises(ValueError, match="GPU"):
            official_cpu.run(
                ROOT / "configs/evaluation/runtime.yaml",
                ROOT / "agent/configs/sampling.yaml",
                tmp_path,
            )
    else:
        official_cpu.run(
            ROOT / "configs/evaluation/runtime.yaml", ROOT / "agent/configs/sampling.yaml", tmp_path
        )
        assert json.loads((tmp_path / "report.json").read_text())["status"] == "completed"
    assert len(aliases) == 3 and stopped == [True]
