"""Exercise the real compiler and request conversion with synthetic inputs only."""

import asyncio
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import write_json
from gemma_agent_dev.config import RuntimeSettings, SamplingSettings
from gemma_agent_dev.official import create_server, expected_model, make_validator


async def request_contract(sampling: dict[str, Any]) -> dict[str, Any]:
    from adk_submission.resolvers.generation import apply_thinking_config_to_model
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from litellm import ModelResponse

    calls = []

    class Capture(LiteLLMClient):  # type: ignore[misc]  # SDK exists only in its CPU environment.
        async def acompletion(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return ModelResponse(
                id="synthetic",
                model="synthetic",
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning_content": "synthetic thought",
                        },
                        "finish_reason": "stop",
                    }
                ],
            )

    model = LiteLlm(model="openai/synthetic", llm_client=Capture(), api_key="EMPTY")
    config = types.GenerateContentConfig(**sampling)
    model = apply_thinking_config_to_model(model, config)
    request = LlmRequest(
        config=config,
        contents=[
            types.Content(role="user", parts=[types.Part(text="Verify a synthetic calculation.")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(text="retained thought", thought=True),
                    types.Part(text="working"),
                ],
            ),
            types.Content(role="user", parts=[types.Part(text="Continue.")]),
        ],
    )
    responses = [response async for response in model.generate_content_async(request)]
    outgoing = calls[0]
    thinking = sampling["thinking_config"]
    body = outgoing["extra_body"]
    enabled = thinking["thinking_budget"] > 0
    if (
        body["chat_template_kwargs"]["enable_thinking"] != enabled
        or outgoing["seed"] != sampling["seed"]
    ):
        raise ValueError("SDK changed seed or thinking enablement")
    if enabled and body["thinking_token_budget"] != thinking["thinking_budget"]:
        raise ValueError("SDK changed thinking budget")
    retained = any(m.get("reasoning") == "retained thought" for m in outgoing["messages"])
    if retained != thinking["include_thoughts"] or not responses:
        raise ValueError("SDK changed thought retention or returned no response")
    return {"seed": outgoing["seed"], "extra_body": body, "retained_thoughts": retained}


def run(runtime_path: Path, sampling_path: Path, output: Path) -> None:
    from swegemma.config import EvalConfig

    runtime = RuntimeSettings.load(runtime_path)
    versions = {name: importlib.metadata.version(name) for name in runtime.packages}
    if versions != runtime.packages:
        raise ValueError("Official CPU SDK versions changed")
    sampling = SamplingSettings.load(sampling_path, runtime)
    source = output / "synthetic"
    source.mkdir()
    (source / "agent.yaml").write_text(
        f"name: synthetic\nmodel: {expected_model(runtime.as_dict())}\n"
        "adapter: main_lora\ninstruction: Verify a synthetic calculation.\n"
        "tools: [run_command]\n"
    )
    for alias in ("main_lora", "tool_lora"):
        folder = source / "adapters" / alias
        folder.mkdir(parents=True)
        write_json(
            folder / "adapter_config.json",
            {
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["q_proj"],
                "peft_type": "LORA",
                "base_model_name_or_path": expected_model(runtime.as_dict()),
            },
        )
        (folder / "adapter_model.safetensors").write_bytes(b"synthetic-compilation-only")
    server = create_server(source, runtime.as_dict())
    try:
        models = server.create_model_registry(
            aliases=[expected_model(runtime.as_dict())], model_prefix="openai/", api_key="EMPTY"
        )
        for name in (expected_model(runtime.as_dict()), "main_lora", "tool_lora"):
            models.get(name)
        config = EvalConfig(
            models=models,
            submission_dir=source,
            tasks_path=output / "unused.jsonl",
            snapshots_dir=output,
            results_dir=output,
        )
        make_validator(config, expected_model(runtime.as_dict()))(source)
        request = asyncio.run(request_contract({**sampling.__dict__}))
        gpu_modules = [name for name in ("torch", "torchvision", "vllm") if name in sys.modules]
        if gpu_modules:
            raise ValueError("CPU compilation imported GPU libraries")
        write_json(
            output / "report.json",
            {
                "status": "completed",
                "gpu_modules": gpu_modules,
                "packages": versions,
                "request": request,
                "scope": "synthetic compilation and request conversion; no weights or inference",
            },
        )
    finally:
        server.stop()


if __name__ == "__main__":
    run(*(Path(p) for p in sys.argv[1:]))
