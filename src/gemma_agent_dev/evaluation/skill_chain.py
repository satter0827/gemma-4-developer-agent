"""Exercise Skill discovery, loading and execution through the compiled SDK agent."""

import hashlib
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any


async def execute(
    submission: Path,
    workspace: Path,
    executor: Any,
    timeout_seconds: int,
    skill_name: str,
    script: str,
    arguments: list[str],
    resources: list[str],
) -> dict[str, Any]:
    from adk_submission import ModelRegistry, compile_submission, discover_adapters
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig
    from swegemma.context import SwegemmaContext

    from gemma_agent_dev.evaluation.config import EvaluationSettings

    calls = [
        ("list_skills", {}),
        ("load_skill", {"skill_name": skill_name}),
        *[
            ("load_skill_resource", {"skill_name": skill_name, "file_path": path})
            for path in resources
        ],
        ("run_skill_script", {"skill_name": skill_name, "file_path": script, "args": arguments}),
    ]
    responses: list[dict[str, Any]] = []
    requests = 0

    class Synthetic(BaseLlm):  # type: ignore[misc]  # Runs only in the isolated official SDK environment.
        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            nonlocal requests
            part = (
                types.Part.from_function_call(name=calls[requests][0], args=calls[requests][1])
                if requests < len(calls)
                else types.Part(text="CPU Skill contract completed")
            )
            requests += 1
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    settings = EvaluationSettings.load(submission / "eval_config.yaml")
    model = Synthetic(model="synthetic")
    models = ModelRegistry()
    models.register("gemma-4-31b-it-qat-w4a16-ct", model)
    config = EvalConfig(
        models=models,
        tasks_path=submission / "unused.jsonl",
        snapshots_dir=submission,
        results_dir=workspace.parent,
        submission_dir=submission,
        **settings.as_dict(),
    )
    context = SwegemmaContext(budget=config.budget, harness=config.harness, repo="test/mock_repo")
    agent = compile_submission(
        submission,
        tool_registry=context.create_tools(),
        model_registry=models,
        adapter_manifest=discover_adapters(
            submission, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS
        ),
        adapter_resolver_fn=lambda base, info: model,
        limits=config.limits,
        generation_constraints=config.generation_constraints,
        code_executor=executor,
        script_timeout=timeout_seconds,
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name="skill-contract",
        user_id="synthetic",
        state={"problem_description": "synthetic issue"},
    )
    runner = Runner(agent=agent, app_name="skill-contract", session_service=sessions)
    async for event in runner.run_async(
        user_id="synthetic",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="synthetic issue")]),
        run_config=RunConfig(max_llm_calls=settings.max_turns),
    ):
        for part in event.content.parts if event.content else []:
            if part.function_response:
                responses.append(
                    {
                        "name": part.function_response.name,
                        "response": part.function_response.response,
                    }
                )
    if [item["name"] for item in responses] != [name for name, _ in calls]:
        raise ValueError("Compiled agent did not execute the Skill tool chain")
    if any(item["response"].get("error") for item in responses):
        raise ValueError("Compiled agent could not read its declared Skill resources")
    resource_hashes = {
        item["response"]["file_path"]: hashlib.sha256(
            item["response"]["content"].encode()
        ).hexdigest()
        for item in responses
        if item["name"] == "load_skill_resource"
    }
    result = json.loads(responses[-1]["response"]["stdout"])
    if result["status"] != "passed" or result["collected"] != 1:
        raise ValueError("Compiled agent Skill verification failed")
    return {
        "tools": [item["name"] for item in responses],
        "resource_sha256": resource_hashes,
        "status": result["status"],
        "collected": result["collected"],
    }
