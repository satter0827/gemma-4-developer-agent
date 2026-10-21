"""Exercise the selected workflow with real ADK execution and synthetic model replies."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.evaluation.config import EvaluationSettings

SYNTHETIC_VERIFICATION_ATTEMPTS = 2


async def contract(submission: Path) -> dict[str, Any]:
    from adk_submission import ModelRegistry, compile_submission, discover_adapters
    from google.adk.agents.run_config import RunConfig
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig
    from swegemma.context import SwegemmaContext

    settings = EvaluationSettings.load(submission / "eval_config.yaml")
    calls: dict[str, int] = {"repair": 0, "verify": 0}
    requests: list[dict[str, Any]] = []
    commands: list[str] = []
    checks = 0

    class Sandbox:
        def exec(self, container: str, command: str, **kwargs: Any) -> Any:
            nonlocal checks
            commands.append(command)
            if command == "synthetic verify":
                checks += 1
                code = int(checks == 1)
                return SimpleNamespace(
                    exit_code=code,
                    stdout="assertion checked",
                    stderr="failed assertion" if code else "",
                    timed_out=False,
                )
            return SimpleNamespace(
                exit_code=0,
                stdout="synthetic diff" if "git diff" in command else "done",
                stderr="",
                timed_out=False,
            )

    class Synthetic(BaseLlm):  # type: ignore[misc]  # Only installed in the official SDK CPU environment.
        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            instruction = str(llm_request.config.system_instruction)
            stage = "verify" if "Repair handoff:" in instruction else "repair"
            calls[stage] += 1
            contents = " ".join(str(c) for c in llm_request.contents)
            requests.append({"stage": stage, "instruction": instruction, "contents": contents})
            if stage == "repair":
                part = (
                    types.Part.from_function_call(
                        name="run_command", args={"command": "synthetic repair"}
                    )
                    if calls[stage] == 1
                    else types.Part(text="short synthetic handoff")
                )
            elif calls[stage] <= SYNTHETIC_VERIFICATION_ATTEMPTS:
                part = types.Part.from_function_call(
                    name="run_command", args={"command": "synthetic verify"}
                )
            elif calls[stage] == SYNTHETIC_VERIFICATION_ATTEMPTS + 1:
                part = types.Part.from_function_call(name="submit_patch", args={})
            else:
                part = types.Part(text="checked and submitted")
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    model = Synthetic(model="synthetic")
    models = ModelRegistry()
    models.register("gemma-4-31b-it-qat-w4a16-ct", model)
    configuration = EvalConfig(
        models=models,
        tasks_path=submission / "unused.jsonl",
        snapshots_dir=submission,
        results_dir=submission.parent,
        submission_dir=submission,
        **settings.as_dict(),
    )
    ctx = SwegemmaContext(
        budget=configuration.budget, harness=configuration.harness, repo="test/mock_repo"
    )
    ctx.docker = Sandbox()
    tools = ctx.create_tools()
    adapters = discover_adapters(submission, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    agent = compile_submission(
        submission,
        tool_registry=tools,
        model_registry=models,
        adapter_manifest=adapters,
        adapter_resolver_fn=lambda base, info: model,
        limits=configuration.limits,
        generation_constraints=configuration.generation_constraints,
    )
    if agent.__class__.__name__ != "SequentialAgent" or [a.name for a in agent.sub_agents] != [
        "repair_agent",
        "verify_agent",
    ]:
        raise ValueError("Sequential stage order changed")
    repair, verifier = agent.sub_agents

    def names(a: Any) -> set[str]:
        return {t.__name__ if callable(t) else t.name for t in a.tools}

    if (
        "submit_patch" in names(repair)
        or "submit_patch" not in names(verifier)
        or repair.output_key != "repair_summary"
        or verifier.include_contents != "none"
        or any(
            not a.disallow_transfer_to_parent or not a.disallow_transfer_to_peers
            for a in agent.sub_agents
        )
    ):
        raise ValueError("Sequential tool permissions or handoff changed")
    service = InMemorySessionService()
    session = await service.create_session(
        app_name="synthetic",
        user_id="synthetic",
        state={"problem_description": "synthetic original issue"},
    )
    runner = Runner(agent=agent, app_name="synthetic", session_service=service)
    events = [
        e
        async for e in runner.run_async(
            user_id="synthetic",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text="synthetic original issue")]
            ),
            run_config=RunConfig(max_llm_calls=settings.max_turns),
        )
    ]
    first_verify = next(r for r in requests if r["stage"] == "verify")
    if (
        "short synthetic handoff" not in first_verify["instruction"]
        or "synthetic original issue" not in first_verify["instruction"]
        or "synthetic repair" in first_verify["contents"]
        or commands[:3] != ["synthetic repair", "synthetic verify", "synthetic verify"]
        or not ctx.patch_submitted
    ):
        raise ValueError("Sequential history isolation, verification or submission changed")
    used = ctx.tool_calls_used
    ctx.tool_calls_used = settings.max_tool_calls
    exhausted = tools.get("run_command")("must not execute")
    if not exhausted.get("error") and exhausted.get("status") != "error":
        raise ValueError("Shared tool budget was not enforced")
    if "must not execute" in commands or not events:
        raise ValueError("Budget exhaustion executed a tool or produced no events")
    return {
        "status": "completed",
        "stage_order": ["repair_agent", "verify_agent"],
        "calls": calls,
        "tool_calls_used": used,
        "shared_tool_limit": settings.max_tool_calls,
        "shared_time_minutes": settings.max_time_minutes,
        "shared_turn_limit": settings.max_turns,
        "verification_checks": checks,
        "history_isolated": True,
        "submitted": ctx.patch_submitted,
        "scope": "synthetic SDK execution; no model inference or accuracy claim",
    }


def run(submission: Path, output: Path) -> dict[str, Any]:
    report = asyncio.run(contract(submission))
    write_json(output, report)
    return report
