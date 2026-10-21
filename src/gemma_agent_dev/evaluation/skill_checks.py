"""Exercise the pinned SDK's Skill materialization and sandbox executor on CPU."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.common.events import event, logged
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.evaluation.config import EvaluationSettings


async def contract(submission: Path, output: Path, timeout_seconds: int) -> dict[str, Any]:
    from adk_eval_core.sandbox import AdkSandboxCodeExecutor, SubprocessSandbox
    from google.adk.skills import load_skill_from_dir
    from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor
    from swegemma.context import SwegemmaContext

    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src/calc.py").write_text("VALUE = 0\n")
    (workspace / "tests/test_calc.py").write_text("def test_value():\n    assert True\n")
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
    ):
        command(args, cwd=workspace, timeout=timeout_seconds)
    (workspace / "src/calc.py").write_text("VALUE = 1\n")
    settings = EvaluationSettings.load(submission / "eval_config.yaml")
    context = SwegemmaContext(
        max_tool_calls=settings.max_tool_calls, max_time_minutes=settings.max_time_minutes
    )
    sandbox = SubprocessSandbox(
        work_dir=workspace,
        environment={"PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]},
    )
    executor = AdkSandboxCodeExecutor(
        sandbox=sandbox, timeout_seconds=timeout_seconds, budget_check_fn=context.check_budget
    )
    runner = _SkillScriptCodeExecutor(executor, script_timeout=timeout_seconds)
    skill = load_skill_from_dir(submission / "skills/python-verification-v1")
    args = [
        "--workspace",
        str(workspace),
        "--test",
        "tests/test_calc.py",
        "--source-root",
        "src",
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    results = {}
    for label, source in (("passed", "VALUE = 1\n"), ("unchanged", "VALUE = 0\n")):
        (workspace / "src/calc.py").write_text(source)
        result = await runner.execute_script_async(None, skill, "scripts/check.py", args)
        report = json.loads(result.get("stdout", ""))
        if (report["status"] == "passed") != (label == "passed") or report["collected"] != 1:
            raise ValueError("Pinned SDK Skill execution did not preserve verification results")
        results[label] = {
            "tool_status": result["status"],
            "status": report["status"],
            "source_changed": report["source_changed"],
            "collected": report["collected"],
        }
        event(
            "skill.contract",
            stage=label,
            count=report["collected"],
            result=report["status"],
        )
    context.tool_calls_used = settings.max_tool_calls
    blocked = await runner.execute_script_async(None, skill, "scripts/check.py", args)
    if blocked.get("stdout") or not blocked.get("stderr"):
        raise ValueError("Shared SDK budget did not block Skill execution")
    return {
        "status": "completed",
        "scope": "synthetic SDK sandbox; no model inference or accuracy claim",
        "cases": results,
        "shared_budget_enforced": True,
    }


@logged("skill.verify")
def run(submission: Path, output: Path, timeout_seconds: int) -> dict[str, Any]:
    report = asyncio.run(contract(submission, output, timeout_seconds))
    write_json(output / "contract.json", report)
    return report
