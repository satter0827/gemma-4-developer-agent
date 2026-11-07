"""Exercise the pinned SDK's Skill materialization and sandbox executor on CPU."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import declared_skills
from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.events import event, logged
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.evaluation.config import EvaluationSettings


async def contract(
    submission: Path, output: Path, timeout_seconds: int, skill_path: Path
) -> dict[str, Any]:
    from adk_eval_core.sandbox import AdkSandboxCodeExecutor, SubprocessSandbox
    from google.adk.skills import load_skill_from_dir
    from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor
    from swegemma.context import SwegemmaContext

    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src/calc.py").write_text("VALUE = 0\n")
    (workspace / "src/template.html").write_text("before\n")
    test = workspace / "tests/test_calc.py"
    focused_test = "from src.calc import VALUE\ndef test_value():\n    assert VALUE == 1\n"
    test.write_text(focused_test)
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
    skill = load_skill_from_dir(skill_path)
    declaration = skill_path / "assets/verification.json"
    declared: dict[str, Any] = (
        json.loads(declaration.read_text())
        if declaration.exists()
        else {"script": "scripts/check.py", "module_argument": None}
    )
    script = declared["script"]
    if (
        not (skill_path / script).resolve().is_relative_to(skill_path.resolve())
        or not (skill_path / script).is_file()
    ):
        raise ValueError("Declared Skill verification script is missing or external")
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
    if declared["module_argument"]:
        args.extend([declared["module_argument"], "src.calc"])
    results = {}
    cases: tuple[tuple[str, str, str, int], ...] = (
        ("passed", "VALUE = 1\n", focused_test, 1),
        ("failed", "VALUE = 200\n", focused_test, 1),
        ("zero_tests", "VALUE = 1\n", "# No test functions\n", 0),
        ("unchanged", "VALUE = 0\n", "def test_smoke():\n    assert True\n", 1),
    )
    if declared["module_argument"]:
        foreign = output / "foreign/src"
        foreign.mkdir(parents=True)
        (foreign / "calc.py").write_text("VALUE = 1\n")
        outside_test = f"import sys\nsys.path.insert(0, {str(foreign.parent)!r})\n" + focused_test
        cases += (
            ("wrong_origin", "VALUE = 1\n", outside_test, 1),
            ("non_python", "VALUE = 0\n", focused_test.replace("== 1", "== 0"), 1),
            ("colocated_test", "VALUE = 0\n", focused_test.replace("== 1", "== 0"), 1),
        )
    for label, source, test_source, collected in cases:
        (workspace / "src/template.html").write_text(
            "after\n" if label == "non_python" else "before\n"
        )
        (workspace / "src/calc.py").write_text(source)
        test.write_text(test_source)
        case_args = list(args)
        if label == "colocated_test":
            (workspace / "src/calc_test.py").write_text(test_source)
            case_args[case_args.index("--test") + 1] = "src/calc_test.py"
        result = await runner.execute_script_async(None, skill, script, case_args)
        report = json.loads(result.get("stdout", ""))
        if (report["status"] == "passed") != (label in {"passed", "non_python"}) or report[
            "collected"
        ] != collected:
            raise ValueError(f"Pinned SDK Skill case {label} returned {report}")
        results[label] = {
            "tool_status": result["status"],
            "status": report["status"],
            "source_changed": report["source_changed"],
            "collected": report["collected"],
            "origins": report.get("origins"),
            "test_exit_code": report.get("test_exit_code"),
        }
        event(
            "skill.contract",
            stage=label,
            count=report["collected"],
            result=report["status"],
        )
    (workspace / "src/calc.py").write_text("VALUE = 1\n")
    test.write_text(focused_test)
    from gemma_agent_dev.evaluation.skill_chain import execute

    chain = await execute(
        submission,
        workspace,
        executor,
        timeout_seconds,
        skill.frontmatter.name,
        script,
        args,
        declared.get("resources", []),
    )

    if chain.get("resource_sha256", {}) != {
        name: file_hash(skill_path / name) for name in declared.get("resources", [])
    }:
        raise ValueError("Compiled Skill resources differ from the candidate")
    context.tool_calls_used = settings.max_tool_calls
    blocked = await runner.execute_script_async(None, skill, script, args)
    if blocked.get("stdout") or not blocked.get("stderr"):
        raise ValueError("Shared SDK budget did not block Skill execution")
    return {
        "status": "completed",
        "scope": "synthetic SDK sandbox; no model inference or accuracy claim",
        "cases": results,
        "compiled_agent_chain": chain,
        "shared_budget_enforced": True,
    }


@logged("skill.verify")
def run(
    submission: Path, output: Path, timeout_seconds: int, *, skill_path: Path | None = None
) -> dict[str, Any]:
    paths = [skill_path] if skill_path is not None else declared_skills(submission)
    if not paths:
        raise ValueError("Candidate declares no Skill to verify")
    reports = {
        path.name: asyncio.run(contract(submission, output / path.name, timeout_seconds, path))
        for path in paths
    }
    report = (
        next(iter(reports.values()))
        if len(reports) == 1
        else {"status": "completed", "skills": reports}
    )
    write_json(output / "contract.json", report)
    return report
