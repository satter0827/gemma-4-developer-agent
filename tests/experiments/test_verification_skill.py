import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gemma_agent_dev.candidates.archive import referenced_files

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "agent/skills/python-verification-v1/scripts/check.py"
SKILL = ModuleType("verification_skill")
SKILL.__file__ = str(SCRIPT)
exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), SKILL.__dict__)
POLICY = {"max_failures": 2, "max_message_characters": 40}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/calc.py").write_text("VALUE = 0\n")
    (tmp_path / "tests/test_calc.py").write_text("def test_value():\n    assert True\n")
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "baseline",
        ],
    ):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src/calc.py").write_text("VALUE = 1\n")
    return tmp_path


@pytest.mark.parametrize("case", ["passed", "failed", "empty", "unchanged", "skipped", "timeout"])
def test_actual_focused_tests(workspace: Path, case: str) -> None:
    test = workspace / "tests/test_calc.py"
    timeout = 20.0
    if case == "failed":
        test.write_text("def test_value():\n    assert False, 'failure detail'\n")
    elif case == "empty":
        test.write_text("")
    elif case == "unchanged":
        (workspace / "src/calc.py").write_text("VALUE = 0\n")
        (workspace / "repro.py").write_text("print('temporary')\n")
    elif case == "skipped":
        test.write_text("import pytest\n@pytest.mark.skip\ndef test_value():\n    pass\n")
    elif case == "timeout":
        test.write_text("import time\ndef test_value():\n    time.sleep(30)\n")
        timeout = 1.0
    report, code = SKILL.run(workspace, ["tests/test_calc.py"], ["src"], timeout, POLICY)
    assert (code == 0) is (case == "passed")
    assert report["status"] == (
        "passed" if case == "passed" else "timeout" if case == "timeout" else "failed"
    )
    if case == "failed":
        assert report["failed"] == 1 and report["failures"]
    if case == "empty":
        assert report["collected"] == 0
    if case == "unchanged":
        assert report["source_changed"] is False


def test_deadline_and_workspace_paths(workspace: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        SKILL.remaining(0)
    for path in ("../outside", "--help", "/nonexistent", "missing"):
        with pytest.raises(ValueError):
            SKILL.inside(workspace, path)
    with pytest.raises(ValueError, match="Positive"):
        SKILL.run(workspace, ["tests/test_calc.py"], ["src"], 0, POLICY)
    with pytest.raises(ValueError, match="Positive"):
        SKILL.run(workspace, ["tests/test_calc.py"], ["src"], 10, {"max_failures": 0})


def test_main_returns_infrastructure_error_without_raw_exception(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--workspace",
            str(workspace),
            "--test",
            "missing",
            "--source-root",
            "src",
            "--timeout-seconds",
            "20",
        ],
    )
    assert SKILL.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "infrastructure_error",
        "exception_type": "ValueError",
    }


@pytest.mark.parametrize("damage", ["missing", "link", "outside", "empty", "type"])
def test_skill_closure_rejects_missing_linked_and_external_resources(
    tmp_path: Path, damage: str
) -> None:
    (tmp_path / "skills/check/scripts").mkdir(parents=True)
    (tmp_path / "skills/check/SKILL.md").write_text("instructions")
    (tmp_path / "skills/check/scripts/check.py").write_text("pass\n")
    refs: Any = ["skills/check"]
    if damage == "missing":
        (tmp_path / "skills/check/SKILL.md").unlink()
    elif damage == "link":
        (tmp_path / "skills/check/scripts/linked.py").symlink_to(SCRIPT)
    elif damage == "outside":
        refs = ["../external"]
    elif damage == "empty":
        refs = []
    else:
        refs = "skills/check"
    (tmp_path / "agent.yaml").write_text("name: test\nskills: " + json.dumps(refs))
    with pytest.raises(ValueError):
        referenced_files(tmp_path, tmp_path / "agent.yaml")


def test_skill_assets_are_exact_candidate_inputs() -> None:
    files = referenced_files(ROOT / "agent", ROOT / "agent/workflow-skills-v1.yaml")
    assert set(name for name in files if name.startswith("skills/")) == {
        "skills/python-verification-v1/SKILL.md",
        "skills/python-verification-v1/assets/policy.json",
        "skills/python-verification-v1/scripts/check.py",
    }


@pytest.mark.parametrize("fault", ["none", "empty", "wrong-outcome", "budget"])
def test_sdk_contract_rejects_changed_execution_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import types
    from types import SimpleNamespace

    from gemma_agent_dev.evaluation import skill_checks

    def module(name: str, **attributes: Any) -> None:
        for i in range(1, len(name.split(".")) + 1):
            key = ".".join(name.split(".")[:i])
            if key not in sys.modules:
                value = types.ModuleType(key)
                value.__path__ = []
                monkeypatch.setitem(sys.modules, key, value)
        value = types.ModuleType(name)
        for key, item in attributes.items():
            setattr(value, key, item)
        monkeypatch.setitem(sys.modules, name, value)

    class Context:
        tool_calls_used = 0

        def __init__(self, **kwargs: Any):
            self.limit = kwargs["max_tool_calls"]

        def check_budget(self) -> str | None:
            return "exhausted" if self.tool_calls_used >= self.limit else None

    class Runner:
        def __init__(self, executor: Any, **kwargs: Any):
            self.executor = executor

        async def execute_script_async(self, *args: Any) -> dict[str, Any]:
            exhausted = self.executor.budget_check_fn()
            if exhausted and fault != "budget":
                return {"stdout": "", "stderr": exhausted}
            workspace = Path(args[-1][1])
            changed = (workspace / "src/calc.py").read_text() == "VALUE = 1\n"
            passed = changed and fault != "wrong-outcome"
            return dict(
                status="success" if passed else "warning",
                stdout=json.dumps(
                    dict(
                        status="passed" if passed else "failed",
                        source_changed=changed,
                        collected=0 if fault == "empty" else 1,
                    )
                ),
                stderr="",
            )

    module(
        "adk_eval_core.sandbox",
        AdkSandboxCodeExecutor=lambda **kwargs: SimpleNamespace(**kwargs),
        SubprocessSandbox=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    module("google.adk.skills", load_skill_from_dir=lambda _: SimpleNamespace())
    module("google.adk.tools.skill_toolset", _SkillScriptCodeExecutor=Runner)
    module("swegemma.context", SwegemmaContext=Context)
    monkeypatch.setattr(skill_checks, "command", lambda *a, **kw: "")
    if fault == "none":
        report = skill_checks.run(ROOT / "agent", tmp_path, 30)
        assert report["shared_budget_enforced"] and (tmp_path / "contract.json").exists()
    else:
        with pytest.raises(ValueError):
            skill_checks.run(ROOT / "agent", tmp_path, 30)
