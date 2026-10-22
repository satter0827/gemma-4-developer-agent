import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_verifier(version: str) -> tuple[ModuleType, Path, dict[str, int]]:
    filename = "issue_verification.py" if version == "v1" else "issue_verification_v2.py"
    script = ROOT / f"agent/skills/issue-workflow-{version}/scripts" / filename
    skill = ModuleType("focused_skill_" + version)
    skill.__file__ = str(script)
    exec(compile(script.read_text(), str(script), "exec"), skill.__dict__)
    return skill, script, json.loads((script.parents[1] / "assets/policy.json").read_text())


@pytest.fixture(params=("v1", "v2"))
def verifier(request: pytest.FixtureRequest) -> tuple[ModuleType, Path, dict[str, int]]:
    return load_verifier(request.param)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/calc.py").write_text("VALUE = 0\n")
    (tmp_path / "src/template.html").write_text("before\n")
    for args in [
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
    ]:
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src/calc.py").write_text("VALUE = 1\n")
    return tmp_path


@pytest.mark.parametrize(
    "case",
    [
        "passed",
        "failed",
        "empty",
        "unchanged",
        "scratch_only",
        "wrong_origin",
        "wrong_origin_inside",
        "missing_import",
        "non_python",
        "timeout",
    ],
)
def test_actual_tests_diff_and_origins_in_one_process(
    workspace: Path, case: str, verifier: tuple[ModuleType, Path, dict[str, int]]
) -> None:
    skill, _, policy = verifier
    source = "from src.calc import VALUE\ndef test_value():\n    assert VALUE == 1\n"
    modules = ["src.calc"]
    timeout = 20
    if case == "failed":
        source = source.replace("== 1", "== 2")
    elif case == "empty":
        source = ""
    elif case == "unchanged":
        (workspace / "src/calc.py").write_text("VALUE = 0\n")
        source = source.replace("== 1", "== 0")
        (workspace / "tests/repro.py").write_text("assert True\n")
    elif case == "scratch_only":
        (workspace / "src/calc.py").write_text("VALUE = 0\n")
        (workspace / "src/__pycache__").mkdir()
        (workspace / "src/__pycache__/generated.pyc").write_bytes(b"cache")
        (workspace / "src/repro.json").write_text("{}")
        (workspace / "src/conftest.py").write_text("# test helper")
        source = source.replace("== 1", "== 0")
    elif case == "wrong_origin":
        source = 'import json\ndef test_value():\n    assert json.loads("1") == 1\n'
        modules = ["json"]
    elif case == "wrong_origin_inside":
        alternate = workspace / "alternate/src"
        alternate.mkdir(parents=True)
        (alternate / "calc.py").write_text("VALUE = 1\n")
        source = f"import sys\nsys.path.insert(0, {str(alternate.parent)!r})\n" + source
    elif case == "missing_import":
        source = "def test_value():\n    assert True\n"
    elif case == "non_python":
        (workspace / "src/calc.py").write_text("VALUE = 0\n")
        (workspace / "src/template.html").write_text("after\n")
        source = source.replace("== 1", "== 0")
    elif case == "timeout":
        source = "import time\ndef test_value():\n    time.sleep(30)\n"
        timeout = 1
    (workspace / "tests/test_calc.py").write_text(source)
    report, code = skill.run(workspace, ["tests/test_calc.py"], ["src"], timeout, policy, modules)
    assert (code == 0) == (case in {"passed", "non_python"})
    if case == "passed":
        assert report["origins"]["src.calc"] == str(workspace / "src/calc.py")
    if case == "non_python":
        assert report["source_paths"] == ["src/template.html"]
    if case in {"wrong_origin", "wrong_origin_inside"}:
        assert report["unexpected_origins"]
    if case == "missing_import":
        assert report["missing_modules"] == ["src.calc"]
    if case == "timeout":
        assert report["status"] == "timeout"
    if case == "unchanged":
        assert not report["source_changed"]


def test_invalid_module_and_cli_failure(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    verifier: tuple[ModuleType, Path, dict[str, int]],
) -> None:
    skill, script, policy = verifier
    (workspace / "tests/test_calc.py").write_text("")
    for modules in [[], ["bad-name"]]:
        with pytest.raises(ValueError):
            skill.run(workspace, ["tests/test_calc.py"], ["src"], 20, policy, modules)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--workspace",
            str(workspace),
            "--test",
            "missing",
            "--source-root",
            "src",
            "--module",
            "src.calc",
            "--timeout-seconds",
            "20",
        ],
    )
    assert skill.main() == 1
    assert json.loads(capsys.readouterr().out)["status"] == "infrastructure_error"


@pytest.mark.parametrize("case", ["collected_suffix", "uncollected_suffix", "configured_pattern"])
def test_v2_excludes_test_only_changes_using_pytest_configuration(
    workspace: Path, case: str
) -> None:
    skill, _, policy = load_verifier("v2")
    (workspace / "src/calc.py").write_text("VALUE = 0\n")
    source = "from src.calc import VALUE\ndef test_value():\n    assert VALUE == 0\n"
    (workspace / "tests/test_calc.py").write_text(source)
    name = "src/check_calc.py" if case == "configured_pattern" else "src/calc_test.py"
    (workspace / name).write_text(source)
    if case == "configured_pattern":
        (workspace / "pytest.ini").write_text("[pytest]\npython_files = check_*.py\n")
    tests = [name] if case == "collected_suffix" else ["tests/test_calc.py"]
    report, code = skill.run(workspace, tests, ["src"], 20, policy, ["src.calc"])
    assert code != 0 and report["status"] == "failed"
    assert report["collected"] == report["passed"] == 1
    assert report["source_changed"] is False and report["source_paths"] == []
    assert report["origins"]["src.calc"] == str(workspace / "src/calc.py")
