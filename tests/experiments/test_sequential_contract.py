"""Exercise the real pinned SDK when its separately prepared CPU environment exists."""

import json
import os
import subprocess
from pathlib import Path

import coverage
import pytest

from gemma_agent_dev.candidates.archive import extract_archive
from gemma_agent_dev.common.artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]
SDK_PYTHON = ROOT / "artifacts/runtime-env/bin/python"


@pytest.mark.skipif(not SDK_PYTHON.exists(), reason="Official SDK CPU environment is not prepared")
def test_real_sequential_contract_in_cpu_environment(
    frozen_experiment: Path, tmp_path: Path
) -> None:
    submission = tmp_path / "sequential"
    extract_archive(frozen_experiment.parent / "sequential-v1/submission.zip", submission, 10000000)
    # Compilation uses synthetic adapters and synthetic model replies, never weights or inference.
    write_json(
        submission / "adapters/main_lora/adapter_config.json",
        {
            "r": 1,
            "lora_alpha": 1,
            "target_modules": ["q_proj"],
            "peft_type": "LORA",
            "base_model_name_or_path": "gemma-4-31b-it-qat-w4a16-ct",
        },
    )
    script = """import importlib.util,sys
from pathlib import Path
# Load only coverage itself; do not mix the development environment's other dependencies.
coverage_root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('coverage',coverage_root/'__init__.py',submodule_search_locations=[str(coverage_root)])
module=importlib.util.module_from_spec(spec);sys.modules['coverage']=module;spec.loader.exec_module(module)
cov=module.Coverage(config_file=sys.argv[2],data_file=sys.argv[3]);cov.start()
from gemma_agent_dev.evaluation.session_worker import deny_external
sys.addaudithook(deny_external)
from gemma_agent_dev.evaluation.sequential_checks import run
try: run(Path(sys.argv[4]),Path(sys.argv[5]))
finally: cov.stop();cov.save()
"""
    data = tmp_path / "sdk.coverage"
    report = tmp_path / "contract.json"
    result = subprocess.run(
        [
            str(SDK_PYTHON),
            "-c",
            script,
            str(Path(coverage.__file__).parent),
            str(ROOT / "pyproject.toml"),
            str(data),
            str(submission),
            str(report),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "OTEL_SDK_DISABLED": "true",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    current = coverage.Coverage.current()
    if current is not None:
        child = coverage.CoverageData(basename=str(data))
        child.read()
        current.get_data().update(child)
    output = json.loads(report.read_text())
    assert output["status"] == "completed" and output["history_isolated"] and output["submitted"]
    assert output["verification_checks"] == 2 and output["tool_calls_used"] == 3
    assert (
        output["shared_tool_limit"] == 10
        and output["shared_time_minutes"] == 1
        and output["shared_turn_limit"] == 50
    )


def run_instrumented(python: Path, script: str, arguments: list[str], tmp_path: Path) -> None:
    data = tmp_path / "child.coverage"
    bootstrap = """import importlib.util,sys
from pathlib import Path
coverage_root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('coverage',coverage_root/'__init__.py',submodule_search_locations=[str(coverage_root)])
module=importlib.util.module_from_spec(spec);sys.modules['coverage']=module;spec.loader.exec_module(module)
cov=module.Coverage(config_file=sys.argv[2],data_file=sys.argv[3]);cov.start()
try: exec(sys.argv[4])
finally: cov.stop();cov.save()
"""
    result = subprocess.run(
        [
            str(python),
            "-c",
            bootstrap,
            str(Path(coverage.__file__).parent),
            str(ROOT / "pyproject.toml"),
            str(data),
            script,
            *arguments,
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "OTEL_SDK_DISABLED": "true",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    current = coverage.Coverage.current()
    if current is not None:
        child = coverage.CoverageData(basename=str(data))
        child.read()
        current.get_data().update(child)


@pytest.mark.skipif(not SDK_PYTHON.exists(), reason="Official SDK CPU environment is not prepared")
def test_real_skill_materialization_and_shared_budget(tmp_path: Path) -> None:
    script = """from gemma_agent_dev.evaluation.session_worker import deny_external
sys.addaudithook(deny_external)
from gemma_agent_dev.evaluation.skill_checks import run
run(Path(sys.argv[5]),Path(sys.argv[6]),30)
"""
    run_instrumented(
        SDK_PYTHON, script, [str(ROOT / "agent"), str(tmp_path / "contract")], tmp_path
    )
    report = json.loads((tmp_path / "contract/contract.json").read_text())
    assert report["shared_budget_enforced"] and report["cases"]["passed"]["status"] == "passed"
    assert report["cases"]["unchanged"]["status"] == "failed"


@pytest.mark.skipif(
    not (ROOT / "artifacts/gepa-env/bin/python").exists(),
    reason="GEPA CPU environment is not prepared",
)
@pytest.mark.parametrize("improves", [True, False])
def test_real_gepa_search_with_bounded_synthetic_outcomes(tmp_path: Path, improves: bool) -> None:
    script = r"""import hashlib,time,json
from gemma_agent_dev.evaluation.session_worker import deny_external
sys.addaudithook(deny_external)
from gemma_agent_dev.evaluation.optimization_engine import search
from gemma_agent_dev.evaluation.optimization_limits import SearchLimits,SearchBudget
policy=SearchLimits.parse(json.loads(sys.argv[6]))
values={}
def freeze(text):
    key=hashlib.sha256(text.encode()).hexdigest();values[key]=text;return key
def evaluate(key,task):
    return dict(resolved='improved' in values[key] and sys.argv[7]=='True',
                termination='completed',implementation_changed=None,tool_argument_errors=None)
result=search('seed',['a','b','c'],['d','e','f'],SearchBudget(policy,time.monotonic()+600,0),
              freeze,evaluate,lambda messages:'```\nimproved\n```')
Path(sys.argv[5]).write_text(json.dumps(result))
"""
    from dataclasses import asdict

    import yaml

    from gemma_agent_dev.evaluation.optimization_limits import SearchLimits

    policy = SearchLimits.parse(
        yaml.safe_load((ROOT / "configs/evaluation/easy-gepa-v1.yaml").read_text())["limits"]
    )
    output = tmp_path / "gepa.json"
    run_instrumented(
        ROOT / "artifacts/gepa-env/bin/python",
        script,
        [str(output), json.dumps(asdict(policy)), str(improves)],
        tmp_path,
    )
    result = json.loads(output.read_text())
    assert result["metric_calls"] <= 8 and result["reflection_calls"] == 1
    assert result["improved"] is improves and result["adopted"] is False


@pytest.mark.skipif(not SDK_PYTHON.exists(), reason="Official SDK CPU environment is not prepared")
def test_sdk_deadline_kills_skill_test_descendants(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "tests").mkdir()
    (workspace / "src/calc.py").write_text("VALUE = 0\n")
    for args in [
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
    ]:
        subprocess.run(args, cwd=workspace, capture_output=True, check=True)
    (workspace / "src/calc.py").write_text("VALUE = 1\n")
    marker = workspace / "leaked"
    ready = workspace / "ready"
    child = f"import time;from pathlib import Path;time.sleep(8);Path({str(marker)!r}).touch()"
    (workspace / "tests/test_timeout.py").write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        'def test_deadline():\n    subprocess.Popen([sys.executable,"-c",'
        + repr(child)
        + "])\n    Path("
        + repr(str(ready))
        + ").touch()\n    time.sleep(30)\n"
    )
    script = r"""import asyncio,time
from gemma_agent_dev.evaluation.session_worker import deny_external
sys.addaudithook(deny_external)
from adk_eval_core.sandbox import AdkSandboxCodeExecutor,SubprocessSandbox
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor
workspace=Path(sys.argv[6])
sandbox=SubprocessSandbox(work_dir=workspace,environment={'PATH':str(Path(sys.executable).parent)+':'+__import__('os').environ['PATH']})
executor=AdkSandboxCodeExecutor(sandbox=sandbox,timeout_seconds=3)
runner=_SkillScriptCodeExecutor(executor,script_timeout=3)
result=asyncio.run(runner.execute_script_async(None,load_skill_from_dir(Path(sys.argv[5])/'skills/python-verification-v1'),'scripts/check.py',['--workspace',str(workspace),'--test','tests/test_timeout.py','--source-root','src','--timeout-seconds','30']))
assert result['status']=='error',result
assert (workspace/'ready').exists(),'test child never started'
time.sleep(6)
assert not (workspace/'leaked').exists(),'SDK deadline left a detached test child'
"""
    run_instrumented(SDK_PYTHON, script, [str(ROOT / "agent"), str(workspace)], tmp_path)
