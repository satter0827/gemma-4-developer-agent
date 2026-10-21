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
