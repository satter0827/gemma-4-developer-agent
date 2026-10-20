import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_coverage.py"
CHECKER = runpy.run_path(str(SCRIPT))


def report(lines: int = 90, branches: int = 80) -> dict[str, Any]:
    return {
        "meta": {"branch_coverage": True},
        "totals": {
            "covered_lines": lines,
            "num_statements": 100,
            "covered_branches": branches,
            "num_branches": 100,
            "percent_covered": 99.99,
        },
    }


@pytest.mark.parametrize(
    "lines,branches,passed", [(90, 80, True), (89, 100, False), (100, 79, False)]
)
def test_coverage_metrics_have_independent_minimums(
    lines: int, branches: int, passed: bool
) -> None:
    result = CHECKER["evaluate"](report(lines, branches), {"line": 90, "branch": 80})
    assert all(row[3] for row in result) is passed


def test_coverage_does_not_round_up_to_the_minimum() -> None:
    data = report()
    data["totals"].update(covered_lines=899999, num_statements=1000000)
    result = CHECKER["evaluate"](data, {"line": 90, "branch": 80})
    assert f"{result[0][1]:.2f}" == "90.00" and not result[0][3]


@pytest.mark.parametrize("value", [0, -1, 101, True, "90", float("nan"), float("inf")])
def test_invalid_coverage_minimum_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError):
        CHECKER["evaluate"](report(), {"line": value, "branch": 80})


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_statements", 0),
        ("covered_lines", True),
        ("covered_branches", 101),
        ("num_branches", 0),
    ],
)
def test_missing_or_invalid_measurements_are_not_success(field: str, value: Any) -> None:
    data = report()
    data["totals"][field] = value
    with pytest.raises(ValueError):
        CHECKER["evaluate"](data, {"line": 90, "branch": 80})


@pytest.mark.parametrize(
    "branch_measured,minimums", [(False, {"line": 90, "branch": 80}), (True, {"line": 90})]
)
def test_both_metrics_must_be_present(branch_measured: bool, minimums: dict[str, int]) -> None:
    data = report()
    data["meta"]["branch_coverage"] = branch_measured
    with pytest.raises(ValueError):
        CHECKER["evaluate"](data, minimums)


@pytest.mark.parametrize("branches,code", [(80, 0), (79, 1)])
def test_coverage_cli_writes_summary_and_returns_failure_when_under_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, branches: int, code: int
) -> None:
    config = tmp_path / "pyproject.toml"
    config.write_text(
        '[tool.coverage.json]\noutput="coverage.json"\n[tool.coverage_gate]\nline=90\nbranch=80\n'
    )
    (tmp_path / "coverage.json").write_text(json.dumps(report(branches=branches)))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(config)])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exit.value.code == code
    assert (tmp_path / "summary.md").is_file()
    assert ("FAIL" in (tmp_path / "summary.md").read_text()) is bool(code)


def test_missing_coverage_data_fails_the_cli(tmp_path: Path) -> None:
    config = tmp_path / "pyproject.toml"
    config.write_text('[tool.coverage.json]\noutput="missing.json"\n')
    assert CHECKER["main"](["--config", str(config)]) == 1
