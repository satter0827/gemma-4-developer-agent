"""Enforce independent line and branch coverage floors from pyproject.toml."""

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

PERCENT = 100

METRICS = {
    "line": ("covered_lines", "num_statements"),
    "branch": ("covered_branches", "num_branches"),
}


def evaluate(
    report: dict[str, Any], minimums: dict[str, Any]
) -> list[tuple[str, float, float, bool]]:
    if report["meta"]["branch_coverage"] is not True or set(minimums) != set(METRICS):
        raise ValueError("Both line and branch coverage must be configured and measured")
    results = []
    for name, (covered_key, total_key) in METRICS.items():
        covered, total = report["totals"][covered_key], report["totals"][total_key]
        minimum = minimums[name]
        if (
            type(covered) is not int
            or type(total) is not int
            or total <= 0
            or not 0 <= covered <= total
            or type(minimum) not in (int, float)
            or not math.isfinite(minimum)
            or not 0 < minimum <= PERCENT
        ):
            raise ValueError(f"Invalid coverage counts or minimum: {name}")
        results.append(
            (name, PERCENT * covered / total, float(minimum), PERCENT * covered >= minimum * total)
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml"
    )
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = tomllib.loads(config_path.read_text())["tool"]
        report_path = config_path.parent / config["coverage"]["json"]["output"]
        report = json.loads(report_path.read_text())
        results = evaluate(report, config["coverage_gate"])
        summary = "| Metric | Coverage | Minimum | Result |\n|---|---:|---:|---|\n"
        for name, actual, minimum, passed in results:
            summary += (
                f"| {name} | {actual:.2f}% | {minimum:g}% | {'PASS' if passed else 'FAIL'} |\n"
            )
        (report_path.parent / "summary.md").write_text(summary)
        print(summary, end="")
        return int(not all(row[3] for row in results))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Coverage check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
