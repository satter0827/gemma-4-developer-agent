"""Validate test evidence shared by native and official patch diagnostics."""

import xml.etree.ElementTree as ET
from pathlib import Path


def junit_counts(path: Path) -> dict[str, int]:
    tree = ET.parse(path).getroot()
    if tree.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Invalid JUnit root element")
    cases = list(tree.iter("testcase"))
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise ValueError("JUnit report has no executed tests")
    return {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }


def validate_test_evidence(exit_code: int | None, path: Path) -> dict[str, int]:
    counts = junit_counts(path)
    if type(exit_code) is not int or exit_code not in {0, 1, 2}:
        raise ValueError(f"Tests did not finish normally: {exit_code}")
    if (exit_code == 0) != (counts["failures"] + counts["errors"] == 0):
        raise ValueError("Test exit code and JUnit results disagree")
    return counts
