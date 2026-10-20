"""Distinguish executed tests from explicitly expected collection failures."""

import xml.etree.ElementTree as ET
from pathlib import Path

PYTEST_COLLECTION_EXIT = 2
NORMAL_TEST_EXITS = frozenset({0, 1})
COLLECTION_FAILURE_MESSAGE = "collection failure"


def junit_cases(path: Path) -> list[ET.Element]:
    tree = ET.parse(path).getroot()
    if tree.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Invalid JUnit root element")
    cases = list(tree.iter("testcase"))
    if not cases or all(case.find("skipped") is not None for case in cases):
        raise ValueError("JUnit report has no executed tests")
    return cases


def is_collection_error(case: ET.Element) -> bool:
    error = case.find("error")
    return error is not None and error.get("message") == COLLECTION_FAILURE_MESSAGE


def junit_counts(path: Path) -> dict[str, int]:
    cases = junit_cases(path)
    if any(is_collection_error(case) for case in cases):
        raise ValueError("Collection errors are not executed tests")
    return {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }


def validate_test_evidence(exit_code: int | None, path: Path) -> dict[str, int]:
    if type(exit_code) is not int or exit_code not in NORMAL_TEST_EXITS:
        raise ValueError(f"Tests did not finish normally: {exit_code}")
    counts = junit_counts(path)
    if (exit_code == 0) != (counts["failures"] + counts["errors"] == 0):
        raise ValueError("Test exit code and JUnit results disagree")
    return counts


def diagnostic_evidence(
    exit_code: int | None, path: Path, expected_outcome: str
) -> tuple[str, dict[str, int]]:
    if (
        type(exit_code) is int
        and exit_code == PYTEST_COLLECTION_EXIT
        and expected_outcome == "collection_error"
    ):
        cases = junit_cases(path)
        if not all(is_collection_error(case) for case in cases):
            raise ValueError("Interrupted or mixed execution is not a collection-only failure")
        return "collection_error", {
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "collection_errors": len(cases),
        }
    counts = validate_test_evidence(exit_code, path)
    return ("passed" if exit_code == 0 else "failed"), counts
