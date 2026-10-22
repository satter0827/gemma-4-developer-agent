"""Verify focused tests and implementation changes without modifying tracked files."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

EXIT_FAILED = 1
EXIT_TIMEOUT = 124


def inside(root: Path, value: str) -> Path:
    path = root / value.split("::")[0]
    if (
        value.startswith("-")
        or ".." in Path(value).parts
        or not path.resolve().is_relative_to(root)
        or not path.exists()
    ):
        raise ValueError("Test and source paths must stay in the repository")
    return path.resolve()


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise subprocess.TimeoutExpired("verification", 0)
    return seconds


def implementation_changes(root: Path, roots: list[str], deadline: float) -> list[str]:
    names = set()
    for args in (
        ["git", "diff", "--no-ext-diff", "--name-only", "-z", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    ):
        result = subprocess.run(
            args, cwd=root, capture_output=True, check=True, timeout=remaining(deadline)
        )
        names.update(result.stdout.decode().strip("\0").split("\0"))
    source_roots = [inside(root, value) for value in roots]
    return sorted(
        name
        for name in names
        if name
        and not re.search(
            r"(?:^|/)(?:tests?(?:/|_)|conftest\.py$|(?:repro|reproduce|scratch|tmp)[^/]*(?:/|$))",
            name,
        )
        and not {"__pycache__", ".pytest_cache"}.intersection(Path(name).parts)
        and Path(name).suffix != ".pyc"
        and any((root / name).resolve().is_relative_to(source) for source in source_roots)
    )


def run(
    workspace: Path,
    tests: list[str],
    roots: list[str],
    timeout: float,
    policy: dict[str, int],
    modules: list[str],
) -> tuple[dict[str, Any], int]:
    root = workspace.resolve(strict=True)
    for test in tests:
        inside(root, test)
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not {"max_failures", "max_message_characters"} <= set(policy)
        or not modules
        or any(not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) for module in modules)
        or any(type(v) is not int or v <= 0 for v in policy.values())
    ):
        raise ValueError("Positive limits are required")
    deadline = time.monotonic() + timeout
    changed = implementation_changes(root, roots, deadline)
    report: dict[str, Any] = {
        "status": "failed",
        "test_exit_code": None,
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "source_changed": bool(changed),
        "source_paths": changed,
        "failures": [],
    }
    with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryFile("w+b") as log:
        junit = Path(temporary) / "tests.xml"
        origins = Path(temporary) / "origins.json"
        plugin = Path(__file__).resolve().with_name("issue_origins.py")
        bootstrap = (
            "import importlib.util,json,os,pytest; "
            "spec=importlib.util.spec_from_file_location('issue_workflow_origins',"
            "os.environ['ISSUE_ORIGIN_PLUGIN']); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "raise SystemExit(pytest.main(json.loads(os.environ['ISSUE_TEST_ARGS']),"
            "plugins=[module]))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", bootstrap],
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "ISSUE_ORIGIN_PLUGIN": str(plugin),
                "ISSUE_ORIGIN_OUTPUT": str(origins),
                "ISSUE_ORIGIN_MODULES": json.dumps(modules),
                "ISSUE_TEST_ARGS": json.dumps([*tests, "--junitxml", str(junit)]),
            },
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Keep descendants in the SDK sandbox group so its deadline kills them too.
            start_new_session=False,
        )
        try:
            code = process.wait(timeout=remaining(deadline))
        except subprocess.TimeoutExpired:
            report["status"] = "timeout"
            code = EXIT_TIMEOUT
        finally:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        report["test_exit_code"] = code
        if code:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - policy["max_message_characters"] * policy["max_failures"]))
            report["test_output_tail"] = log.read().decode("utf-8", errors="replace")
        if junit.exists():
            cases = list(ET.parse(junit).getroot().iter("testcase"))
            identities = [(c.get("classname"), c.get("name")) for c in cases]
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate JUnit cases")
            report["collected"] = len(cases)
            for case in cases:
                failure = case.find("failure")
                error = case.find("error")
                if failure is not None or error is not None:
                    report["failed" if failure is not None else "errors"] += 1
                    node = failure if failure is not None else error
                    assert node is not None
                    if len(report["failures"]) < policy["max_failures"]:
                        report["failures"].append(
                            {
                                "test": case.get("name"),
                                "message": node.get("message", "")[
                                    : policy["max_message_characters"]
                                ],
                            }
                        )
                elif case.find("skipped") is not None:
                    report["skipped"] += 1
                else:
                    report["passed"] += 1
        elif code == 0:
            raise ValueError("Successful test process did not create JUnit")
        report["origins"] = json.loads(origins.read_text()) if origins.exists() else {}
        report["missing_modules"] = [
            name
            for name in modules
            if not any(key == name or key.startswith(name + ".") for key in report["origins"])
        ]
        report["unexpected_origins"] = {
            name: origin
            for name, origin in report["origins"].items()
            if not any(
                Path(origin).resolve().is_relative_to(inside(root, value)) for value in roots
            )
        }
    if (
        code == 0
        and report["collected"]
        and report["passed"]
        and not report["failed"]
        and not report["errors"]
        and changed
        and not report["missing_modules"]
        and not report["unexpected_origins"]
    ):
        report["status"] = "passed"
        return report, 0
    return report, code or EXIT_FAILED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        policy = json.loads(
            (Path(__file__).resolve().parents[1] / "assets/policy.json").read_text()
        )
        report, code = run(
            args.workspace, args.test, args.source_root, args.timeout_seconds, policy, args.module
        )
    except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
        report, code = (
            {"status": "infrastructure_error", "exception_type": type(error).__name__},
            EXIT_FAILED,
        )
    print(json.dumps(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
