"""Select ranked tasks using controls before any model result exists."""

import json
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.config import mapping, positive, project_path
from gemma_agent_dev.diagnostic_config import DiagnosticSettings
from gemma_agent_dev.diagnostics import check_prepared, run_case
from gemma_agent_dev.test_evidence import validate_test_evidence


def qualify(root: Path, config: Path) -> Path:
    path = project_path(root, str(config))
    data = mapping(
        yaml.safe_load(path.read_text()),
        {"diagnostic_config", "per_repository", "candidates"},
        "qualification",
    )
    positive(data["per_repository"], "tasks per repository", integer=True)
    settings = DiagnosticSettings.load(root, Path(data["diagnostic_config"]))
    groups = data["candidates"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Expected ranked repository candidates")
    ids = [task_id for queue in groups.values() if isinstance(queue, list) for task_id in queue]
    if ids != [t.instance_id for t in settings.tasks] or len(ids) != len(set(ids)):
        raise ValueError("Qualification queues must match the diagnostic task order")
    destination = settings.output_dir / "qualified.json"
    if destination.exists():
        raise FileExistsError("Task selection is already frozen; do not reselect after evaluation")
    check_prepared(settings)
    records = settings.read_tasks()
    specifications = {t.instance_id: t for t in settings.tasks}
    output = settings.output_dir / new_run_id()
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "kind": "qualification",
        "status": "running",
        "config": str(path.relative_to(root)),
        "config_sha256": file_hash(path),
        "diagnostic_fingerprint": settings.fingerprint(),
        "task_ids": [],
        "candidates": [],
        "cases": [],
    }
    write_json(output / "report.json", report)
    try:
        for repo, queue in groups.items():
            selected: list[str] = []
            for task_id in queue:
                if len(selected) == data["per_repository"]:
                    break
                task = specifications[task_id]
                record = records[task_id]
                if record["repo"].split("/")[-1].lower() != repo:
                    raise ValueError("Ranked task belongs to another repository")
                cases = {}
                for name, patch in task.patches(record).items():
                    case = run_case(settings, task, record, name, patch, output / task_id / name)
                    report["cases"].append(case)
                    cases[name] = case
                if any("error" in c or "source_origins" not in c for c in cases.values()):
                    raise RuntimeError(f"Task environment is invalid: {task_id}; inspect {output}")
                eligible = cases["reference"]["passed"] and not cases["unmodified"]["passed"]
                reason = (
                    "eligible"
                    if eligible
                    else (
                        "reference_failed"
                        if not cases["reference"]["passed"]
                        else "unmodified_passed"
                    )
                )
                report["candidates"].append({"task_id": task_id, "reason": reason})
                if eligible:
                    selected.append(task_id)
                print(f"{task_id}: {reason}", flush=True)
                write_json(output / "report.json", report)
            if len(selected) != data["per_repository"]:
                raise RuntimeError(
                    f"Not enough discriminating tasks for {repo}; GPU evaluation is blocked"
                )
            report["task_ids"].extend(selected)
        check_prepared(settings)
        report["status"] = "completed"
        write_json(output / "report.json", report)
        write_json(
            destination,
            {
                "report": str((output / "report.json").relative_to(root)),
                "sha256": file_hash(output / "report.json"),
            },
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "report.json", report)
        raise
    return destination


def load_selection(root: Path, path: Path) -> dict[str, Any]:
    pointer = json.loads(project_path(root, str(path)).read_text())
    report_path = project_path(root, pointer["report"])
    if file_hash(report_path) != pointer["sha256"]:
        raise ValueError("Qualified task evidence changed")
    report: dict[str, Any] = json.loads(report_path.read_text())
    config = project_path(root, report["config"])
    if (
        report["status"] != "completed"
        or report["kind"] != "qualification"
        or file_hash(config) != report["config_sha256"]
    ):
        raise ValueError("Qualified task configuration changed or did not complete")
    data = yaml.safe_load(config.read_text())
    settings = DiagnosticSettings.load(root, Path(data["diagnostic_config"]))
    if report["diagnostic_fingerprint"] != settings.fingerprint():
        raise ValueError("Qualified task inputs changed")
    expected = []
    for queue in data["candidates"].values():
        eligible = [
            c["task_id"]
            for c in report["candidates"]
            if c["task_id"] in queue and c["reason"] == "eligible"
        ]
        if len(eligible) != data["per_repository"]:
            raise ValueError("Qualified task set is incomplete")
        expected.extend(eligible)
    if report["task_ids"] != expected or len(expected) != len(set(expected)):
        raise ValueError("Qualified task order changed")
    cases = {(c["task_id"], c["case"]): c for c in report["cases"]}
    if len(cases) != len(report["cases"]) or set(cases) != {
        (c["task_id"], label) for c in report["candidates"] for label in ("reference", "unmodified")
    }:
        raise ValueError("Qualified control cases are missing or duplicated")
    for candidate in report["candidates"]:
        task_id = candidate["task_id"]
        reference, unmodified = (cases[(task_id, name)] for name in ("reference", "unmodified"))
        is_eligible = reference["passed"] and not unmodified["passed"]
        reason = (
            "eligible"
            if is_eligible
            else ("reference_failed" if not reference["passed"] else "unmodified_passed")
        )
        if candidate["reason"] != reason:
            raise ValueError("Qualified decision disagrees with controls")
    for queue in data["candidates"].values():
        observed = [c["task_id"] for c in report["candidates"] if c["task_id"] in queue]
        if observed != queue[: len(observed)]:
            raise ValueError("Qualification skipped a higher-ranked candidate")
    for case in report["cases"]:
        folder = report_path.parent / case["task_id"] / case["case"]
        files = inventory(folder)
        files.pop("case.json", None)
        if files != case["artifacts"] or json.loads((folder / "case.json").read_text()) != case:
            raise ValueError("Qualified task artifacts changed")
        validate_test_evidence(case["exit_code"], folder / "junit.xml")
        if case["source_origins"].get("valid") is not True or case["passed"] != (
            case["exit_code"] == 0
        ):
            raise ValueError("Qualified source or test evidence is invalid")
    return report
