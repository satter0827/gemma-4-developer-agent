"""Build a reproducible task ledger without reselecting the frozen evaluation cohort."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import yaml

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.diagnostics.qualification import load_selection
from gemma_agent_dev.evaluation.config import RuntimeSettings
from gemma_agent_dev.evaluation.results import load_run, read_object, require_evaluation_evidence
from gemma_agent_dev.tasks.catalog import TaskCatalog


class TaskReport(TypedDict):
    tasks_sha256: str
    sdk_version: str
    sdk_sha256: str
    tasks: list[dict[str, Any]]
    costs: list[dict[str, Any]]


ANNOTATION_FIELDS = frozenset({"target", "behavior", "reproduction", "fix"})


def read_inventory(
    path: Path, expected_hash: str, expected_count: int
) -> dict[str, dict[str, Any]]:
    if file_hash(path) != expected_hash:
        raise ValueError("Task inventory checksum mismatch")
    tasks = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("instance_id"), str)
            or row["instance_id"] in tasks
            or any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("repo", "patch", "test_patch", "problem_statement")
            )
        ):
            raise ValueError("Invalid or duplicate task inventory record")
        tasks[row["instance_id"]] = row
    if len(tasks) != expected_count:
        raise ValueError("Task inventory count changed")
    return tasks


def annotations(raw: Any, ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or not set(raw) <= ids:
        raise ValueError("Unknown task annotation")
    for value in raw.values():
        if not isinstance(value, dict) or set(value) != ANNOTATION_FIELDS | {"basis"}:
            raise ValueError("Annotation must include every assessment and its basis")
        if any(value[k] not in {"explicit", "absent", "unknown"} for k in ANNOTATION_FIELDS):
            raise ValueError("Invalid annotation assessment")
        if not isinstance(value["basis"], str) or not value["basis"].strip():
            raise ValueError("Annotation requires a basis")
    return raw


def build_report(
    root: Path, config_path: Path, observations_config: Path | None = None
) -> TaskReport:
    config = mapping(
        yaml.safe_load(project_path(root, str(config_path)).read_text()),
        {
            "catalog",
            "runtime",
            "cpu_environment",
            "wheel_cache",
            "timeout_seconds",
            "task_count",
            "qualified",
            "diagnostics",
            "annotations",
        },
        "task analysis",
    )
    positive(config["timeout_seconds"], "analysis timeout", integer=True)
    positive(config["task_count"], "task count", integer=True)
    catalog = TaskCatalog.load(root, Path(config["catalog"]))
    tasks = read_inventory(catalog.tasks_file, catalog.tasks_sha256, config["task_count"])
    notes = annotations(
        yaml.safe_load(project_path(root, config["annotations"]).read_text()), set(tasks)
    )
    runtime = RuntimeSettings.load(project_path(root, config["runtime"]))
    wheel_name = f"swegemma-{runtime.packages['swegemma']}-py3-none-any.whl"
    wheel = project_path(root, config["wheel_cache"]) / wheel_name
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "metrics.json"
        subprocess.run(
            [
                str(project_path(root, config["cpu_environment"]) / "bin/python"),
                "-m",
                "gemma_agent_dev.tasks.metrics",
                str(catalog.tasks_file),
                str(wheel),
                runtime.wheelhouse_hashes[wheel_name],
                runtime.packages["swegemma"],
                str(output),
            ],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            timeout=config["timeout_seconds"],
            check=True,
            capture_output=True,
            text=True,
        )
        metrics = json.loads(output.read_text())
    by_id = {m["instance_id"]: m for m in metrics}
    if len(by_id) != len(metrics) or set(by_id) != set(tasks):
        raise ValueError("Official structural metrics do not match the inventory")
    qualified = load_selection(root, Path(config["qualified"]), current=False)["task_ids"]
    diagnostic = yaml.safe_load(project_path(root, config["diagnostics"]).read_text())
    diagnostic_ids = {t["instance_id"] for t in diagnostic["tasks"]}
    if not set(qualified) | diagnostic_ids | set(catalog.tasks) <= set(tasks):
        raise ValueError("Unknown task group member")
    observations: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in tasks}
    costs = []
    seen_runs: set[str] = set()
    history: dict[str, Any] = {"runs": [], "scenario_counts": []}
    if observations_config is not None:
        history = mapping(
            yaml.safe_load(project_path(root, str(observations_config)).read_text()),
            {"runs", "scenario_counts"},
            "historical observations",
        )
        if not isinstance(history["scenario_counts"], list) or not history["scenario_counts"]:
            raise ValueError("Expected cost scenarios")
        for count in history["scenario_counts"]:
            positive(count, "scenario count", integer=True)
        if not isinstance(history["runs"], list) or not history["runs"]:
            raise ValueError("Expected historical runs")
    for spec in history["runs"]:
        mapping(spec, {"manifest", "quota_before", "quota_after", "hashes"}, "observation")
        mapping(spec["hashes"], {"manifest", "quota_before", "quota_after"}, "observation hashes")
        for name in ("manifest", "quota_before", "quota_after"):
            if file_hash(project_path(root, spec[name])) != spec["hashes"][name]:
                raise ValueError("Historical observation changed")
        run = load_run(project_path(root, spec["manifest"]).parent, require_evaluation=True)
        require_evaluation_evidence(run)
        if not run.complete or run.manifest.get("tasks_sha256") != catalog.tasks_sha256:
            raise ValueError("Incomplete or incompatible historical run")
        if run.manifest["run_id"] in seen_runs:
            raise ValueError("Duplicate historical run")
        seen_runs.add(run.manifest["run_id"])
        for task_id, row in run.rows.items():
            if task_id not in observations:
                raise ValueError("Unknown historical task")
            observations[task_id].append(
                {
                    "run_id": run.manifest["run_id"],
                    "resolved": row["resolved"],
                    "duration_seconds": row["duration_seconds"],
                }
            )
        before = read_object(project_path(root, spec["quota_before"]))
        after = read_object(project_path(root, spec["quota_after"]))
        used = after["used_hours"] - before["used_hours"]
        if before["refresh_at"] != after["refresh_at"] or used <= 0:
            raise ValueError("Quota observations cannot be compared")
        timing = run.manifest["timings"]
        total, evaluation = timing["total_seconds"], timing["evaluation_seconds"]
        costs.append(
            {
                "run_id": run.manifest["run_id"],
                "tasks": len(run.task_ids),
                "startup_seconds": timing["model_startup_seconds"],
                "evaluation_seconds": evaluation,
                "generation_seconds": None,
                "scoring_seconds": None,
                "save_seconds": None,
                "total_seconds": total,
                "quota_hours": round(used, 5),
                "scenarios": {
                    str(n): {
                        "wall_seconds": round(
                            total - evaluation + evaluation / len(run.task_ids) * n, 2
                        ),
                        "quota_hours": round(
                            used
                            / total
                            * (total - evaluation + evaluation / len(run.task_ids) * n),
                            5,
                        ),
                    }
                    for n in history["scenario_counts"]
                },
            }
        )
    rows = [
        {
            **by_id[task_id],
            "executable": task_id in catalog.tasks,
            "comparison": task_id in qualified,
            "diagnostic": task_id in diagnostic_ids,
            "annotation": notes.get(task_id),
            "observations": observations[task_id],
        }
        for task_id in sorted(tasks)
    ]
    return {
        "tasks_sha256": catalog.tasks_sha256,
        "sdk_version": runtime.packages["swegemma"],
        "sdk_sha256": runtime.wheelhouse_hashes[wheel_name],
        "tasks": rows,
        "costs": costs,
    }


def render(report: TaskReport) -> str:
    rows = report["tasks"]
    text = [
        "# 問題台帳",
        "",
        "分類の意味は[分類基準](criteria.md)を参照してください。",
        "",
        f"全体{len(rows)}問、実行設定{sum(r['executable'] for r in rows)}問、"
        f"既存比較{sum(r['comparison'] for r in rows)}問、"
        f"診断{sum(r['diagnostic'] for r in rows)}問です。各区分は重複します。",
        "",
        f"課題データSHA-256：`{report['tasks_sha256']}`。公式SDK：{report['sdk_version']}。",
        "",
        "| 問題ID | 構造指標 | ファイル数 | 変更行数 | 対象区分 | 文面の確認 | 過去の公式判定 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        groups = (
            "・".join(
                label
                for key, label in (
                    ("executable", "実行"),
                    ("comparison", "比較"),
                    ("diagnostic", "診断"),
                )
                if row[key]
            )
            or "—"
        )
        observed = (
            " / ".join(
                f"{s['run_id']}：{'解決' if s['resolved'] else '未解決'}"
                for s in row["observations"]
            )
            or "未評価"
        )
        note = "確認済み" if row["annotation"] else "未確認"
        text.append(
            f"| {row['instance_id']} | {row['difficulty_tier']} {row['composite_score']:.1f} | "
            f"{row['num_files']} | {row['total_churn']} | {groups} | {note} | {observed} |"
        )
    reviewed = [r for r in rows if r["annotation"]]
    if reviewed:
        labels = {"explicit": "明記", "absent": "記載なし", "unknown": "未確認"}
        text += [
            "",
            "## 課題文の確認",
            "",
            "| 問題ID | 対象箇所 | 期待動作 | 再現方法 | 修正方針 | 根拠 |",
            "|---|---|---|---|---|---|",
        ]
        for row in reviewed:
            note = row["annotation"]
            values = " | ".join(
                labels[note[k]] for k in ("target", "behavior", "reproduction", "fix")
            )
            basis = note["basis"].replace("|", "／").replace("\n", " ")
            text.append(f"| {row['instance_id']} | {values} | {basis} |")
    text += [
        "",
        "## GPU費用",
        "",
        "生成・採点・保存の個別時間は未確認です。"
        if report["costs"]
        else "過去のGPU実行は結合していません。",
        "",
        "| 実行ID | 起動（秒） | 生成・採点合計（秒） | 全体（秒） | quota時間 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cost in report["costs"]:
        text.append(
            f"| {cost['run_id']} | {cost['startup_seconds']:.2f} | "
            f"{cost['evaluation_seconds']:.2f} | {cost['total_seconds']:.2f} | "
            f"{cost['quota_hours']:.5f} |"
        )
    if report["costs"]:
        text += [
            "",
            "問題数以外の固定費を維持し、生成・採点時間だけを比例させた見積もりです。問題の構成や実行順による差を保証しません。",
            "",
            "| 問題数 | 全体時間の幅（秒） | quota時間の幅 |",
            "|---:|---:|---:|",
        ]
        for n in report["costs"][0]["scenarios"]:
            estimates = [c["scenarios"][n] for c in report["costs"]]
            wall = [e["wall_seconds"] for e in estimates]
            quota = [e["quota_hours"] for e in estimates]
            text.append(
                f"| {n} | {min(wall):.2f}–{max(wall):.2f} | {min(quota):.5f}–{max(quota):.5f} |"
            )
    return "\n".join(text) + "\n"


def generate(
    root: Path, config: Path, *, write_docs: bool, observations_config: Path | None = None
) -> Path:
    report = build_report(root, config, observations_config)
    output = root / "artifacts/task-analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    rendered = render(report)
    (output / "tasks.md").write_text(rendered)
    if write_docs:
        (root / "docs/reference/tasks.md").write_text(rendered)
    return output
