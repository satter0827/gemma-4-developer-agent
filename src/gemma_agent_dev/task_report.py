"""Build a reproducible task ledger without reselecting the frozen evaluation cohort."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.artifacts import file_hash, write_json
from gemma_agent_dev.config import RuntimeSettings, mapping, positive, project_path
from gemma_agent_dev.results import load_run, read_object
from gemma_agent_dev.task_catalog import TaskCatalog

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


def build_report(root: Path, config_path: Path) -> dict[str, Any]:
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
            "runs",
            "scenario_counts",
            "annotations",
        },
        "task analysis",
    )
    positive(config["timeout_seconds"], "analysis timeout", integer=True)
    positive(config["task_count"], "task count", integer=True)
    if not isinstance(config["scenario_counts"], list) or not config["scenario_counts"]:
        raise ValueError("Expected cost scenarios")
    for count in config["scenario_counts"]:
        positive(count, "scenario count", integer=True)
    catalog = TaskCatalog.load(root, Path(config["catalog"]))
    tasks = read_inventory(catalog.tasks_file, catalog.tasks_sha256, config["task_count"])
    notes = annotations(config["annotations"], set(tasks))
    runtime = RuntimeSettings.load(project_path(root, config["runtime"]))
    wheel_name = f"swegemma-{runtime.packages['swegemma']}-py3-none-any.whl"
    wheel = project_path(root, config["wheel_cache"]) / wheel_name
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "metrics.json"
        subprocess.run(
            [
                str(project_path(root, config["cpu_environment"]) / "bin/python"),
                "-m",
                "gemma_agent_dev.task_metrics",
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
    qualified_pointer = read_object(project_path(root, config["qualified"]))
    qualified_path = project_path(root, qualified_pointer["report"])
    if file_hash(qualified_path) != qualified_pointer["sha256"]:
        raise ValueError("Frozen cohort evidence changed")
    qualified = read_object(qualified_path)["task_ids"]
    diagnostic = yaml.safe_load(project_path(root, config["diagnostics"]).read_text())
    diagnostic_ids = {t["instance_id"] for t in diagnostic["tasks"]}
    if not set(qualified) | diagnostic_ids | set(catalog.tasks) <= set(tasks):
        raise ValueError("Unknown task group member")
    observations: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in tasks}
    costs = []
    seen_runs: set[str] = set()
    for spec in config["runs"]:
        run = load_run(project_path(root, spec["manifest"]).parent, require_evaluation=True)
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
                    for n in config["scenario_counts"]
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


def render(report: dict[str, Any]) -> str:
    rows = report["tasks"]
    text = [
        "# 問題台帳",
        "",
        "`task-report`で生成します。分類の意味は[分析基準](criteria.md)を参照してください。",
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
        "実測値と問題数別の見積もりを分けます。生成・採点・保存の個別時間は既存記録から分離できず、未確認です。",
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


def generate(root: Path, config: Path, *, write_docs: bool) -> Path:
    report = build_report(root, config)
    output = root / "artifacts/task-analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    rendered = render(report)
    (output / "tasks.md").write_text(rendered)
    if write_docs:
        (root / "docs/analysis/tasks.md").write_text(rendered)
    return output
