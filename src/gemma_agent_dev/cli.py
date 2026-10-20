"""Prepare private Kaggle code and inspect official evaluation artifacts."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from gemma_agent_dev.preflight import artifact_limit
from gemma_agent_dev.results import Run, comparison_differences, load_run, read_artifact

ARTIFACT_TYPES = {
    "patch": "patches/{task}.patch",
    "test-log": "test_outputs/{task}.log",
    "agent-log": "logs/{task}.log",
    "trace": "traces/trace_{task}.json",
}
CONDITION_LABELS = {
    "task_ids": "対象課題",
    "evaluation": "実行上限",
    "environment": "実行環境",
    "tasks_sha256": "課題データ",
    "verification_timeout_seconds": "パッチ検証時間",
    "vllm_config": "推論サーバー設定",
    "submission_files": "エージェント・提出ファイル",
    "model_source": "モデル",
    "docker_image": "実行イメージ",
    "vllm": "推論設定",
    "context_cache": "キャッシュ設定",
    "compaction": "履歴の圧縮設定",
    "incomplete": "完了状況",
}


def project_root() -> Path:
    current = Path.cwd().resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").is_file() and (path / "src/gemma_agent_dev").is_dir():
            return path
    raise ValueError("プロジェクトのフォルダー内で実行してください。")


def artifact_directory(root: Path, category: str) -> Path:
    directory = (root / "artifacts" / category).resolve()
    if not directory.is_relative_to(root / "artifacts"):
        raise ValueError("生成物の保存先がartifactsフォルダーの外を指しています。")
    return directory


def load_runs(directory: Path) -> tuple[list[Run], list[str]]:
    runs: list[Run] = []
    errors = []
    seen = set()
    for manifest in sorted(directory.glob("**/manifest.json")):
        try:
            if not manifest.resolve().is_relative_to(directory):
                raise ValueError("結果ファイルが保存先の外を指しています。")
            run = load_run(manifest.parent)
            run_id = run.manifest["run_id"]
            if run_id in seen:
                runs = [item for item in runs if item.manifest["run_id"] != run_id]
                raise ValueError(f"実行IDが重複しています：{run_id}")
            seen.add(run_id)
            runs.append(run)
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"結果を読み込めません：{manifest.parent.name}（{error}）")
    return runs, errors


def find_run(runs: list[Run], run_id: str) -> Run:
    matches = [run for run in runs if run.manifest["run_id"] == run_id]
    if not matches:
        raise ValueError(f"実行IDが見つかりません：{run_id}")
    return matches[0]


def task_status(run: Run, task: str) -> str:
    if run.kind == "verification":
        record = run.controls.get(task)
        if record is None:
            return "未完了"
        return (
            "検証通過"
            if record["reference"]["resolved"] and not record["unmodified"]["resolved"]
            else "検証失敗"
        )
    row = run.rows.get(task)
    if row is None:
        return "未完了"
    if row["resolved"]:
        return "解決"
    return "実行エラー" if row.get("error") else "未解決"


def run_status(run: Run) -> str:
    if run.problems:
        return "要確認"
    return {"completed": "完了", "running": "未完了", "failed": "失敗"}[run.manifest["status"]]


def condition(run: Run, field: str) -> object:
    if field == "incomplete":
        return run_status(run)
    return run.manifest.get(field, run.manifest.get("runtime_config", {}).get(field))


def report_problems(run: Run) -> bool:
    for problem in run.problems:
        print(f"確認事項：{run.manifest['run_id']}：{problem}", file=sys.stderr)
    return bool(run.problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提出コードの準備と評価結果の確認")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-kaggle", help="Private用の提出準備コードを生成")
    prepare.add_argument("--name", required=True, help="生成先の名前")
    prepare.add_argument("--preflight-report", type=Path, help="完了したローカル事前検証の記録")
    prepare.add_argument("--verify-only", action="store_true", help="GPUを使わず参照検証のみ実行")
    prepare.add_argument("--diagnostics-config", type=Path, help="CPU診断に使用する設定ファイル")
    preflight = commands.add_parser("preflight", help="ローカル検証と提出候補の固定")
    preflight.add_argument("--config", type=Path, required=True)
    for name, help_text in (
        ("prepare-diagnostics", "ローカル診断の入力とPython環境を準備"),
        ("diagnose", "準備済み環境でパッチを診断"),
    ):
        diagnostic = commands.add_parser(name, help=help_text)
        diagnostic.add_argument("--config", type=Path, required=True, help="診断設定ファイル")
    commands.add_parser("results", help="保存した評価結果を一覧表示")
    show = commands.add_parser("show", help="評価結果の詳細を表示")
    show.add_argument("run_id", metavar="RUN_ID")
    show.add_argument("--task", help="課題ID")
    show.add_argument("--artifact", choices=ARTIFACT_TYPES, help="表示するファイル")
    compare = commands.add_parser("compare", help="2つの実験を比較")
    compare.add_argument("left", metavar="RUN_ID")
    compare.add_argument("right", metavar="RUN_ID")
    args = parser.parse_args(argv)
    if args.command == "show" and args.artifact and not args.task:
        parser.error("--artifactには--taskの指定が必要です。")
    if args.command == "prepare-kaggle" and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name
    ):
        parser.error("--nameには英数字・ハイフン・アンダースコアを使用してください。")
    try:
        root = project_root()
        if args.command == "preflight":
            from gemma_agent_dev.preflight import run as run_preflight

            print(f"検証記録：{run_preflight(root, args.config)}")
            return 0
        if args.command in {"prepare-diagnostics", "diagnose"}:
            from gemma_agent_dev.diagnostic_config import DiagnosticSettings
            from gemma_agent_dev.diagnostics import diagnose
            from gemma_agent_dev.diagnostics import prepare as prepare_diagnostics

            settings = DiagnosticSettings.load(root, args.config)
            output = (
                prepare_diagnostics(settings)
                if args.command == "prepare-diagnostics"
                else diagnose(settings)
            )
            print(f"保存先：{output}")
            return 0
        if args.command == "prepare-kaggle":
            from gemma_agent_dev.kaggle_bundle import prepare as build_bundle

            destination = artifact_directory(root, "kaggle") / args.name
            if args.diagnostics_config:
                if args.verify_only or args.preflight_report:
                    parser.error("--diagnostics-configは他の評価設定と併用できません。")
                from gemma_agent_dev.diagnostic_bundle import prepare as build_diagnostics

                digest = build_diagnostics(root, destination, args.diagnostics_config)
                print(f"保存先：{destination}\nソースSHA-256：{digest}")
                return 0
            if args.preflight_report is None:
                raise ValueError("--preflight-reportを指定してください。")
            digest = build_bundle(
                root, destination, report_path=args.preflight_report, verify_only=args.verify_only
            )
            print(f"保存先：{destination}\nソースSHA-256：{digest}")
            return 0
        runs, errors = load_runs(artifact_directory(root, "results"))
        for error in errors:
            print(error, file=sys.stderr)
        invalid = bool(errors)
        if args.command == "results":
            if not runs and not errors:
                print("評価結果はまだありません。")
            if runs:
                print("実行ID\t状態\t段階\t予定\t完了\t解決\t解決率\t所要時間（秒）\t種別")
            for run in runs:
                rate = run.resolution_rate
                print(
                    "\t".join(
                        str(value)
                        for value in (
                            run.manifest["run_id"],
                            run_status(run),
                            run.manifest.get("phase", "—"),
                            len(run.task_ids),
                            run.completed_count,
                            run.resolved if run.kind == "evaluation" else "—",
                            f"{rate:.1%}" if rate is not None else "—",
                            run.manifest.get("timings", {}).get("total_seconds", "—"),
                            "参照検証" if run.kind == "verification" else "エージェント評価",
                        )
                    )
                )
                invalid = report_problems(run) or invalid
        elif args.command == "show":
            run = find_run(runs, args.run_id)
            invalid = report_problems(run) or invalid
            if args.task:
                if args.task not in run.task_ids:
                    raise ValueError(f"対象課題に含まれません：{args.task}")
                print(f"課題：{args.task}\n結果：{task_status(run, args.task)}")
                if args.artifact:
                    relative = ARTIFACT_TYPES[args.artifact].format(
                        task=args.task.replace("/", "__")
                    )
                    print(read_artifact(run, relative, artifact_limit(root)))
                else:
                    records = run.controls if run.kind == "verification" else run.rows
                    print(json.dumps(records.get(args.task), ensure_ascii=False, indent=2))
            else:
                print(f"状態：{run_status(run)}\n実行条件と記録：")
                print(json.dumps(run.manifest, ensure_ascii=False, indent=2))
        else:
            left, right = find_run(runs, args.left), find_run(runs, args.right)
            invalid = report_problems(left) or invalid
            invalid = report_problems(right) or invalid
            if left.kind != "evaluation" or right.kind != "evaluation":
                raise ValueError("参照検証はエージェント評価と比較できません。")
            differences = comparison_differences(left, right)
            if differences:
                print("比較条件に差があります。")
                for field in differences:
                    a, b = condition(left, field), condition(right, field)
                    if field == "submission_files" and isinstance(a, dict) and isinstance(b, dict):
                        changed = sorted(
                            name for name in a.keys() | b.keys() if a.get(name) != b.get(name)
                        )
                        print(f"{CONDITION_LABELS[field]}：" + "、".join(changed))
                        continue
                    print(
                        f"{CONDITION_LABELS[field]}："
                        + json.dumps(
                            {
                                args.left: condition(left, field),
                                args.right: condition(right, field),
                            },
                            ensure_ascii=False,
                        )
                    )
            else:
                print("同じ課題と実行条件で比較できます。")
            print(f"課題\t{args.left}\t{args.right}\t左の時間（秒）\t右の時間（秒）")
            for task in sorted(set(left.task_ids) | set(right.task_ids)):
                print(
                    "\t".join(
                        str(value)
                        for value in (
                            task,
                            task_status(left, task),
                            task_status(right, task),
                            left.rows.get(task, {}).get("duration_seconds", "—"),
                            right.rows.get(task, {}).get("duration_seconds", "—"),
                        )
                    )
                )
        return int(invalid)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(f"処理できません：{error}", file=sys.stderr)
        return 1
