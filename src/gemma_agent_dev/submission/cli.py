"""Submission command input and concise Japanese operational summaries."""

import argparse
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.evidence import archive_files, qualify_candidate
from gemma_agent_dev.candidates.experiment_evidence import qualify_experiment
from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.config import project_path
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.submission.config import SubmissionSettings, timestamp
from gemma_agent_dev.submission.service import execute_run, resolve_attempt, synchronize
from gemma_agent_dev.submission.state import add_candidate, empty_state
from gemma_agent_dev.submission.store import StateStore

REASONS = {
    "ready": "提出できる候補があります",
    "accepted": "提出を受け付けました",
    "already_accepted": "当日分は受付済みです",
    "outside_window": "送信時刻の範囲外です",
    "daily_limit": "当日の提出枠がありません",
    "manual_hold": "手動提出から24時間は保留します",
    "no_candidate": "新しい候補がありません",
    "acceptance_unknown": "受付状況の確認が必要です",
    "operation_failed": "処理に失敗しました。接続・権限・候補を確認してください",
    "monitor_only": "採点結果を確認しました",
    "after_deadline": "提出期限を過ぎています",
    "monitoring_finished": "結果確認を終了しました",
}


def configure(commands: Any) -> None:
    parser = commands.add_parser("submission", help="検証済み候補の登録と提出状況の管理")
    parser.set_defaults(handler=run)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--api-python", type=Path, help="Kaggle SDK環境のPython。省略時は実行中のPython"
    )
    actions = parser.add_subparsers(dest="submission_action", required=True)
    for name in (
        "init",
        "sync",
        "plan",
        "run",
        "resolve",
        "register",
        "register-experiment",
        "submit",
    ):
        action = actions.add_parser(name)
        action.add_argument("--execute", action="store_true", help="状態更新・提出を明示的に有効化")
        if name in {"plan", "run"}:
            action.add_argument(
                "--mode", choices=("morning", "backup", "monitor"), default="morning"
            )
        if name in {"register", "register-experiment"}:
            inputs = (
                ("cpu-session", "gpu-session")
                if name == "register-experiment"
                else ("cpu-manifest", "gpu-manifest")
            )
            for label in (*inputs, "output"):
                action.add_argument(f"--{label}", type=Path, required=True)
            for label in ("owner", "kernel", "kernel-path"):
                action.add_argument(f"--{label}", required=True)
            action.add_argument("--version", type=int, required=True)
            if name == "register-experiment":
                action.add_argument("--variant", required=True)
        if name == "submit":
            action.add_argument("--candidate", required=True)
        if name == "resolve":
            action.add_argument("--token", required=True)
            resolution = action.add_mutually_exclusive_group(required=True)
            resolution.add_argument("--submission-id", type=int)
            resolution.add_argument("--confirm-not-accepted", action="store_true")


def summary(value: dict[str, Any], competition: str, *, execute: bool) -> str:
    lines = [
        "# 提出状況",
        "",
        REASONS[value["reason"]],
        "",
        "実行：" + ("送信許可・状態更新あり" if execute else "未送信確認・状態更新なし"),
        "",
        "## 公開スコア上位の提出",
        "",
    ]
    lines.extend(f"- 提出ID {s['id']}：{s['score']}" for s in value["top"])
    if not value["top"]:
        lines.append("スコアが確定した提出はありません。")
    if value["incidents"]:
        lines += [
            "",
            "継続中の確認事項があります。提出記録と候補の取得・照合状況を確認してください。",
        ]
    lines += [
        "",
        "最終選択のチェック状態：未確認。提出画面で上記IDの選択を確認してください。",
        f"[提出画面](https://www.kaggle.com/competitions/{competition}/submissions)",
        "",
    ]
    return "\n".join(lines)


def run(root: Path, args: argparse.Namespace) -> int:
    settings = SubmissionSettings.load(root, args.config)
    execute = args.execute and args.submission_action in {"run", "submit"}
    # Publish the stop signal before Git/API operations, including when state loading fails.
    if execute and datetime.now(UTC) >= timestamp(settings.monitoring_end):
        if output := os.environ.get("GITHUB_OUTPUT"):
            with Path(output).open("a") as handle:
                handle.write("stop=true\n")
    gateway = KaggleGateway(root, settings, args.api_python or Path(sys.executable))
    store = StateStore(
        root, settings.state_branch, settings.competition, settings.api_timeout_seconds
    )
    if args.submission_action in {"register", "register-experiment"}:
        source = {
            "owner": args.owner,
            "kernel": args.kernel,
            "version": args.version,
            "path": args.kernel_path,
        }
        if args.submission_action == "register-experiment":
            candidate = qualify_experiment(
                project_path(root, str(args.cpu_session)),
                project_path(root, str(args.gpu_session)),
                args.variant,
                source,
                settings.max_archive_bytes,
            )
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / "submission.zip"
                gateway.call(
                    "kernel",
                    source=candidate["source"],
                    destination=str(archive),
                    max_bytes=settings.max_archive_bytes,
                )
                if (
                    file_hash(archive) != candidate["archive_sha256"]
                    or archive_files(archive, settings.max_archive_bytes) != candidate["files"]
                ):
                    raise ValueError("Saved Notebook archive differs from the qualified candidate")
        else:
            candidate = qualify_candidate(
                project_path(root, str(args.cpu_manifest)),
                project_path(root, str(args.gpu_manifest)),
                source,
                settings.max_archive_bytes,
            )
        path = project_path(root, str(args.output))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, candidate)
        if args.execute:
            state = store.load()
            if state is None:
                raise ValueError("提出状態を先に初期化してください。")
            add_candidate(state, path)
            store.save(state)
        print(f"候補ID：{candidate['id']}\n保存先：{path}")
        return 0
    state = store.load()
    if args.submission_action == "init":
        if state is not None:
            raise ValueError("提出状態は初期化済みです。")
        state = empty_state(settings.competition)
        for seed in settings.seed_candidates:
            add_candidate(state, project_path(root, seed))
        if args.execute:
            store.save(state)
        print(
            f"登録候補：{len(state['candidates'])}件。"
            f"状態更新：{'あり' if args.execute else 'なし'}"
        )
        return 0
    if state is None:
        raise ValueError("提出状態を先に初期化してください。")
    if args.submission_action in {"sync", "resolve"}:
        synchronize(state, gateway, settings, datetime.now(UTC))
        if args.submission_action == "resolve":
            resolve_attempt(state, args.token, args.submission_id)
        if args.execute:
            store.save(state)
        print("提出状況を確認しました。状態更新：" + ("あり" if args.execute else "なし"))
        return 0
    # plan is always read-only even if --execute was supplied accidentally.
    value = execute_run(
        state,
        store,
        gateway,
        settings,
        mode="manual" if args.submission_action == "submit" else args.mode,
        execute=execute,
        clock=lambda: datetime.now(UTC),
        candidate_id=args.candidate if args.submission_action == "submit" else None,
    )
    rendered = summary(value, settings.competition, execute=execute)
    print(rendered)
    if output := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(output).open("a") as handle:
            handle.write(rendered)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as handle:
            handle.write(f"stop={str(value['stop']).lower()}\n")
    local = root / "artifacts/submission-status"
    local.mkdir(parents=True, exist_ok=True)
    write_json(local / "report.json", value)
    return int(execute and value["notify"])
