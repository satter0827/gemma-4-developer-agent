"""Expose explicit CPU preparation, experiment inspection and metadata synchronization."""

import argparse
import json
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.evaluation.experiments import Experiment, content_hash


def configure(commands: Any) -> None:
    for name, description in (
        ("validate-experiment", "実験条件と固定した実行順序を確認"),
        ("prepare-experiment-tasks", "選択したEasy課題のCPU環境を準備"),
        ("freeze-experiment", "品質検査とCPU対照検証で比較候補を固定"),
        ("prepare-tracking", "MLflow専用CPU環境を準備"),
        ("plan-experiment", "未使用Easy課題と費用幅を確認"),
        ("prepare-optimization-environment", "GEPA専用CPU環境を準備"),
        ("prepare-optimization", "最適化の種候補とCPU対照を固定"),
        ("validate-optimization", "最適化の対象・上限・必要時間を確認"),
    ):
        parser = commands.add_parser(name, help=description)
        parser.add_argument("--config", type=Path, required=True)
        parser.set_defaults(handler=run)
    verify = commands.add_parser("verify-experiment", help="凍結したバンドルを公式SDKでCPU検証")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--sdk-config", type=Path, required=True)
    verify.set_defaults(handler=run)
    optimization = commands.add_parser("verify-optimization", help="GEPAと固定SDKでCPU契約を検証")
    optimization.add_argument("--bundle", type=Path, required=True)
    optimization.add_argument("--sdk-config", type=Path, required=True)
    optimization.set_defaults(handler=run)
    summary = commands.add_parser("summarize-experiment", help="完了した同一課題の比較を集計")
    summary.add_argument("--session", type=Path, required=True)
    summary.add_argument("--review", type=Path, help="採否と原因を記入した設定")
    summary.add_argument("--record-output", type=Path, help="公開記録の新規保存先")
    summary.set_defaults(handler=run)
    sync = commands.add_parser("sync-tracking", help="Gitの公開記録をMLflowのSQLiteへ同期")
    sync.add_argument("--config", type=Path, required=True)
    sync.add_argument("--record", type=Path, action="append", required=True)
    sync.set_defaults(handler=run)


def run(root: Path, args: argparse.Namespace) -> int:
    if args.command == "validate-optimization":
        from gemma_agent_dev.evaluation.optimization import Optimization

        print(
            json.dumps(
                Optimization.load(root, args.config).definition(root), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "prepare-optimization-environment":
        from gemma_agent_dev.evaluation.optimization_preparation import prepare_environment

        print(prepare_environment(root, args.config))
    elif args.command == "prepare-optimization":
        from gemma_agent_dev.evaluation.optimization_preparation import prepare as prepare_search

        print(prepare_search(root, args.config))
    elif args.command == "verify-optimization":
        from gemma_agent_dev.notebooks.optimization import verify_local as verify_optimization

        print(verify_optimization(root, args.bundle.resolve(), args.sdk_config))
    elif args.command == "validate-experiment":
        experiment = Experiment.load(root, args.config)
        definition = experiment.definition(root)
        print(
            json.dumps(
                {"definition_sha256": content_hash(definition), **definition},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "prepare-experiment-tasks":
        from gemma_agent_dev.diagnostics.config import DiagnosticSettings
        from gemma_agent_dev.diagnostics.local import prepare

        experiment = Experiment.load(root, args.config)
        print(prepare(DiagnosticSettings.load(root, experiment.diagnostic_config)))
    elif args.command == "freeze-experiment":
        from gemma_agent_dev.evaluation.experiment_preparation import prepare as freeze

        print(freeze(root, args.config))
    elif args.command == "verify-experiment":
        from gemma_agent_dev.notebooks.experiments import verify_local

        print(verify_local(root, args.bundle.resolve(), args.sdk_config))
    elif args.command == "summarize-experiment":
        from gemma_agent_dev.candidates.preflight import artifact_limit
        from gemma_agent_dev.evaluation.experiment_results import summarize

        if bool(args.review) != bool(args.record_output):
            raise ValueError("Specify --review and --record-output together")
        report = summarize(args.session.resolve(), max_bytes=artifact_limit(root))
        if args.record_output:
            from gemma_agent_dev.evaluation.experiment_records import export

            export(
                root, args.session.resolve(), args.review, args.record_output, artifact_limit(root)
            )
        write_json(args.session.parent / "paired-summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "plan-experiment":
        from gemma_agent_dev.tasks.rotation import plan

        print(json.dumps(plan(root, args.config), ensure_ascii=False, indent=2))
    elif args.command == "prepare-tracking":
        from gemma_agent_dev.evaluation.tracking import prepare as prepare_tracking

        print(prepare_tracking(root, args.config))
    else:
        from gemma_agent_dev.evaluation.tracking import sync

        print(sync(root, args.config, args.record))
    return 0
