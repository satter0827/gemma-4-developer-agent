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
    ):
        parser = commands.add_parser(name, help=description)
        parser.add_argument("--config", type=Path, required=True)
        parser.set_defaults(handler=run)
    verify = commands.add_parser("verify-experiment", help="凍結したバンドルを公式SDKでCPU検証")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--sdk-config", type=Path, required=True)
    verify.set_defaults(handler=run)
    summary = commands.add_parser("summarize-experiment", help="完了した同一課題の比較を集計")
    summary.add_argument("--session", type=Path, required=True)
    summary.set_defaults(handler=run)
    sync = commands.add_parser("sync-tracking", help="Gitの公開記録をMLflowのSQLiteへ同期")
    sync.add_argument("--config", type=Path, required=True)
    sync.add_argument("--record", type=Path, action="append", required=True)
    sync.set_defaults(handler=run)


def run(root: Path, args: argparse.Namespace) -> int:
    if args.command == "validate-experiment":
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
        from gemma_agent_dev.evaluation.experiment_results import summarize

        report = summarize(args.session.resolve())
        write_json(args.session.parent / "paired-summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare-tracking":
        from gemma_agent_dev.evaluation.tracking import prepare as prepare_tracking

        print(prepare_tracking(root, args.config))
    else:
        from gemma_agent_dev.evaluation.tracking import sync

        print(sync(root, args.config, args.record))
    return 0
