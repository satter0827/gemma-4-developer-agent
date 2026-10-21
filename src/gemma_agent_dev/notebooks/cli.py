"""Generate private Notebook bundles without starting remote execution."""

import argparse
import re
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.paths import artifact_directory


def configure(commands: Any) -> None:
    prepare = commands.add_parser("prepare-kaggle", help="Private用の提出準備コードを生成")
    prepare.add_argument("--name", required=True, help="生成先の名前")
    prepare.add_argument("--experiment-report", type=Path, help="固定した比較実験の記録")
    prepare.add_argument("--optimization-report", type=Path, help="固定した最適化の準備記録")
    prepare.add_argument("--quota", type=Path, help="GPU実行前に観測した枠と消費量")
    prepare.add_argument("--preflight-report", type=Path, help="完了したローカル事前検証の記録")
    prepare.add_argument("--verify-only", action="store_true", help="GPUを使わず参照検証のみ実行")
    prepare.add_argument("--cpu-report", type=Path, help="同じ候補を確認したCPU実行のmanifest.json")
    prepare.add_argument("--diagnostics-config", type=Path, help="CPU診断に使用する設定ファイル")
    prepare.set_defaults(handler=run, command_parser=prepare)


def run(root: Path, args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        args.command_parser.error("--nameには英数字・ハイフン・アンダースコアを使用してください。")
    from gemma_agent_dev.notebooks.evaluation import prepare as build_bundle

    destination = artifact_directory(root, "kaggle") / args.name
    if args.optimization_report:
        if args.experiment_report or args.diagnostics_config or args.preflight_report:
            args.command_parser.error("--optimization-reportは他の評価設定と併用できません。")
        import json

        from gemma_agent_dev.notebooks.optimization import prepare as build_optimization

        digest = build_optimization(
            root,
            destination,
            report_path=args.optimization_report.resolve(),
            verify_only=args.verify_only,
            cpu_report=args.cpu_report,
            quota=json.loads(args.quota.read_text()) if args.quota else None,
        )
        print(f"保存先：{destination}\nソースSHA-256：{digest}")
        return 0
    if args.experiment_report:
        if args.diagnostics_config or args.preflight_report:
            args.command_parser.error("--experiment-reportは他の評価設定と併用できません。")
        import json

        from gemma_agent_dev.notebooks.experiments import prepare as build_experiment

        quota = json.loads(args.quota.read_text()) if args.quota else None
        digest = build_experiment(
            root,
            destination,
            report_path=args.experiment_report.resolve(),
            verify_only=args.verify_only,
            cpu_report=args.cpu_report,
            quota=quota,
        )
        print(f"保存先：{destination}\nソースSHA-256：{digest}")
        return 0
    if args.quota:
        args.command_parser.error("--quotaには--experiment-reportの指定が必要です。")
    if args.diagnostics_config:
        if args.verify_only or args.preflight_report or args.cpu_report:
            args.command_parser.error("--diagnostics-configは他の評価設定と併用できません。")
        from gemma_agent_dev.notebooks.diagnostics import prepare as build_diagnostics

        digest = build_diagnostics(root, destination, args.diagnostics_config)
        print(f"保存先：{destination}\nソースSHA-256：{digest}")
        return 0
    if args.preflight_report is None:
        raise ValueError("--preflight-reportを指定してください。")
    digest = build_bundle(
        root,
        destination,
        report_path=args.preflight_report,
        verify_only=args.verify_only,
        cpu_report=args.cpu_report,
    )
    print(f"保存先：{destination}\nソースSHA-256：{digest}")
    return 0
