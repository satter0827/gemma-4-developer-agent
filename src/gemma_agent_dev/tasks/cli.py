"""Build the task reference ledger with optional historical observations."""

import argparse
from pathlib import Path
from typing import Any

from gemma_agent_dev.tasks.report import generate


def configure(commands: Any) -> None:
    parser = commands.add_parser("task-report", help="固定課題の参照台帳を生成")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--observations-config", type=Path, help="過去の評価結果とGPU費用の集計設定"
    )
    parser.add_argument("--write-docs", action="store_true", help="参照用台帳も更新")
    parser.set_defaults(handler=run)


def run(root: Path, args: argparse.Namespace) -> int:
    output = generate(
        root, args.config, write_docs=args.write_docs, observations_config=args.observations_config
    )
    print(f"保存先：{output}")
    return 0
