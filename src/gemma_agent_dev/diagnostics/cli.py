"""Prepare task environments, diagnose patches, and validate fixed comparisons."""

import argparse
from pathlib import Path
from typing import Any

from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import diagnose, prepare
from gemma_agent_dev.diagnostics.qualification import qualify


def configure(commands: Any) -> None:
    for name, help_text in (
        ("prepare-diagnostics", "ローカル診断の入力とPython環境を準備"),
        ("diagnose", "準備済み環境でパッチを診断"),
        ("qualify", "未修正と参照パッチで比較課題を検証"),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--config", type=Path, required=True)
        parser.set_defaults(handler=run)
        if name == "qualify":
            parser.add_argument(
                "--fixed-selection", action="store_true", help="固定済み課題を入れ替えずに再検証"
            )


def run(root: Path, args: argparse.Namespace) -> int:
    if args.command == "qualify":
        output = qualify(root, args.config, fixed_selection=args.fixed_selection)
    else:
        settings = DiagnosticSettings.load(root, args.config)
        output = prepare(settings) if args.command == "prepare-diagnostics" else diagnose(settings)
    print(f"保存先：{output}")
    return 0
