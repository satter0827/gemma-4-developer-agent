"""Parse candidate validation arguments and display the resulting evidence."""

import argparse
import json
from pathlib import Path
from typing import Any


def configure(commands: Any) -> None:
    preflight = commands.add_parser("preflight", help="ローカル検証と提出候補の固定")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--profile", required=True)
    preflight.set_defaults(handler=run)


def run(root: Path, args: argparse.Namespace) -> int:
    from gemma_agent_dev.candidates.preflight import run as run_preflight

    path = run_preflight(root, args.config, args.profile)
    report = json.loads(path.read_text())
    print(
        f"検証記録：{path}\n用途：{report['purpose']}\n実行設定："
        + json.dumps(report["execution"], ensure_ascii=False)
    )
    return 0
