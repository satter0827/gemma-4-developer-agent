"""Route explicit feature commands without performing work during import."""

import argparse
import subprocess
import sys

from gemma_agent_dev.candidates import cli as candidates
from gemma_agent_dev.common.paths import project_root
from gemma_agent_dev.diagnostics import cli as diagnostics
from gemma_agent_dev.evaluation import cli as evaluation
from gemma_agent_dev.notebooks import cli as notebooks
from gemma_agent_dev.submission import cli as submission
from gemma_agent_dev.tasks import cli as tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提出コードの準備と評価結果の確認")
    commands = parser.add_subparsers(dest="command", required=True)
    tasks.configure(commands)
    diagnostics.configure(commands)
    evaluation.configure(commands)
    candidates.configure(commands)
    notebooks.configure(commands)
    submission.configure(commands)
    args = parser.parse_args(argv)
    try:
        return int(args.handler(project_root(), args))
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
