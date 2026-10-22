"""Route explicit feature commands without performing work during import."""

import argparse
import json
import subprocess
import sys

from gemma_agent_dev.candidates import cli as candidates
from gemma_agent_dev.common.artifacts import new_run_id
from gemma_agent_dev.common.events import LogSettings, event, phase, session
from gemma_agent_dev.common.paths import project_root
from gemma_agent_dev.diagnostics import cli as diagnostics
from gemma_agent_dev.evaluation import cli as evaluation
from gemma_agent_dev.evaluation import experiments_cli as experiments
from gemma_agent_dev.notebooks import cli as notebooks
from gemma_agent_dev.submission import cli as submission
from gemma_agent_dev.tasks import cli as tasks
from gemma_agent_dev.workflows import cli as workflows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提出コードの準備と評価結果の確認")
    commands = parser.add_subparsers(dest="command", required=True)
    workflows.configure(commands)
    tasks.configure(commands)
    diagnostics.configure(commands)
    evaluation.configure(commands)
    experiments.configure(commands)
    candidates.configure(commands)
    notebooks.configure(commands)
    submission.configure(commands)
    arguments = sys.argv[1:] if argv is None else argv
    workflow_json = arguments[:1] == ["workflow"] and "--json" in arguments
    try:
        args = parser.parse_args(arguments)
    except SystemExit as error:
        if not workflow_json or error.code == 0:
            raise
        print(json.dumps(workflows.failure("arguments", "invalid_arguments")))
        return int(error.code or 2)
    try:
        root = project_root()
        execution_id = new_run_id()
        with session(root / "artifacts/logs" / execution_id, LogSettings.load(root), execution_id):
            try:
                with phase("command", stage=args.command):
                    return int(args.handler(root, args))
            except BaseException:
                event("command.exception", level="ERROR", exception=True)
                raise
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(f"処理できません：{error}", file=sys.stderr)
        if workflow_json:
            print(json.dumps(workflows.failure(args.workflow_action, type(error).__name__)))
        return 1
