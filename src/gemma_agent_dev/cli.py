"""Command-line entry points for packaging Kaggle code and browsing results."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-kaggle", help="Build a private Kaggle script")
    prepare.add_argument("--root", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    view = commands.add_parser("view", help="Browse local evaluation artifacts")
    view.add_argument("--results", type=Path, default=Path("results"))
    view.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if args.command == "prepare-kaggle":
        from gemma_agent_dev.kaggle_bundle import prepare as build_bundle

        print(build_bundle(args.root.resolve(), args.output.resolve()))
    else:
        environment = {**os.environ, "GEMMA_RESULTS_DIR": str(args.results.resolve())}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).with_name("app.py")),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(args.port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            env=environment,
            check=True,
        )
