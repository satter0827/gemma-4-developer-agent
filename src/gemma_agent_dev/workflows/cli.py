"""Present explicit CPU workflows using a common machine-readable result."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from gemma_agent_dev.workflows import service
from gemma_agent_dev.workflows.records import checked_path, exclusive, inspect
from gemma_agent_dev.workflows.transfer import fetch


def configure(commands: Any) -> None:
    parser = commands.add_parser("workflow", help="準備・診断・結果整理・候補準備を実行")
    actions = parser.add_subparsers(dest="workflow_action", required=True)
    for name in ("prepare", "diagnose", "report", "candidate", "status"):
        action = actions.add_parser(name)
        action.add_argument("--json", action="store_true", help="結果と次の操作をJSONで表示")
        action.set_defaults(handler=run)
        if name in {"prepare", "diagnose"}:
            action.add_argument("--config", type=Path, required=True)
            action.add_argument("--resume", type=Path)
        if name == "prepare":
            action.add_argument("--sdk-config", type=Path, required=True)
            action.add_argument("--quality-report", type=Path)
        if name == "status":
            action.add_argument("--run", type=Path, required=True)
        if name == "report":
            sources = action.add_mutually_exclusive_group(required=True)
            sources.add_argument("--input", type=Path)
            sources.add_argument("--remote-kernel")
            action.add_argument("--remote-version", type=int)
            action.add_argument("--remote-path")
            action.add_argument("--policy", type=Path)
            action.add_argument("--api-python", type=Path)
            action.add_argument("--review", type=Path)
            action.add_argument("--record-output", type=Path)
            action.add_argument("--tracking-config", type=Path)
        if name == "candidate":
            for field in ("cpu", "gpu", "source", "policy"):
                action.add_argument("--" + field, type=Path, required=True)
            action.add_argument("--variant", required=True)
            action.add_argument("--api-python", type=Path, required=True)


def run(root: Path, args: argparse.Namespace) -> int:
    try:
        with redirect_stdout(sys.stderr):
            result = execute(root, args)
    except Exception as error:
        result = failure(args.workflow_action, type(error).__name__)
        result["error"] = str(error)
        # Include the persisted failure receipt without guessing the latest directory.
        for note in getattr(error, "__notes__", []):
            if note.startswith("workflow_record="):
                path = Path(note.split("=", 1)[1])
                result.update(workflow_id=path.parent.name, artifacts={"workflow": str(path)})
                try:
                    record = json.loads(checked_path(root, path).read_text())
                    result.update(
                        stage=record["stage"],
                        artifacts={**record["artifacts"], "workflow": str(path)},
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        print(f"処理できません：{error}", file=sys.stderr)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"状態：{result['status']}\n工程：{result['stage']}")
        print(json.dumps(result["artifacts"], ensure_ascii=False, indent=2))
        print("次の操作：" + json.dumps(result["next_actions"], ensure_ascii=False))
    return int(result["status"] == "failed")


def failure(stage: str, reason: str) -> dict[str, Any]:
    return {
        "workflow_id": None,
        "status": "failed",
        "stage": stage,
        "artifacts": {},
        "reason_codes": [reason],
        "next_actions": [{"action": "inspect_inputs_and_workflow_record"}],
    }


def execute(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    def path(name: str) -> Path:
        return checked_path(root, getattr(args, name))

    if args.workflow_action == "status":
        return inspect(root, path("run"))
    with exclusive(root):
        if args.workflow_action == "prepare":
            return service.prepare_comparison(
                root,
                path("config"),
                path("sdk_config"),
                args.resume,
                quality_report=path("quality_report") if args.quality_report is not None else None,
            )
        if args.workflow_action == "diagnose":
            return service.diagnose_patch(root, path("config"), args.resume)
        if args.workflow_action == "report":
            source = args.input
            remote = (args.remote_version, args.remote_path, args.policy, args.api_python)
            if args.remote_kernel:
                if any(value is None for value in remote):
                    raise ValueError(
                        "Remote output requires version, manifest path, policy and SDK Python"
                    )
                source = fetch(
                    root,
                    args.remote_kernel,
                    args.remote_version,
                    args.remote_path,
                    path("policy"),
                    # Resolving a venv symlink selects its SDK-free base interpreter.
                    args.api_python.absolute(),
                )
            elif any(value is not None for value in remote):
                raise ValueError("Remote arguments require --remote-kernel")
            return service.report_results(
                root,
                checked_path(root, source),
                args.review,
                args.record_output,
                args.tracking_config,
            )
        return service.prepare_candidate(
            root,
            path("cpu"),
            path("gpu"),
            args.variant,
            path("source"),
            path("policy"),
            args.api_python.absolute(),
        )
