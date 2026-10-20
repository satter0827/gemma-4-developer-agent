import importlib.metadata
import json
import logging
import shutil
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gemma_agent_dev import official, runtime
from gemma_agent_dev.artifacts import file_hash, inventory, write_json
from gemma_agent_dev.official import select_tasks


def test_selected_controls_match_evaluation_order() -> None:
    tasks = [
        SimpleNamespace(instance_id=name, repo="fixture/repo", patch="fix", test_patch="test")
        for name in ("other", "b", "a")
    ]
    assert [task.instance_id for task in select_tasks(tasks, ["a", "b"])] == ["a", "b"]
    for ids in ([], ["a", "a"], ["missing"]):
        with pytest.raises(ValueError):
            select_tasks(tasks, ids)
    tasks[1].patch = ""
    with pytest.raises(ValueError, match="reference"):
        select_tasks(tasks, ["b"])
    with pytest.raises(ValueError, match="Duplicate"):
        select_tasks([tasks[2], tasks[2]], ["a"])


@dataclass
class ServerConfig:
    model: str = "fixture"


@pytest.mark.parametrize(
    "verify_only,save_results,logged_error,startup_error,final_save_error,stop_error,candidate_changed",
    [
        (True, False, False, False, False, False, False),
        (False, True, False, False, False, False, False),
        (False, True, False, False, True, False, False),
        (False, True, False, False, False, True, False),
        (False, False, False, False, False, False, False),
        (False, True, True, False, False, False, False),
        (False, False, False, True, False, False, False),
        (True, False, False, False, False, False, True),
        (False, True, False, False, False, False, True),
    ],
)
def test_pipeline_verifies_before_model_start_and_validates_saved_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verify_only: bool,
    save_results: bool,
    logged_error: bool,
    startup_error: bool,
    final_save_error: bool,
    stop_error: bool,
    candidate_changed: bool,
) -> None:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "agent", root / "agent")
    data = tmp_path / "data"
    (data / "sample_submission").mkdir(parents=True)
    (data / "tasks.jsonl").write_text("fixture")
    config = json.loads((source / "configs/runtime.json").read_text())
    config["paths"]["data"] = str(data)
    config["paths"]["output"] = str(tmp_path / "results")
    (root / "configs/runtime.json").write_text(json.dumps(config))
    config["paths"]["submission_export"] = str(tmp_path / "submission.zip")
    (root / "configs/runtime.json").write_text(json.dumps(config))
    with zipfile.ZipFile(root / "candidate.zip", "w") as archive:
        for path in (root / "agent").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root / "agent"))
    candidate_sha = file_hash(root / "candidate.zip")
    (data / "snapshots").mkdir()
    for task in config["task_ids"]:
        (data / "snapshots" / (task + ".tgz")).write_bytes(b"fixture")
    (root / "candidate.json").write_text(
        json.dumps(
            {
                "sha256": candidate_sha,
                "files": inventory(root / "agent"),
                "max_unpacked_bytes": 1000000,
                "tasks_sha256": file_hash(data / "tasks.jsonl"),
                "snapshots": {
                    task: file_hash(data / "snapshots" / (task + ".tgz"))
                    for task in config["task_ids"]
                },
            }
        )
    )
    original_write = write_json

    def write(path: Path, value: Any) -> None:
        if final_save_error and value.get("status") == "completed":
            raise OSError("Final save failed")
        original_write(path, value)

    monkeypatch.setattr(runtime, "write_json", write)
    monkeypatch.chdir(root)
    monkeypatch.setattr(runtime, "install_wheelhouse", lambda *_: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: config["packages"][name])
    calls: list[str] = []
    configurations = []

    def module(name: str, **attributes: Any) -> None:
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key not in sys.modules:
                item = types.ModuleType(key)
                item.__path__ = []
                monkeypatch.setitem(sys.modules, key, item)
        item = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(item, key, value)
        monkeypatch.setitem(sys.modules, name, item)

    class Evaluator:
        def __init__(self, configuration: Any) -> None:
            self.config = configuration
            self.sandbox = None
            configurations.append(configuration)

        async def run(self) -> Any:
            calls.append("evaluate")
            if save_results:
                rows = [
                    dict(
                        instance_id=task,
                        resolved=index == 0,
                        duration_seconds=1,
                        agent_patch_size=1,
                        test_exit_code=0 if index == 0 else 1,
                        error=None,
                    )
                    for index, task in enumerate(self.config.task_ids)
                ]
                (self.config.results_dir / "task_results.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                (self.config.results_dir / "summary.json").write_text(
                    json.dumps(dict(total_tasks=2, resolved=1))
                )
                for row in rows:
                    task = row["instance_id"]
                    for name, text in {
                        f"logs/{task}.log": "agent log",
                        f"traces/trace_{task}.json": "{}",
                        f"patches/{task}.patch": "x",
                        f"test_outputs/{task}.log": "test log",
                    }.items():
                        path = self.config.results_dir / name
                        path.parent.mkdir(exist_ok=True)
                        path.write_text(text)
            if logged_error:
                logging.getLogger("swegemma.evaluate").error(
                    "Failed to save final evaluation results: disk failure"
                )
            return SimpleNamespace(total=2, resolved=1)

    async def verify(_sandbox: Any, _config: Any, task: Any, *_: Any, **kwargs: Any) -> Any:
        assert task.instance_id in config["task_ids"]
        calls.append("reference" if kwargs["agent_patch"] else "unmodified")
        resolved = bool(kwargs["agent_patch"])
        if candidate_changed:
            (_config.submission_dir / "prompts/system.md").write_text("changed")
        return SimpleNamespace(
            resolved=resolved,
            test_exit_code=0 if resolved else 1,
            error=None,
            duration_seconds=1,
            test_output="verification log",
        )

    class Server:
        adapter_manifest = None
        config = ServerConfig()
        log_path = str(tmp_path / "server.log")

        def create_model_registry(self, **_: Any) -> dict[str, str]:
            return {}

        def start(self) -> None:
            assert calls == ["unmodified", "reference"] * 2
            calls.append("start")
            if startup_error:
                self.stop()
                raise RuntimeError("Server startup failed")

        def stop(self) -> None:
            calls.append("stop")
            Path(self.log_path).unlink(missing_ok=True)
            if stop_error:
                raise RuntimeError("Stop failed")

    Path(Server.log_path).write_text("server log")
    tasks = [
        SimpleNamespace(instance_id=name, repo="fixture/repo", patch="fix", test_patch="test")
        for name in ["unrelated", *config["task_ids"]]
    ]
    module("litellm")

    def device_count() -> int:
        assert not verify_only
        return 4

    module(
        "torch", cuda=SimpleNamespace(device_count=device_count, get_device_name=lambda _: "GPU")
    )
    module("google.adk.agents.context_cache_config", ContextCacheConfig=lambda **kw: kw)
    module("google.adk.apps._configs", EventsCompactionConfig=lambda **kw: kw)
    module("swegemma.config", EvalConfig=lambda **kw: SimpleNamespace(**kw))
    module("swegemma.models", load_tasks=lambda _: tasks)
    module("swegemma.evaluate", Evaluator=Evaluator)
    module(
        "swegemma.deduplication",
        resolve_task_snapshot_paths=lambda _dir, task, _repo: (
            data / "snapshots" / (task + ".tgz"),
            None,
            None,
        ),
    )
    module("swegemma.harness.verification", verify_task=verify)
    monkeypatch.setattr(official, "create_server", lambda *_: Server())
    monkeypatch.setattr(official, "make_validator", lambda *_: lambda _: None)
    handlers = list(logging.getLogger("swegemma.evaluate").handlers)
    failed = not verify_only and (
        not save_results or logged_error or startup_error or final_save_error or stop_error
    )
    failed = failed or candidate_changed
    if failed:
        message = "artifacts"
        if logged_error:
            message = "reported errors"
        if startup_error:
            message = "startup failed"
        if final_save_error:
            message = "Final save failed"
        if stop_error:
            message = "Stop failed"
        if candidate_changed:
            message = "Validated submission changed"
        with pytest.raises((RuntimeError, OSError, ValueError), match=message):
            runtime.run(
                root,
                {"source_sha256": "fixture", "candidate_sha256": candidate_sha},
                verify_only=verify_only,
            )
    else:
        runtime.run(
            root,
            {"source_sha256": "fixture", "candidate_sha256": candidate_sha},
            verify_only=verify_only,
        )
    assert logging.getLogger("swegemma.evaluate").handlers == handlers
    output = next((tmp_path / "results").iterdir())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == ("failed" if failed else "completed")
    assert configurations[0].timeout_seconds == 60
    assert configurations[0].verification_timeout_seconds == 300
    assert calls.count("start") == int(not verify_only)
    assert calls[-1] == "stop"
    assert (output / "vllm.log").read_text() == "server log"
    assert not Path(Server.log_path).exists()
    assert (tmp_path / "submission.zip").exists() is not failed
    if not failed:
        assert file_hash(tmp_path / "submission.zip") == candidate_sha
    if verify_only:
        assert manifest["kind"] == "verification"
        assert "evaluate" not in calls
        assert not (output / "task_results.jsonl").exists()


def test_evaluation_requires_trace_patch_and_test_logs(
    tmp_path: Path,
    make_run: Any,
) -> None:
    path = make_run()
    result = SimpleNamespace(total=2, resolved=1)
    with pytest.raises(ValueError, match="Missing evaluation"):
        runtime.validate_evaluation(path, result)
    for task in ("task_a", "task_b"):
        for name, content in {
            f"logs/{task}.log": "log",
            f"traces/trace_{task}.json": "{}",
            f"patches/{task}.patch": "patch",
            f"test_outputs/{task}.log": "test log",
        }.items():
            target = path / name
            target.parent.mkdir(exist_ok=True)
            target.write_text(content)
    hashes = runtime.validate_evaluation(path, result)
    assert "traces/trace_task_a.json" in hashes
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_artifacts"] = hashes
    manifest_path.write_text(json.dumps(manifest))
    from gemma_agent_dev.results import load_run

    assert load_run(path).complete
    (path / "logs/task_a.log").write_text("changed")
    assert not load_run(path).complete
