import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.artifacts import write_json


@pytest.fixture
def make_run(tmp_path: Path) -> Callable[..., Path]:
    def create(
        name: str = "run-a",
        *,
        ids: tuple[str, ...] = ("task_a", "task_b"),
        completed: int = 2,
        status: str = "completed",
    ) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": name,
            "task_ids": ids,
            "status": status,
            "evaluation": {"max_time_minutes": 1},
            "environment": {"sandbox": "subprocess"},
            "tasks_sha256": "fixture",
            "verification_timeout_seconds": 300,
            "vllm_config": {"max_model_len": 32768},
            "submission_files": {"agent.yaml": "fixture", "prompts/system.md": "fixture"},
            "runtime_config": {
                "model_source": "fixture",
                "docker_image": "fixture",
                "vllm": {},
                "context_cache": {},
                "compaction": {},
            },
        }
        write_json(directory / "manifest.json", manifest)
        rows = [
            {
                "instance_id": task,
                "resolved": index == 0,
                "duration_seconds": 2.5,
                "agent_patch_size": 10,
                "test_exit_code": 0 if index == 0 else 1,
                "error": None,
            }
            for index, task in enumerate(ids[:completed])
        ]
        (directory / "task_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        write_json(
            directory / "summary.json",
            {
                "total_tasks": len(rows),
                "resolved": sum(row["resolved"] for row in rows),
            },
        )
        return directory

    return create
