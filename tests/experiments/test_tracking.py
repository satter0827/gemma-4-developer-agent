import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.evaluation import tracking, tracking_worker

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tracking_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path
    (root / "agent/prompts").mkdir(parents=True)
    (root / "agent/prompts/test.md").write_text("Authored {problem_description}")
    (root / "requirements.in").write_text("mlflow-skinny==3.17.2")
    (root / "requirements.lock").write_text("pinned hashes")
    config = {
        "requirements": "requirements.in",
        "lock_file": "requirements.lock",
        "environment": "artifacts/tracking-env",
        "database": "artifacts/tracking/db.sqlite",
        "output_dir": "artifacts/tracking-checks",
        "python_version": "3.12",
        "timeout_seconds": 30,
        "experiment_name": "fixture",
    }
    path = root / "tracking.yaml"
    path.write_text(yaml.safe_dump(config))
    record = {
        "record_id": "fixture-cpu-v1",
        "experiment_id": "fixture-v1",
        "hypothesis": "public-api-check",
        "status": "cpu-verified",
        "task_ids": ["task_1"],
        "definition_sha256": "0" * 64,
        "source_sha256": "4" * 64,
        "candidate_hashes": {"a": "1" * 64, "b": "2" * 64},
        "metrics": {"controls": 2},
        "interpretation": "CPU controls passed; accuracy not evaluated.",
        "prompts": [
            {
                "prompt_id": "test-v1",
                "path": "agent/prompts/test.md",
                "sha256": file_hash(root / "agent/prompts/test.md"),
            }
        ],
    }
    record_path = root / "record.json"
    write_json(record_path, record)
    return root, path, record_path


@pytest.mark.parametrize(
    "damage",
    [
        "keys",
        "status",
        "metrics",
        "infinite",
        "interpretation",
        "hash",
        "prompt",
        "outside",
        "duplicate-prompt",
        "candidates",
        "prompts",
    ],
)
def test_public_schema_rejects_unsafe_or_invalid_records(
    tracking_project: Any, damage: str
) -> None:
    root, config, path = tracking_project
    d = json.loads(path.read_text())
    if damage == "keys":
        d["raw_trace"] = "forbidden"
    elif damage == "status":
        d["status"] = "success"
    elif damage == "metrics":
        d["metrics"] = {"payload": "raw text"}
    elif damage == "infinite":
        d["metrics"] = {"value": float("nan")}
    elif damage == "interpretation":
        d["interpretation"] = "two\nlines"
    elif damage == "hash":
        d["definition_sha256"] = "bad"
    elif damage == "prompt":
        d["prompts"][0]["sha256"] = "1" * 64
    elif damage == "outside":
        d["prompts"][0]["path"] = "requirements.in"
    elif damage == "duplicate-prompt":
        d["prompts"] *= 2
    elif damage == "candidates":
        d["candidate_hashes"] = {}
    else:
        d["prompts"] = None
    write_json(path, d)
    with pytest.raises(ValueError):
        tracking.load_records(root, [path])


def test_duplicate_git_records_rejected(tracking_project: Any) -> None:
    root, config, path = tracking_project
    assert tracking.load_records(root, [path])[0]["metrics"] == {"controls": 2}
    with pytest.raises(ValueError, match="Duplicate"):
        tracking.load_records(root, [path, path])


@pytest.mark.parametrize(
    "damage", ["environment", "database", "output_dir", "overlap", "python", "timeout", "name"]
)
def test_tracking_settings_reject_bad_storage(tracking_project: Any, damage: str) -> None:
    root, path, _ = tracking_project
    d = yaml.safe_load(path.read_text())
    if damage in {"environment", "database", "output_dir"}:
        d[damage] = "src/unsafe"
    elif damage == "overlap":
        d["database"] = "artifacts/tracking-env/db"
    elif damage == "python":
        d["python_version"] = "3.13"
    elif damage == "timeout":
        d["timeout_seconds"] = 0
    else:
        d["experiment_name"] = "bad/name"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        tracking.TrackingSettings.load(root, path)


def test_prepare_check_and_sync_are_separate(
    tracking_project: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, record = tracking_project
    calls = []

    def command(args: list[str], **kw: Any) -> str:
        calls.append(args)
        if "-c" in args:
            return json.dumps({"python": "3.12.1", "packages": {"mlflow-skinny": "3.17.2"}})
        if args[:2] == ["uv", "venv"]:
            p = Path(args[-1]) / "bin/python"
            p.parent.mkdir(parents=True)
            p.touch()
        if "-m" in args:
            request = json.loads(Path(args[-2]).read_text())
            assert "template" in request["records"][0]["prompts"][0]
            assert not any(k in request["records"][0] for k in ["logs", "patch", "trace"])
            write_json(Path(args[-1]), {"status": "completed"})
        return "ok"

    monkeypatch.setattr(tracking, "command", command)
    receipt = tracking.prepare(root, config)
    assert receipt.exists()
    assert sum(c[:2] == ["uv", "venv"] for c in calls) == 1
    tracking.prepare(root, config)
    assert sum(c[:2] == ["uv", "venv"] for c in calls) == 1
    calls.clear()
    output = tracking.sync(root, config, [record])
    assert (output / "report.json").exists() and not (output / "request.json").exists()
    assert not any(c[0] == "uv" for c in calls)
    with pytest.raises(ValueError, match="at least"):
        tracking.sync(root, config, [])
    (root / "requirements.lock").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        tracking.sync(root, config, [record])
    monkeypatch.setattr(
        tracking,
        "installed",
        lambda *a: {"python": "3.12.1", "packages": {"mlflow-skinny": "changed"}},
    )
    with pytest.raises(ValueError, match="installation"):
        tracking.prepare(root, config)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "runs": [],
        "prompts": {},
        "metrics": [],
        "error": False,
        "exists": False,
    }

    class Page(list[Any]):
        token = None

    class Client:
        def __init__(self, **kw: Any) -> None:
            assert kw["tracking_uri"].startswith("sqlite:///")

        def get_experiment_by_name(self, name: str) -> Any:
            return SimpleNamespace(experiment_id="1") if state["exists"] else None

        def create_experiment(self, name: str) -> str:
            state["exists"] = True
            return "1"

        def search_runs(self, ids: Any, **kw: Any) -> Any:
            return Page(state["runs"])

        def create_run(self, experiment_id: str, tags: Any) -> Any:
            r = SimpleNamespace(
                info=SimpleNamespace(run_id=str(len(state["runs"])), status="RUNNING"),
                data=SimpleNamespace(tags=tags.copy()),
            )
            state["runs"].append(r)
            return r

        def set_tag(self, id: str, k: str, v: Any) -> None:
            state["runs"][int(id)].data.tags[k] = v

        def log_param(self, *a: Any) -> None:
            pass

        def log_metric(self, *a: Any) -> None:
            if state["error"]:
                raise RuntimeError("storage failed")
            state["metrics"].append(a)

        def set_terminated(self, id: str, status: str) -> None:
            if status == "FAILED" and state.get("termination_error"):
                raise OSError("status storage failed")
            state["runs"][int(id)].info.status = status

    def load(name: str, **kw: Any) -> Any:
        return state["prompts"].get(name)

    def register(name: str, template: str, tags: Any) -> None:
        state["prompts"][name] = SimpleNamespace(tags=tags, template=template)

    module: Any = types.ModuleType("mlflow")
    module.MlflowClient = Client
    module.set_tracking_uri = lambda uri: None
    module.set_registry_uri = lambda uri: None
    module.genai = SimpleNamespace(load_prompt=load, register_prompt=register)
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return state


def request(root: Path, path: Path) -> dict[str, Any]:
    rows = tracking.load_records(root, [path])
    r = rows[0]
    r["content_sha256"] = "3" * 64
    for p in r["prompts"]:
        p["template"] = (root / p["path"]).read_text()
    return {
        "database": str(root / "artifacts/test.db"),
        "experiment_name": "fixture",
        "records": rows,
    }


def test_worker_idempotence_and_immutable_content(fake_mlflow: Any, tracking_project: Any) -> None:
    root, config, path = tracking_project
    r = request(root, path)
    assert tracking_worker.synchronize(r)["created"] == ["fixture-cpu-v1"]
    assert tracking_worker.synchronize(r)["skipped"] == ["fixture-cpu-v1"]
    assert len(fake_mlflow["runs"]) == 1 and len(fake_mlflow["metrics"]) == 1
    r["records"][0]["content_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="different content"):
        tracking_worker.synchronize(r)
    fake_mlflow["runs"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        tracking_worker.synchronize(r)


def test_worker_prompt_conflict_and_retry_failed_projection(
    fake_mlflow: Any, tracking_project: Any
) -> None:
    root, config, path = tracking_project
    r = request(root, path)
    fake_mlflow["prompts"]["test-v1"] = SimpleNamespace(tags={"source_sha256": "changed"})
    with pytest.raises(ValueError, match="prompt ID"):
        tracking_worker.synchronize(r)
    fake_mlflow["prompts"].clear()
    fake_mlflow["error"] = True
    with pytest.raises(RuntimeError, match="storage failed"):
        tracking_worker.synchronize(r)
    assert fake_mlflow["runs"][0].info.status == "FAILED"
    fake_mlflow["error"] = False
    assert (
        tracking_worker.synchronize(r)["created"] == ["fixture-cpu-v1"]
        and len(fake_mlflow["runs"]) == 1
    )


def test_network_guard_and_worker_failure_report(
    fake_mlflow: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_worker.deny_network("open", ())
    with pytest.raises(RuntimeError, match="forbidden"):
        tracking_worker.deny_network("socket.connect", ())
    p = tmp_path / "request.json"
    p.write_text("{}")
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["worker", str(p), str(out)])
    monkeypatch.setattr(sys, "addaudithook", lambda f: None)
    with pytest.raises(KeyError):
        tracking_worker.main()
    assert json.loads(out.read_text())["status"] == "failed"


def test_experiment_conditions_are_immutable_across_record_ids(
    fake_mlflow: Any, tracking_project: Any
) -> None:
    root, config, path = tracking_project
    r = request(root, path)
    tracking_worker.synchronize(r)
    r["records"][0]["record_id"] = "second-result"
    r["records"][0]["definition_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="experiment ID"):
        tracking_worker.synchronize(r)


def test_git_records_are_valid_without_mlflow_or_competition_data() -> None:
    paths = sorted((ROOT / "docs/records/experiments").glob("*.json"))
    rows = tracking.load_records(ROOT, paths)
    from gemma_agent_dev.evaluation.experiments import Experiment, content_hash

    for row in rows:
        definition = Experiment.load(ROOT, Path(f"configs/evaluation/{row['experiment_id']}.yaml"))
        assert row["experiment_id"] == definition.experiment_id
        assert row["definition_sha256"] == content_hash(definition.definition(ROOT))
        assert set(row["candidate_hashes"]) == set(definition.variants)


def test_storage_failure_does_not_replace_original_error(
    fake_mlflow: Any, tracking_project: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, config, path = tracking_project
    fake_mlflow.update(error=True, termination_error=True)
    with pytest.raises(RuntimeError, match="storage failed") as failure:
        tracking_worker.synchronize(request(root, path))
    assert any("Cannot mark projection failed" in note for note in failure.value.__notes__)
    p = tmp_path / "request.json"
    p.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["worker", str(p), str(tmp_path / "report.json")])
    monkeypatch.setattr(sys, "addaudithook", lambda f: None)

    def failed_write(*args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tracking_worker, "write_json", failed_write)
    with pytest.raises(KeyError) as original:
        tracking_worker.main()
    assert any("Cannot save synchronization failure" in note for note in original.value.__notes__)
