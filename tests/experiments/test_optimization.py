import copy
import importlib.metadata
import io
import json
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from gemma_agent_dev.common.artifacts import file_hash, inventory
from gemma_agent_dev.evaluation import optimization_candidates as candidates
from gemma_agent_dev.evaluation import optimization_engine as engine
from gemma_agent_dev.evaluation import optimization_preparation as preparation
from gemma_agent_dev.evaluation import optimization_runtime as runtime
from gemma_agent_dev.evaluation.optimization import Optimization
from gemma_agent_dev.evaluation.optimization_limits import SearchBudget, SearchLimits

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def optimization_project(preflight_project: Path) -> Path:
    root = preflight_project
    tasks = root / "tasks.jsonl"
    first = json.loads(tasks.read_text().splitlines()[0])
    second = {**first, "instance_id": "fixture_2"}
    tasks.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    catalog_path = root / "catalog.yaml"
    catalog = yaml.safe_load(catalog_path.read_text())
    catalog["tasks_sha256"] = file_hash(tasks)
    catalog["tasks"].append({**catalog["tasks"][0], "instance_id": "fixture_2"})
    catalog_path.write_text(yaml.safe_dump(catalog))
    diagnostic_path = root / "diagnostics.yaml"
    diagnostic = yaml.safe_load(diagnostic_path.read_text())
    diagnostic["tasks"].append(
        {**copy.deepcopy(diagnostic["tasks"][0]), "instance_id": "fixture_2"}
    )
    diagnostic_path.write_text(yaml.safe_dump(diagnostic))
    (root / "pool.yaml").write_text(
        yaml.safe_dump(
            dict(
                pool_id="fixture-pool",
                difficulty="Easy",
                tasks_sha256=file_hash(tasks),
                sdk_version="0.2.11",
                tuning=["fixture_1", "fixture_2"],
                confirmation=["unused_1"],
                uses=[
                    dict(
                        experiment_id="fixture-optimization",
                        purpose="tuning",
                        task_ids=["fixture_1", "fixture_2"],
                    )
                ],
            )
        )
    )
    config = yaml.safe_load((ROOT / "configs/evaluation/easy-gepa-v1.yaml").read_text())
    config.update(
        optimization_id="fixture-optimization",
        pool="pool.yaml",
        diagnostic_config="diagnostics.yaml",
        preflight_config="preflight.yaml",
        runtime_config="runtime.json",
        environment="gepa.yaml",
        search_task_ids=["fixture_1"],
        selection_task_ids=["fixture_2"],
    )
    (root / "optimization.yaml").write_text(yaml.safe_dump(config))
    environment = yaml.safe_load((ROOT / "configs/environments/gepa.yaml").read_text())
    wheel = root / environment["wheel_cache"] / environment["wheel_name"]
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"synthetic-wheel")
    environment["wheel_sha256"] = file_hash(wheel)
    (root / "gepa.yaml").write_text(yaml.safe_dump(environment))
    return root


def limits() -> SearchLimits:
    return SearchLimits.parse(
        yaml.safe_load((ROOT / "configs/evaluation/easy-gepa-v1.yaml").read_text())["limits"]
    )


def test_optimization_definition_is_independent_and_tuning_only(optimization_project: Path) -> None:
    spec = Optimization.load(optimization_project, Path("optimization.yaml"))
    assert spec.search_task_ids == ["fixture_1"] and spec.selection_task_ids == ["fixture_2"]
    definition = spec.definition(optimization_project)
    assert definition["gpu_budget"] is None
    assert definition["required_seconds"] > 5400
    assert spec.preflight(optimization_project).purpose == "optimization"


@pytest.mark.parametrize(
    "damage",
    [
        "keys",
        "missing",
        "overlap",
        "confirmation",
        "outside",
        "unreferenced",
        "seed-agent",
        "minibatch",
        "controls",
        "budget",
    ],
)
def test_invalid_optimization_inputs(optimization_project: Path, damage: str) -> None:
    path = optimization_project / "optimization.yaml"
    data = yaml.safe_load(path.read_text())
    if damage == "keys":
        data["unexpected"] = True
    elif damage == "missing":
        data["runtime_config"] = "missing.yaml"
    elif damage == "overlap":
        data["selection_task_ids"] = data["search_task_ids"]
    elif damage == "confirmation":
        data["selection_task_ids"] = ["unused_1"]
    elif damage == "outside":
        data["prompt_path"] = "../outside.md"
    elif damage == "unreferenced":
        data["prompt_path"] = "prompts/system.md"
    elif damage == "seed-agent":
        data["seed_agent"] = "agent/sub_agents/code_analyzer.yaml"
    elif damage == "minibatch":
        data["limits"]["reflection_minibatch_size"] = 2
    elif damage == "controls":
        p = optimization_project / "diagnostics.yaml"
        d = yaml.safe_load(p.read_text())
        d["tasks"].pop()
        p.write_text(yaml.safe_dump(d))
    else:
        data["gpu_budget"] = dict(
            quota_hours=4,
            reserve_hours=1,
            max_runs=1,
            quota_per_wall_hour=2,
            job_timeout_seconds=100,
        )
    path.write_text(yaml.safe_dump(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        Optimization.load(optimization_project, path)


def test_budget_admission_and_exact_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = limits()
    budget = SearchBudget(policy, 1000, 10)
    monkeypatch.setattr(time, "monotonic", lambda: 500)
    budget.check(cells=1, reflection=True)
    assert not budget.stop(None)
    budget.metric_calls = policy.max_metric_calls
    assert budget.stop(None)
    with pytest.raises(RuntimeError, match="evaluation"):
        budget.check(cells=1)
    budget.metric_calls = 0
    budget.reflection_calls = policy.max_reflection_calls
    with pytest.raises(RuntimeError, match="reflection"):
        budget.check(reflection=True)
    budget.deadline = 700
    with pytest.raises(TimeoutError):
        budget.check(cells=1)
    raw = asdict(policy)
    raw["seed"] = -1
    with pytest.raises(ValueError):
        SearchLimits.parse(raw)
    raw["seed"] = 0
    raw["max_metric_calls"] = False
    with pytest.raises(ValueError):
        SearchLimits.parse(raw)


@pytest.fixture
def fake_gepa(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("gepa")
    adapter_module = ModuleType("gepa.core.adapter")

    class Adapter:
        propose_new_texts = None

        @classmethod
        def __class_getitem__(cls, value: Any) -> Any:
            return cls

    adapter_module.GEPAAdapter = Adapter  # type: ignore[attr-defined]
    adapter_module.EvaluationBatch = SimpleNamespace  # type: ignore[attr-defined]

    def optimize(**kwargs: Any) -> Any:
        assert not kwargs["use_merge"] and not kwargs["use_wandb"] and not kwargs["use_mlflow"]
        adapter = kwargs["adapter"]
        original = kwargs["seed_candidate"]
        batch = adapter.evaluate(kwargs["trainset"], original, True)
        feedback = adapter.make_reflective_dataset(original, batch, ["instruction"])
        assert set(feedback["instruction"][0]) == {"Feedback"}
        assert "reference" not in json.dumps(feedback)
        text = kwargs["reflection_lm"]([{"role": "user", "content": "safe procedure feedback"}])
        candidate = {"instruction": text}
        adapter.evaluate(kwargs["valset"], candidate)
        assert kwargs["stop_callbacks"](None)
        return SimpleNamespace(best_candidate=candidate)

    module.optimize = optimize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gepa", module)
    monkeypatch.setitem(sys.modules, "gepa.core.adapter", adapter_module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.4")


@pytest.mark.parametrize("outcome", ["success", "unknown", "environment", "changed-component"])
def test_engine_never_scores_infrastructure_as_zero(fake_gepa: None, outcome: str) -> None:
    known: dict[str, str] = {}

    def freeze(text: str) -> str:
        known[text] = text
        return "candidate-" + str(len(known))

    def evaluate(key: str, task: str) -> dict[str, Any]:
        return dict(
            resolved=None if outcome == "unknown" else True,
            termination="execution_error" if outcome == "environment" else "completed",
            implementation_changed=None,
            tool_argument_errors=None,
        )

    if outcome == "changed-component":
        with pytest.raises(ValueError):
            engine.search(
                "seed",
                ["a"],
                ["a"],
                SearchBudget(limits(), time.monotonic() + 600, 0),
                freeze,
                evaluate,
                lambda _: "improved",
            )
    elif outcome != "success":
        with pytest.raises(RuntimeError):
            engine.search(
                "seed",
                ["a"],
                ["b"],
                SearchBudget(limits(), time.monotonic() + 600, 0),
                freeze,
                evaluate,
                lambda _: "improved",
            )
    else:
        report = engine.search(
            "seed",
            ["a"],
            ["b"],
            SearchBudget(limits(), time.monotonic() + 600, 0),
            freeze,
            evaluate,
            lambda _: "improved",
        )
        assert report["metric_calls"] == 2 and report["reflection_calls"] == 1
        assert report["improved"] and not report["adopted"]


@pytest.mark.parametrize(
    "value",
    [
        {},
        dict(
            resolved=False,
            termination="completed",
            implementation_changed=0,
            tool_argument_errors=None,
        ),
        dict(
            resolved=False,
            termination="completed",
            implementation_changed=None,
            tool_argument_errors=-1,
        ),
    ],
)
def test_invalid_search_outcomes(value: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        engine.SearchOutcome.parse(value)


def test_candidates_preserve_fixed_bytes_and_variables(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    (seed / "prompts").mkdir(parents=True)
    (seed / "agent.yaml").write_text("name: seed\ninstruction: !include prompts/main.md\n")
    (seed / "prompts/main.md").write_text("Original {problem_description}")
    (seed / "eval_config.yaml").write_text("fixed: true")
    before = inventory(seed)
    item = candidates.freeze(
        seed,
        tmp_path / "candidates",
        "prompts/main.md",
        "Improved {problem_description}",
        lambda _: None,
    )
    assert (
        inventory(seed) == before and item.files["eval_config.yaml"] == before["eval_config.yaml"]
    )
    assert item.sha256 == file_hash(item.directory.parent / "submission.zip")
    with pytest.raises(ValueError, match="variables"):
        candidates.freeze(
            seed, tmp_path / "invalid", "prompts/main.md", "Lost variable", lambda _: None
        )


def test_reflection_uses_only_loopback_and_checks_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter(
        [
            {"count": 20},
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "procedure"}}],
                "usage": {"completion_tokens": 10},
            },
        ]
    )
    requests = []

    def urlopen(request: Any, **kwargs: Any) -> Any:
        requests.append(request)
        return io.BytesIO(json.dumps(next(values)).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    budget = SearchBudget(limits(), time.monotonic() + 600, 0)
    assert (
        runtime.reflect(
            SimpleNamespace(base_url="http://127.0.0.1:8000/v1"),
            {"vllm": {"max_model_len": 32768}},
            budget,
            [{"role": "user", "content": "general feedback"}],
        )
        == "procedure"
    )
    assert requests[0].full_url.endswith("/tokenize") and requests[1].full_url.endswith(
        "/chat/completions"
    )
    with pytest.raises(ValueError, match="local"):
        runtime.reflect(SimpleNamespace(base_url="https://external.example/v1"), {}, budget, [])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b'{"count":999999}'))
    with pytest.raises(ValueError, match="input"):
        runtime.reflect(
            SimpleNamespace(base_url="http://localhost:8000/v1"),
            {"vllm": {"max_model_len": 32768}},
            budget,
            [],
        )


def test_gepa_environment_prepare_and_verify(
    optimization_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = optimization_project
    calls = []

    def command(args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        if args[:2] == ["uv", "venv"]:
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        return json.dumps({"python": "3.12.1", "packages": {"gepa": "0.1.4", "PyYAML": "6.0.3"}})

    monkeypatch.setattr(preparation, "command", command)
    path = preparation.prepare_environment(root, Path("gepa.yaml"))
    settings = preparation.GepaEnvironment.load(root, Path("gepa.yaml"))
    settings.verify(root)
    preparation.prepare_environment(root, Path("gepa.yaml"))
    assert sum(c[:2] == ["uv", "venv"] for c in calls) == 1
    data = json.loads(path.read_text())
    data["installed"]["packages"]["gepa"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        settings.verify(root)


@pytest.fixture
def frozen_optimization(optimization_project: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from gemma_agent_dev.common.artifacts import write_json

    root = optimization_project
    monkeypatch.setattr(preparation, "check_installation", lambda _: None)
    monkeypatch.setattr(preparation.GepaEnvironment, "verify", lambda *a: None)
    monkeypatch.setattr(preparation, "check_prepared", lambda _: None)
    monkeypatch.setattr(
        preparation,
        "verify_quality",
        lambda settings, output: (output / "quality.log").write_text("checked"),
    )

    def controls(settings: Any) -> Path:
        folder = root / "artifacts/control-proof"
        folder.mkdir(parents=True)
        cases = []
        for task in ["fixture_1", "fixture_2"]:
            for label in ["reference", "unmodified"]:
                target = folder / task / label
                target.mkdir(parents=True)
                (target / "result.txt").write_text("synthetic control evidence")
                record = dict(
                    task_id=task,
                    case=label,
                    matched=True,
                    passed=label == "reference",
                    artifacts=inventory(target),
                )
                write_json(target / "case.json", record)
                cases.append(record)
        write_json(folder / "report.json", dict(status="completed", cases=cases))
        return folder

    monkeypatch.setattr(preparation, "diagnose", controls)
    return preparation.prepare(root, Path("optimization.yaml"))


@pytest.mark.parametrize(
    "damage", ["none", "input", "archive", "case", "controls", "status", "outside"]
)
def test_optimization_freeze_binds_inputs_and_evidence(
    optimization_project: Path, frozen_optimization: Path, damage: str
) -> None:
    root = optimization_project
    report = json.loads(frozen_optimization.read_text())
    if damage == "none":
        spec, loaded = preparation.load_preparation(root, frozen_optimization)
        assert loaded["status"] == "completed" and spec.optimization_id == "fixture-optimization"
        return
    if damage == "input":
        (root / "agent/prompts/workflow.md").write_text("changed")
    elif damage == "archive":
        (frozen_optimization.parent / "seed/submission.zip").write_bytes(b"changed")
    elif damage == "case":
        (root / report["controls"]["path"]).parent.joinpath(
            "fixture_1/reference/result.txt"
        ).write_text("changed")
    elif damage == "controls":
        (root / report["controls"]["path"]).write_text("{}")
    elif damage == "status":
        report["status"] = "failed"
        frozen_optimization.write_text(json.dumps(report))
    else:
        with pytest.raises(ValueError):
            preparation.load_preparation(root, root / "outside.json")
        return
    with pytest.raises((ValueError, KeyError)):
        preparation.load_preparation(root, frozen_optimization)


def test_optimization_preparation_failure_is_persisted(
    optimization_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preparation, "check_installation", lambda _: None)
    monkeypatch.setattr(preparation.GepaEnvironment, "verify", lambda *a: None)
    monkeypatch.setattr(preparation, "check_prepared", lambda _: None)

    def fail(*args: Any) -> None:
        raise RuntimeError("quality failed")

    monkeypatch.setattr(preparation, "verify_quality", fail)
    with pytest.raises(RuntimeError, match="quality failed"):
        preparation.prepare(optimization_project, Path("optimization.yaml"))
    path = next((optimization_project / "artifacts/optimizations").glob("*/preparation.json"))
    assert json.loads(path.read_text())["status"] == "failed"


def test_optimization_bundle_is_closed_and_gpu_requires_explicit_policy(
    optimization_project: Path, frozen_optimization: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zipfile

    from gemma_agent_dev.candidates.archive import extract_archive
    from gemma_agent_dev.common.sources import validate_bundle_sources
    from gemma_agent_dev.notebooks import optimization as bundles

    root = optimization_project
    bundle = root / "bundle"
    digest = bundles.prepare(
        root, bundle, report_path=frozen_optimization, verify_only=True, cpu_report=None
    )
    with zipfile.ZipFile(bundle / "source-bundle.zip") as archive:
        names = set(archive.namelist())
        assert "logging.json" in names and "optimization.json" in names
        assert not any(
            "/logs/" in name or "tracking" in name or "diagnostics/local" in name for name in names
        )
    extract_archive(bundle / "source-bundle.zip", root / "unpacked", 10000000)
    validate_bundle_sources(root / "unpacked", list((root / "unpacked/src").rglob("*.py")))
    assert digest == file_hash(bundle / "source-bundle.zip")
    with pytest.raises(ValueError, match="explicitly configured"):
        bundles.prepare(
            root, root / "gpu", report_path=frozen_optimization, verify_only=False, cpu_report=None
        )
    monkeypatch.setattr(
        bundles,
        "load_preparation",
        lambda *a: (
            SimpleNamespace(
                **{
                    **Optimization.load(root, Path("optimization.yaml")).__dict__,
                    "gpu_budget": SimpleNamespace(
                        job_timeout_seconds=6000, admit=lambda **kw: {"allowed": True}
                    ),
                }
            ),
            json.loads(frozen_optimization.read_text()),
        ),
    )
    with pytest.raises(ValueError, match="CPU proof"):
        bundles.prepare(
            root,
            root / "gpu-missing",
            report_path=frozen_optimization,
            verify_only=False,
            cpu_report=None,
        )
    monkeypatch.setattr(bundles, "load_cpu", lambda *a: {})
    with pytest.raises(ValueError, match="quota"):
        bundles.prepare(
            root,
            root / "gpu-quota",
            report_path=frozen_optimization,
            verify_only=False,
            cpu_report=Path("proof"),
        )
    bundles.prepare(
        root,
        root / "gpu-ready",
        report_path=frozen_optimization,
        verify_only=False,
        cpu_report=Path("proof"),
        quota={"consumed": 0},
    )
    assert json.loads((root / "gpu-ready/kernel-metadata.json").read_text())["enable_gpu"] is True


@pytest.mark.parametrize("failure", ["none", "search", "wheel", "compile", "storage"])
def test_cpu_optimization_runtime_isolated_and_failure_preserving(
    optimization_project: Path,
    frozen_optimization: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from gemma_agent_dev.candidates.archive import extract_archive
    from gemma_agent_dev.common.artifacts import write_json
    from gemma_agent_dev.notebooks import optimization as bundles

    root = optimization_project
    bundle = root / "bundle"
    bundles.prepare(
        root, bundle, report_path=frozen_optimization, verify_only=True, cpu_report=None
    )
    extracted = root / "unpacked"
    extract_archive(bundle / "source-bundle.zip", extracted, 10000000)
    spec = json.loads((extracted / "optimization.json").read_text())
    calls = []

    def command(args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        if failure == "compile":
            raise RuntimeError("compile failed")
        return ""

    monkeypatch.setattr(runtime, "command", command)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.1.4")

    def copy_controls(path: Path, expected: str, ids: list[str], destination: Path) -> None:
        destination.mkdir()
        write_json(destination / "origins.json", {"valid": True})
        write_json(
            destination / "controls.json",
            [
                {
                    "instance_id": task,
                    "unmodified": {"resolved": False, "error": None},
                    "reference": {"resolved": True, "error": None},
                }
                for task in ids
            ],
        )

    monkeypatch.setattr(runtime, "copy_native_controls", copy_controls)

    def search(
        original: str,
        train: list[str],
        selection: list[str],
        budget: Any,
        freeze: Any,
        evaluate: Any,
        reflect: Any,
    ) -> dict[str, Any]:
        seed = freeze(original)
        assert evaluate(seed, train[0])["resolved"] is False
        assert "```" in reflect([])
        new = freeze(original + "\nAdditional verification.")
        assert evaluate(new, selection[0])["resolved"] is True
        if failure in {"search", "storage"}:
            raise RuntimeError("search failed")
        budget.metric_calls = 2
        budget.reflection_calls = 1
        return {
            "status": "completed",
            "improved": True,
            "candidate_ids": [seed, new],
            "best_candidate_id": new,
            "adopted": False,
            "evaluations": [
                {"candidate_id": seed, "task_id": train[0]},
                {"candidate_id": new, "task_id": selection[0]},
            ],
            "metric_calls": 2,
            "reflection_calls": 1,
        }

    monkeypatch.setattr(runtime, "search", search)
    if failure == "wheel":
        (extracted / "optimization-wheels" / spec["gepa"]["wheel_name"]).write_bytes(b"changed")
    if failure == "storage":
        original_write = write_json

        def save(path: Path, data: Any) -> None:
            if data.get("status") == "failed":
                raise OSError("disk failed")
            original_write(path, data)

        monkeypatch.setattr(runtime, "write_json", save)
    bindings = {
        "output": str(root / "execution"),
        "sdk_python": "synthetic-python",
        "native_controls": str(
            root / json.loads(frozen_optimization.read_text())["controls"]["path"]
        ),
    }
    if failure == "none":
        path = runtime.run(
            extracted, {"source_sha256": "0" * 64}, verify_only=True, bindings=bindings
        )
        report = json.loads(path.read_text())
        assert report["kind"] == "optimization-verification" and report["status"] == "completed"
        assert report["adopted"] is False and len(report["candidates"]) == 2 and calls
        assert (
            bundles.load_cpu(
                path, "0" * 64, spec["definition_sha256"], spec["candidate"]["sha256"]
            )["metric_calls"]
            == 2
        )
        (path.parent / "candidates" / report["best_candidate_id"] / "submission.zip").write_bytes(
            b"changed"
        )
        with pytest.raises(ValueError, match="candidate"):
            bundles.load_cpu(path, "0" * 64, spec["definition_sha256"], spec["candidate"]["sha256"])
        with pytest.raises(ValueError, match="bindings"):
            runtime.run(extracted, {}, verify_only=False, bindings=bindings)
    else:
        with pytest.raises((ValueError, RuntimeError)) as error:
            runtime.run(extracted, {}, verify_only=True, bindings=bindings)
        if failure == "storage":
            assert any("save optimization" in note for note in error.value.__notes__)


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "controls",
        "environment",
        "incomplete",
        "start",
        "search-and-stop",
        "deadline",
        "cell",
    ],
)
def test_gpu_optimization_boundaries_with_synthetic_server(
    optimization_project: Path,
    frozen_optimization: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from gemma_agent_dev.candidates.archive import extract_archive
    from gemma_agent_dev.common.artifacts import write_json
    from gemma_agent_dev.evaluation import official, session
    from gemma_agent_dev.notebooks import environment
    from gemma_agent_dev.notebooks import optimization as bundles

    root = optimization_project
    bundle = root / "bundle"
    bundles.prepare(
        root, bundle, report_path=frozen_optimization, verify_only=True, cpu_report=None
    )
    extracted = root / "unpacked"
    extract_archive(bundle / "source-bundle.zip", extracted, 10000000)
    spec = json.loads((extracted / "optimization.json").read_text())
    spec["definition"]["gpu_budget"] = {
        "job_timeout_seconds": 100 if failure == "deadline" else 6000
    }
    spec["definition"]["runtime"]["paths"]["output"] = str(root / "execution")
    write_json(extracted / "optimization.json", spec)
    calls = []
    parent = ModuleType("swegemma")
    parent.__path__ = []
    models = ModuleType("swegemma.models")
    models.load_tasks = lambda _: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "swegemma", parent)
    monkeypatch.setitem(sys.modules, "swegemma.models", models)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.1.4" if name == "gepa" else spec["definition"]["runtime"]["packages"][name],
    )
    monkeypatch.setattr(environment, "install_wheelhouse", lambda *a: calls.append("install"))
    monkeypatch.setattr(runtime, "command", lambda *a, **kw: "")
    monkeypatch.setattr(session, "prepare_candidates", lambda *a: {})
    server = SimpleNamespace(log_path=str(root / "model.log"))
    Path(server.log_path).write_text("synthetic model")
    monkeypatch.setattr(official, "create_server", lambda *a: server)
    monkeypatch.setattr(session, "evaluation_config", lambda *a: SimpleNamespace())
    monkeypatch.setattr(official, "make_validator", lambda *a: lambda _: None)

    async def controls(*args: Any) -> bool:
        write_json(args[2] / "controls.json", {"synthetic": True})
        return failure != "controls"

    monkeypatch.setattr(official, "verify_controls", controls)

    def start(*args: Any) -> float:
        calls.append("start")
        if failure == "start":
            raise RuntimeError("model start failed")
        return 0.1

    def stop(*args: Any) -> None:
        calls.append("stop")
        if failure == "search-and-stop":
            raise OSError("model stop failed")

    monkeypatch.setattr(official, "start_model", start)
    monkeypatch.setattr(official, "stop_model", stop)

    def cell(
        root: Path,
        folder: Path,
        policy: Any,
        cell_spec: Any,
        server: Any,
        submission: Any,
        cell: Any,
        *args: Any,
    ) -> dict[str, Any]:
        folder.mkdir(parents=True)
        if failure == "cell":
            write_json(folder / "manifest.json", {"status": "failed"})
            raise RuntimeError("cell failed")
        return {**cell, "manifest": str(folder / "manifest.json"), "sha256": "0" * 64}

    monkeypatch.setattr(session, "run_cell", cell)

    def saved(path: Path) -> Any:
        return SimpleNamespace(
            complete=failure != "incomplete",
            rows={
                "fixture_1": {
                    "resolved": False,
                    "error": "environment error" if failure == "environment" else None,
                }
            },
        )

    monkeypatch.setattr(runtime, "load_run", saved)
    monkeypatch.setattr(
        runtime,
        "task_evidence",
        lambda *a: dict(
            time_limit_reached=False,
            tool_limit_reached=False,
            trace_error=None,
            ambiguous_tool_observations=False,
            tool_argument_errors=[],
        ),
    )
    monkeypatch.setattr(runtime, "implementation_changed", lambda *a: True)
    monkeypatch.setattr(
        runtime,
        "behavior",
        lambda *a: dict(
            submission_count=None,
            nonempty_submission_then_empty_final=None,
            first_implementation_edit_trace_call=None,
            skill_script_calls=None,
        ),
    )
    monkeypatch.setattr(runtime, "reflect", lambda *a: "general procedure")

    def search(
        original: str,
        train: Any,
        selection: Any,
        budget: Any,
        freeze: Any,
        evaluate: Any,
        reflect: Any,
    ) -> dict[str, Any]:
        key = freeze(original)
        assert reflect([]) == "general procedure"
        evaluate(key, "fixture_1")
        if failure == "search-and-stop":
            raise ValueError("search failed")
        return dict(status="completed", improved=False, candidate_ids=[key], adopted=False)

    monkeypatch.setattr(runtime, "search", search)
    if failure == "none":
        path = runtime.run(extracted, {}, verify_only=False)
        assert json.loads(path.read_text())["status"] == "completed"
    else:
        with pytest.raises((RuntimeError, ValueError, TimeoutError)) as error:
            runtime.run(extracted, {}, verify_only=False)
        if failure == "search-and-stop":
            assert str(error.value) == "search failed"
    assert calls[-1] == "stop"
    if failure == "cell":
        report = json.loads(next((root / "execution").glob("*/search.json")).read_text())
        assert len(report["actual_order"]) == len(report["cells"]) == 1
    if failure in {"controls", "deadline"}:
        assert "start" not in calls


@pytest.mark.parametrize("damage", ["none", "metadata", "hash"])
def test_gepa_wheel_download_is_explicit_and_hash_checked(
    optimization_project: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    root = optimization_project
    settings = preparation.GepaEnvironment.load(root, Path("gepa.yaml"))
    settings.wheel().unlink()
    metadata = {
        "urls": [
            {
                "filename": settings.wheel_name if damage != "metadata" else "other.whl",
                "digests": {"sha256": settings.wheel_sha256},
                "url": "https://files.pythonhosted.org/synthetic.whl",
            }
        ]
    }
    responses = iter(
        [json.dumps(metadata).encode(), b"changed" if damage == "hash" else b"synthetic-wheel"]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(next(responses)))

    def command(args: list[str], **kwargs: Any) -> str:
        if args[:2] == ["uv", "venv"]:
            path = Path(args[-1]) / "bin/python"
            path.parent.mkdir(parents=True)
            path.touch()
        return json.dumps({"python": "3.12.1", "packages": {"gepa": "0.1.4"}})

    monkeypatch.setattr(preparation, "command", command)
    if damage == "none":
        assert preparation.prepare_environment(root, Path("gepa.yaml")).exists()
        assert settings.wheel().read_bytes() == b"synthetic-wheel"
    else:
        with pytest.raises(ValueError):
            preparation.prepare_environment(root, Path("gepa.yaml"))
        assert not (settings.wheel_cache / settings.wheel_name).exists()


@pytest.mark.parametrize("damage", ["none", "sdk", "bundle", "worker"])
def test_optimization_verification_runs_only_offline_cpu_worker(
    optimization_project: Path,
    frozen_optimization: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    from gemma_agent_dev.common.artifacts import write_json
    from gemma_agent_dev.notebooks import optimization as bundles

    root = optimization_project
    bundle = root / "bundle"
    digest = bundles.prepare(
        root, bundle, report_path=frozen_optimization, verify_only=True, cpu_report=None
    )
    sdk = root / "artifacts/sdk-env"
    sdk.mkdir(parents=True)
    write_json(
        sdk / "prepared.json",
        {
            "inputs": {"version": "wrong" if damage == "sdk" else "fixed"},
            "installed": {},
            "wheels": {},
        },
    )
    monkeypatch.setattr(
        bundles, "sdk_settings", lambda *a: dict(environment=sdk, wheel_cache=root / "wheels")
    )
    monkeypatch.setattr(bundles, "fingerprint", lambda *a: {"version": "fixed"})
    monkeypatch.setattr(bundles, "installed_state", lambda *a: {})
    proof = root / "artifacts/cpu-proof/search.json"
    proof.parent.mkdir(parents=True)
    prepared = json.loads(frozen_optimization.read_text())
    definition = prepared["definition"]

    def command(args: list[str], **kwargs: Any) -> str:
        assert "gemma_agent_dev.evaluation.optimization_worker" in args
        assert kwargs["env"]["UV_OFFLINE"] == "1" and kwargs["env"]["PIP_NO_INDEX"] == "1"
        if damage == "worker":
            raise RuntimeError("worker failed")
        controls = proof.parent / "controls"
        controls.mkdir()
        ids = definition["search_task_ids"] + definition["selection_task_ids"]
        write_json(controls / "origins.json", {"valid": True})
        write_json(
            controls / "controls.json",
            [
                {
                    "instance_id": task,
                    "reference": {"resolved": True},
                    "unmodified": {"resolved": False},
                }
                for task in ids
            ],
        )
        candidate = proof.parent / "candidates/generated"
        (candidate / "submission").mkdir(parents=True)
        (candidate / "submission/agent.yaml").write_text("name: compiled")
        (candidate / "submission.zip").write_bytes(b"synthetic-cpu-compiled-candidate")
        report = dict(
            kind="optimization-verification",
            status="completed",
            source_sha256=digest,
            definition_sha256=prepared["definition_sha256"],
            definition=definition,
            task_ids=ids,
            candidate=prepared["candidate"],
            improved=True,
            adopted=False,
            metric_calls=1,
            reflection_calls=1,
            evaluations=[{"candidate_id": "generated", "task_id": ids[0]}],
            best_candidate_id="generated",
            controls=inventory(controls),
            candidates={
                "generated": {
                    "sha256": file_hash(candidate / "submission.zip"),
                    "files": inventory(candidate / "submission"),
                }
            },
        )
        write_json(proof, report)
        return str(proof) + "\n"

    monkeypatch.setattr(bundles, "command", command)
    if damage == "bundle":
        (bundle / "source-bundle.zip").write_bytes(b"changed")
    if damage == "none":
        assert bundles.verify_local(root, bundle, Path("sdk.yaml")) == proof
    else:
        with pytest.raises((ValueError, RuntimeError)):
            bundles.verify_local(root, bundle, Path("sdk.yaml"))


def test_completed_context_budget_outcome_uses_the_official_result() -> None:
    value = engine.SearchOutcome.parse(
        dict(
            resolved=False,
            termination="context_window_exceeded",
            implementation_changed=None,
            tool_argument_errors=None,
        )
    )
    assert value.feedback()["resolved"] is False and value.termination == "context_window_exceeded"


def test_optimization_identity_ignores_other_reservations_but_checks_selected_role(
    optimization_project: Path,
) -> None:
    root = optimization_project
    before = Optimization.load(root, Path("optimization.yaml")).definition(root)
    pool_path = root / "pool.yaml"
    data = yaml.safe_load(pool_path.read_text())
    data["uses"].append(
        dict(experiment_id="unrelated", purpose="confirmation", task_ids=["unused_1"])
    )
    pool_path.write_text(yaml.safe_dump(data))
    assert Optimization.load(root, Path("optimization.yaml")).definition(root) == before
    data["tuning"].remove("fixture_2")
    data["confirmation"].append("fixture_2")
    data["uses"][0]["task_ids"] = ["fixture_1"]
    pool_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        Optimization.load(root, Path("optimization.yaml"))
