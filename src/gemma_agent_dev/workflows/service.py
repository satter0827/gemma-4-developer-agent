"""Compose existing preparation, diagnostics, analysis and candidate services."""

import json
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.experiment_evidence import qualify_experiment
from gemma_agent_dev.candidates.preflight import artifact_limit
from gemma_agent_dev.common.artifacts import file_hash, python_inventory, write_json
from gemma_agent_dev.diagnostics.config import DiagnosticSettings
from gemma_agent_dev.diagnostics.local import check_prepared, diagnose, prepare
from gemma_agent_dev.evaluation import sdk_checks
from gemma_agent_dev.evaluation.analysis import analyze
from gemma_agent_dev.evaluation.experiment_preparation import load_preparation, selected_settings
from gemma_agent_dev.evaluation.experiment_preparation import prepare as freeze
from gemma_agent_dev.evaluation.experiment_records import export
from gemma_agent_dev.evaluation.experiment_results import load_cpu, summarize
from gemma_agent_dev.evaluation.experiments import Experiment
from gemma_agent_dev.evaluation.results import load_run
from gemma_agent_dev.evaluation.tracking import sync
from gemma_agent_dev.notebooks.experiments import prepare as bundle
from gemma_agent_dev.notebooks.experiments import verify_local
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.submission.config import SubmissionSettings
from gemma_agent_dev.workflows.records import Flow, checked_path


def input_files(root: Path, paths: list[Path]) -> dict[str, str]:
    files = {
        "src/gemma_agent_dev/" + key: value
        for key, value in python_inventory(root / "src/gemma_agent_dev").items()
    }
    for path in paths:
        resolved = checked_path(root, path)
        files[str(resolved.relative_to(root))] = file_hash(resolved)
    return files


def task_environment(settings: DiagnosticSettings) -> Path:
    receipt = settings.cache_dir / "prepared.json"
    if (
        not receipt.exists()
        or json.loads(receipt.read_text())["fingerprint"] != settings.preparation_fingerprint()
    ):
        prepare(settings)
    check_prepared(settings)
    return receipt


def sdk_environment(root: Path, config: Path) -> Path:
    data = sdk_checks.settings(root, config)
    receipt = data["environment"] / "prepared.json"
    if not receipt.exists() or json.loads(receipt.read_text())["inputs"] != sdk_checks.fingerprint(
        data
    ):
        sdk_checks.prepare(root, config)
    state = json.loads(receipt.read_text())
    if state["installed"] != sdk_checks.installed_state(root, data):
        raise ValueError("SDK environment changed; prepare it explicitly")
    return Path(receipt)


def prepare_comparison(
    root: Path,
    config: Path,
    sdk_config: Path,
    resume: Path | None,
    *,
    quality_report: Path | None = None,
) -> dict[str, Any]:
    experiment = Experiment.load(root, config)
    definition = experiment.definition(root)
    settings = selected_settings(root, experiment)
    inputs = {
        "definition": definition,
        "sdk": sdk_checks.fingerprint(sdk_checks.settings(root, sdk_config)),
        "candidate_inputs": {name: item.inputs() for name, item in settings.items()},
    }
    if quality_report is not None:
        from gemma_agent_dev.candidates.quality import reuse

        inputs["quality"] = reuse(root, quality_report)
    files = input_files(
        root, [config, sdk_config] + ([quality_report] if quality_report is not None else [])
    )
    for item in settings.values():
        files.update(
            {name: digest for name, digest in item.inputs().items() if not name.startswith("@")}
        )
    with Flow(root, "prepare", inputs, files, resume) as flow:
        diagnostic = DiagnosticSettings.load(root, experiment.diagnostic_config)
        flow.step("task_environment", lambda: task_environment(diagnostic))
        check_prepared(diagnostic)
        flow.step("sdk_environment", lambda: sdk_environment(root, sdk_config))
        sdk_environment(root, sdk_config)
        preparation = flow.step(
            "freeze",
            lambda: (
                freeze(root, config, quality_report=quality_report)
                if quality_report is not None
                else freeze(root, config)
            ),
        )
        load_preparation(root, preparation)
        flow.report["completion"]["quality"] = "completed"

        def make_bundle() -> Path:
            destination = flow.path.parent / "cpu-bundle"
            # Each failed attempt gets a new destination; existing partial bundles stay intact.
            from gemma_agent_dev.common.artifacts import new_run_id

            destination = destination / new_run_id()
            bundle(
                root,
                destination,
                report_path=preparation,
                verify_only=True,
                cpu_report=None,
                local_only=True,
            )
            return destination / "source-manifest.json"

        metadata_path = flow.step("bundle", make_bundle)
        cpu = flow.step(
            "cpu_verification", lambda: verify_local(root, metadata_path.parent, sdk_config)
        )
        flow.report["artifacts"]["source_bundle"] = str(metadata_path.parent / "source-bundle.zip")
        metadata = json.loads(metadata_path.read_text())
        load_cpu(
            cpu,
            metadata["source_sha256"],
            metadata["definition_sha256"],
            experiment.task_ids,
            {
                name: value["sha256"]
                for name, value in json.loads(preparation.read_text())["variants"].items()
            },
        )
        flow.report["completion"]["cpu_verification"] = "completed"
        flow.report["completion_budget"] = experiment.session.completion(
            definition["evaluation"], definition["runtime"], len(experiment.task_ids)
        )
        return flow.finish(
            "cpu_verified",
            [
                {
                    "action": "gpu_bundle",
                    "requires": "observed_quota_and_complete_time_budget",
                    "optional_argv": ["--bundle-dataset", "PRIVATE_SOURCE_DATASET"],
                    "dataset_file_name": "source-bundle.bin",
                    "source_bundle": str(metadata_path.parent / "source-bundle.zip"),
                    "argv": [
                        "gemma-dev",
                        "prepare-kaggle",
                        "--name",
                        experiment.experiment_id + "-gpu",
                        "--experiment-report",
                        str(preparation),
                        "--cpu-report",
                        str(cpu),
                        "--quota",
                        "OBSERVED_QUOTA_JSON",
                    ],
                }
            ],
        )


def diagnose_patch(root: Path, config: Path, resume: Path | None) -> dict[str, Any]:
    settings = DiagnosticSettings.load(root, config)
    files = input_files(
        root,
        [
            config,
            settings.catalog_path,
            settings.tasks_file,
            settings.requirements,
            settings.lock_file,
        ],
    )
    for task in settings.tasks:
        if task.generated_patch is not None:
            files[str(task.generated_patch.relative_to(root))] = file_hash(task.generated_patch)
    with Flow(root, "diagnose", {"fingerprint": settings.fingerprint()}, files, resume) as flow:
        flow.step("task_environment", lambda: task_environment(settings))
        check_prepared(settings)
        report = flow.step("diagnostic", lambda: diagnose(settings) / "report.json")
        data = json.loads(report.read_text())
        if data["status"] != "completed" or any(not case["matched"] for case in data["cases"]):
            raise ValueError("Diagnostic evidence did not match the expected outcomes")
        flow.report["completion"]["cpu_verification"] = "completed"
        return flow.finish(
            "patch_diagnosed", [{"action": "inspect_diagnostic", "artifact": str(report)}]
        )


def report_results(
    root: Path,
    source: Path,
    review: Path | None,
    record_output: Path | None,
    tracking_config: Path | None,
) -> dict[str, Any]:
    if bool(review) != bool(record_output) or (
        tracking_config is not None and record_output is None
    ):
        raise ValueError(
            "Review and record output must be supplied together; tracking needs a record"
        )
    files = input_files(root, [source] + ([review] if review is not None else []))
    with Flow(root, "report", {"source": str(source)}, files, None) as flow:
        data = json.loads(source.read_text())

        def calculate() -> Path:
            if data.get("kind") == "experiment-session":
                result = summarize(source, max_bytes=artifact_limit(root))
                directory = flow.path.parent / "analysis"
                directory.mkdir()
                path = directory / "report.json"
                write_json(path, result)
                return path
            return (
                analyze(
                    load_run(source.parent, require_evaluation=True),
                    root / "artifacts/analysis",
                    artifact_limit(root),
                )
                / "report.json"
            )

        report = flow.step("analysis", calculate)
        if data.get("kind") == "experiment-session":
            summary = json.loads(report.read_text())
            complete = summary["status"] == "completed" and not (
                summary["counts"]["incomplete"] or summary["counts"]["not_run"]
            )
            flow.report["completion"].update(
                gpu_evaluation="completed" if complete else "incomplete",
                comparison="completed" if complete and not summary["cell_errors"] else "incomplete",
            )
        else:
            run = load_run(source.parent, require_evaluation=True)
            flow.report["completion"]["gpu_evaluation"] = (
                "completed" if run.complete else "incomplete"
            )
        if review is not None and record_output is not None:
            flow.step(
                "public_record",
                lambda: export(root, source, review, record_output, artifact_limit(root)),
            )
            if tracking_config is not None:
                flow.step("tracking", lambda: sync(root, tracking_config, [record_output]))
        return flow.finish(
            "results_analyzed",
            [
                {
                    "action": "review",
                    "artifact": str(report),
                    "requires": "explicit_interpretation_and_decision",
                }
            ],
        )


def prepare_candidate(
    root: Path,
    cpu: Path,
    gpu: Path,
    variant: str,
    source_path: Path,
    policy: Path,
    api_python: Path,
) -> dict[str, Any]:
    settings = SubmissionSettings.load(root, policy)
    source = json.loads(source_path.read_text())
    with Flow(
        root,
        "candidate",
        {"variant": variant, "source": source},
        input_files(root, [cpu, gpu, source_path, policy]),
        None,
    ) as flow:

        def qualify() -> Path:
            candidate = qualify_experiment(cpu, gpu, variant, source, settings.max_archive_bytes)
            directory = flow.path.parent / "candidate"
            directory.mkdir()
            archive = directory / "submission.zip"
            gateway = KaggleGateway(root, settings, api_python)
            gateway.call(
                "kernel",
                source=source,
                destination=str(archive),
                max_bytes=settings.max_archive_bytes,
            )
            if file_hash(archive) != candidate["archive_sha256"]:
                raise ValueError("Saved Notebook archive differs from the verified candidate")
            path = directory / "candidate.json"
            write_json(path, candidate)
            return path

        candidate = flow.step("candidate_evidence", qualify)
        flow.report["completion"].update(
            cpu_verification="completed", gpu_evaluation="completed", comparison="completed"
        )
        return flow.finish(
            "candidate_prepared",
            [
                {
                    "action": "explicit_registration",
                    "artifact": str(candidate),
                    "requires": "candidate_review",
                }
            ],
        )
