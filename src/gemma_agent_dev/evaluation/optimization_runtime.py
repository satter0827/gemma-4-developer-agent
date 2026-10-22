"""Execute bounded prompt search with one server and isolated official evaluation cells."""

import copy
import importlib.metadata
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from gemma_agent_dev.candidates.archive import extract_archive
from gemma_agent_dev.common.artifacts import file_hash, inventory, new_run_id, write_json
from gemma_agent_dev.common.events import bundle_session, context, event, phase
from gemma_agent_dev.common.processes import command
from gemma_agent_dev.evaluation.analysis import (
    SCORABLE_TERMINATIONS,
    behavior,
    implementation_changed,
    task_evidence,
    termination_reason,
)
from gemma_agent_dev.evaluation.control_evidence import copy_native_controls
from gemma_agent_dev.evaluation.optimization_candidates import FrozenSearchCandidate, freeze
from gemma_agent_dev.evaluation.optimization_engine import search
from gemma_agent_dev.evaluation.optimization_limits import SearchBudget, SearchLimits
from gemma_agent_dev.evaluation.results import load_run


def reflect(
    server: Any, runtime: dict[str, Any], budget: SearchBudget, messages: list[dict[str, str]]
) -> str:
    budget.check()
    base = server.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Reflection must use the shared local model server")
    payload = {
        "model": "main_lora",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def request(url: str, body: dict[str, Any]) -> dict[str, Any]:
        budget.check()
        timeout = min(
            budget.limits.reflection_timeout_seconds,
            budget.deadline - time.monotonic() - budget.limits.cleanup_reserve_seconds,
        )
        with urllib.request.urlopen(
            urllib.request.Request(
                url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            ),
            timeout=timeout,
        ) as response:
            return dict(json.load(response))

    tokens = request(
        base.removesuffix("/v1") + "/tokenize", {**payload, "add_generation_prompt": True}
    )["count"]
    if (
        type(tokens) is not int
        or tokens <= 0
        or tokens > budget.limits.max_input_tokens
        or tokens + budget.limits.max_output_tokens > runtime["vllm"]["max_model_len"]
    ):
        raise ValueError("Reflection input exceeds the declared token limits")
    result = request(
        base + "/chat/completions",
        {
            **payload,
            "max_tokens": budget.limits.max_output_tokens,
            "temperature": 0,
            "seed": budget.limits.seed,
        },
    )
    choice = result["choices"][0]
    text = choice["message"]["content"]
    if (
        choice["finish_reason"] != "stop"
        or not isinstance(text, str)
        or not text.strip()
        or result["usage"]["completion_tokens"] > budget.limits.max_output_tokens
    ):
        raise ValueError("Reflection is incomplete or exceeds the output token limit")
    return text


def run(
    root: Path,
    provenance: dict[str, Any],
    *,
    verify_only: bool,
    bindings: dict[str, str] | None = None,
) -> Path:
    spec = json.loads((root / "optimization.json").read_text())
    definition = spec["definition"]
    runtime = copy.deepcopy(definition["runtime"])
    limits = SearchLimits.parse(definition["limits"])
    if bindings and (
        not verify_only or set(bindings) != {"output", "sdk_python", "native_controls"}
    ):
        raise ValueError("Local optimization bindings are restricted to CPU verification")
    if not verify_only and definition["gpu_budget"] is None:
        raise ValueError("Optimization GPU time and quota must be explicitly configured")
    output = Path(bindings["output"] if bindings else runtime["paths"]["output"]) / new_run_id()
    output.mkdir(parents=True)
    with (
        bundle_session(root, output, provenance),
        context(
            experiment_id=definition["optimization_id"],
            session_id=output.name,
            stage="cpu_verification" if verify_only else "agent_evaluation",
        ),
        phase("optimization.session"),
    ):
        return execute(root, output, spec, provenance, limits, verify_only, bindings)


def execute(
    root: Path,
    output: Path,
    spec: dict[str, Any],
    provenance: dict[str, Any],
    limits: SearchLimits,
    verify_only: bool,
    bindings: dict[str, str] | None,
) -> Path:
    definition = spec["definition"]
    runtime = copy.deepcopy(definition["runtime"])
    runtime["vllm"]["extra_args"] = ["--no-enable-prefix-caching"]
    path = output / "search.json"
    report: dict[str, Any] = {
        "kind": "optimization-verification" if verify_only else "optimization-session",
        "status": "running",
        **provenance,
        "definition_sha256": spec["definition_sha256"],
        "definition": definition,
        "task_ids": definition["search_task_ids"] + definition["selection_task_ids"],
        "candidate": spec["candidate"],
        "candidates": {},
        "cells": [],
        "actual_order": [],
        "timings": {},
        "adopted": False,
    }
    write_json(path, report)
    started = time.monotonic()
    budget_seconds = (
        limits.cpu_timeout_seconds
        if verify_only
        else definition["gpu_budget"]["job_timeout_seconds"]
    )
    deadline = min(
        started + budget_seconds,
        float(os.environ.get("GEMMA_JOB_DEADLINE_MONOTONIC", str(started + budget_seconds))),
    )
    budget = SearchBudget(
        limits,
        deadline,
        0 if verify_only else limits.cell_seconds(runtime, definition["evaluation"]),
    )
    server = None
    failure: BaseException | None = None
    try:
        wheel = root / "optimization-wheels" / spec["gepa"]["wheel_name"]
        if file_hash(wheel) != spec["gepa"]["wheel_sha256"]:
            raise ValueError("GEPA wheel changed")
        if not bindings:
            from gemma_agent_dev.notebooks.environment import install_wheelhouse

            install_wheelhouse(
                Path(runtime["paths"]["wheelhouse"]),
                output,
                runtime["vllm"]["startup_timeout"],
                runtime["wheelhouse_hashes"],
            )
            command(
                [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
                cwd=root,
                timeout=limits.cpu_timeout_seconds,
            )
        if importlib.metadata.version("gepa") != "0.1.4":
            raise ValueError("GEPA installation changed")
        seed = output / "seed"
        if file_hash(root / "seed/submission.zip") != spec["candidate"]["sha256"]:
            raise ValueError("Optimization seed archive changed")
        extract_archive(root / "seed/submission.zip", seed, spec["candidate"]["max_unpacked_bytes"])
        if inventory(seed) != spec["candidate"]["files"]:
            raise ValueError("Optimization seed files changed")
        if not bindings:
            import asyncio

            from swegemma.models import load_tasks

            from gemma_agent_dev.evaluation.official import (
                create_server,
                expected_model,
                make_validator,
                start_model,
                verify_controls,
            )
            from gemma_agent_dev.evaluation.session import evaluation_config, prepare_candidates

            if {name: importlib.metadata.version(name) for name in runtime["packages"]} != runtime[
                "packages"
            ]:
                raise ValueError("Official SDK changed")
            # Reuse the snapshot/source checks, with one seed and no paired-change policy.
            checking = {
                **spec,
                "variants": {"seed": spec["candidate"]},
                "definition": {"task_ids": report["task_ids"]},
            }
            prepare_candidates(root, output / "checked", checking, runtime)
            server = create_server(seed, runtime)
            (output / "vllm.log").hardlink_to(server.log_path)
            (output / "controls").mkdir()
            configuration = evaluation_config(
                runtime, server, seed, output / "controls", report["task_ids"]
            )
            validator = make_validator(configuration, expected_model(runtime))
            validator(seed)
            if not asyncio.run(
                verify_controls(
                    configuration,
                    load_tasks(Path(runtime["paths"]["data"]) / "tasks.jsonl"),
                    output / "controls",
                    spec["sources"],
                    root / "sandbox-wheels",
                )
            ):
                raise RuntimeError("Optimization CPU controls failed")
            report["controls"] = inventory(output / "controls")
            if not verify_only:
                budget.check(cells=1)
                if (
                    deadline - time.monotonic()
                    < runtime["vllm"]["startup_timeout"]
                    + limits.cell_seconds(runtime, definition["evaluation"])
                    + limits.cleanup_reserve_seconds
                ):
                    raise TimeoutError("Insufficient time to start the model and evaluate a task")
                report["timings"]["model_startup_seconds"] = start_model(server, runtime)
        else:

            def validator(candidate: Path) -> None:
                command(
                    [
                        bindings["sdk_python"],
                        "-m",
                        "gemma_agent_dev.evaluation.optimization_validation",
                        str(root),
                        str(candidate),
                    ],
                    cwd=root,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(root / "src"),
                        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                        "OTEL_SDK_DISABLED": "true",
                    },
                    timeout=limits.cpu_timeout_seconds,
                )

            copy_native_controls(
                Path(bindings["native_controls"]),
                spec["native_controls"]["sha256"],
                report["task_ids"],
                output / "controls",
            )
            report["controls"] = inventory(output / "controls")

        candidates: dict[str, FrozenSearchCandidate] = {}

        def freeze_instruction(text: str) -> str:
            budget.check()
            item = freeze(seed, output / "candidates", definition["prompt_path"], text, validator)
            candidates[item.candidate_id] = item
            report["candidates"][item.candidate_id] = {"sha256": item.sha256, "files": item.files}
            write_json(path, report)
            return item.candidate_id

        original: str = (seed / definition["prompt_path"]).read_text()

        def evaluate_instruction(key: str, task: str) -> dict[str, Any]:
            if verify_only:
                return {
                    "resolved": (candidates[key].directory / definition["prompt_path"]).read_text()
                    != original,
                    "termination": "completed",
                    "implementation_changed": None,
                    "tool_argument_errors": None,
                }
            from gemma_agent_dev.evaluation.session import cell_reference, run_cell

            item = candidates[key]
            cell = {"task_id": task, "variant": key}
            candidate = {"sha256": item.sha256, "files": item.files}
            cell_spec = {
                **spec,
                "variants": {key: candidate},
                "definition": {"experiment_id": definition["optimization_id"]},
            }
            folder = output / "cells" / f"{len(report['cells']):03d}-{task}-{key}"
            report["actual_order"].append(cell)
            write_json(path, report)
            try:
                record = run_cell(
                    root,
                    folder,
                    runtime,
                    cell_spec,
                    server,
                    item.directory,
                    cell,
                    provenance,
                    report["controls"],
                )
            except BaseException as error:
                try:
                    if (folder / "manifest.json").is_file():
                        report["cells"].append(cell_reference(folder, cell))
                        write_json(path, report)
                except OSError:
                    error.add_note("Cannot save failed optimization cell reference")
                raise
            report["cells"].append(record)
            write_json(path, report)
            saved = load_run(folder)
            row = saved.rows[task]
            evidence = task_evidence(saved, task, spec["max_artifact_bytes"])
            termination = termination_reason(saved, task, evidence)
            if not saved.complete or termination not in SCORABLE_TERMINATIONS:
                raise RuntimeError("Official evaluation failed or did not complete")
            return {
                **behavior(saved, task, spec["max_artifact_bytes"]),
                "resolved": row["resolved"],
                "termination": termination,
                "implementation_changed": implementation_changed(
                    saved, task, spec["max_artifact_bytes"]
                ),
                "tool_argument_errors": len(evidence["tool_argument_errors"])
                if evidence["trace_error"] is None and not evidence["ambiguous_tool_observations"]
                else None,
            }

        def reflection(messages: list[dict[str, str]]) -> str:
            if verify_only:
                return (
                    "```\n" + original + "\nVerify that focused tests collect cases "
                    "and implementation files changed.\n```"
                )
            return reflect(server, runtime, budget, messages)

        report.update(
            search(
                original,
                definition["search_task_ids"],
                definition["selection_task_ids"],
                budget,
                freeze_instruction,
                evaluate_instruction,
                reflection,
            )
        )
        event(
            "optimization.outcome",
            result="improved" if report["improved"] else "unchanged",
            metric_calls=budget.metric_calls,
            reflection_calls=budget.reflection_calls,
        )
        if verify_only:
            report["scope"] = (
                "actual GEPA and pinned SDK compilation; synthetic outcomes; no accuracy claim"
            )
            if not report["improved"] or report["reflection_calls"] != limits.max_reflection_calls:
                raise ValueError(
                    "Synthetic GEPA verification did not exercise a generated candidate"
                )
    except BaseException as error:
        failure = error
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        event("optimization.exception", level="ERROR", exception=True)
    finally:
        if server is not None:
            try:
                from gemma_agent_dev.evaluation.official import stop_model

                stop_model(server)
            except BaseException as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                failure = failure or error
        report["metric_calls"] = budget.metric_calls
        report["reflection_calls"] = budget.reflection_calls
        report["timings"]["total_seconds"] = time.monotonic() - started
        if failure is not None:
            report["status"] = "failed"
        try:
            write_json(path, report)
        except OSError as error:
            if failure is None:
                failure = error
            else:
                failure.add_note("Cannot save optimization evidence")
    if failure is not None:
        raise failure
    return path
