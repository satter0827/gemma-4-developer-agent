"""Propose frozen task orders without selecting on model outcomes."""

import hashlib
from pathlib import Path
from typing import Any

import yaml

from gemma_agent_dev.common.config import mapping, positive, project_path
from gemma_agent_dev.evaluation.experiments import Experiment
from gemma_agent_dev.evaluation.session_policy import VARIANTS_PER_PAIR
from gemma_agent_dev.evaluation.tracking import load_records
from gemma_agent_dev.tasks.pools import TaskPool

SECONDS_PER_HOUR = 3600


def plan(root: Path, config: Path) -> dict[str, Any]:
    data = mapping(
        yaml.safe_load(project_path(root, str(config)).read_text()),
        {"experiment", "task_count", "repository_order", "seed", "records"},
        "task rotation",
    )
    positive(data["task_count"], "task_count", integer=True)
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("Expected a nonnegative rotation seed")
    repos = data["repository_order"]
    if (
        not isinstance(repos, list)
        or not repos
        or len(repos) != len(set(map(str, repos)))
        or any(not isinstance(r, str) or not r.isidentifier() for r in repos)
    ):
        raise ValueError("Expected ordered unique repositories")
    experiment = Experiment.load(root, Path(data["experiment"]))
    pool = TaskPool.load(root, experiment.pool)
    record_dir = project_path(root, data["records"])
    if not record_dir.is_dir():
        raise ValueError("Missing public experiment record directory")
    records = load_records(root, sorted(record_dir.glob("*.json")))
    observations = [item for record in records for item in record.get("observations", [])]
    attempted = {r["task_id"] for r in observations if r["status"] != "not_run"}
    completed = {r["task_id"] for r in observations if r["status"] == "completed"}
    reserved = {
        task
        for use in pool.uses
        if use["experiment_id"] != experiment.experiment_id
        for task in use["task_ids"]
    }
    queues = {
        repo: sorted(
            (
                task
                for task in pool.tuning
                if task.rsplit("_", 1)[0] == repo and task not in attempted | reserved
            ),
            key=lambda task: hashlib.sha256(f"{data['seed']}:{task}".encode()).hexdigest(),
        )
        for repo in repos
    }
    if {t.rsplit("_", 1)[0] for t in pool.tuning} != set(repos):
        raise ValueError("Repository rotation must cover the full tuning pool")
    selected: list[str] = []
    while len(selected) < data["task_count"] and any(queues.values()):
        for repo in repos:
            if queues[repo] and len(selected) < data["task_count"]:
                selected.append(queues[repo].pop(0))
    if len(selected) != data["task_count"]:
        raise ValueError("Not enough unused tuning tasks; explicit reuse needs a new plan")
    definition = experiment.definition(root)
    runtime, evaluation = definition["runtime"], definition["evaluation"]
    pair_seconds = (
        experiment.session.pair_seconds(evaluation, runtime["verification_timeout_seconds"])
        - experiment.session.cleanup_reserve_seconds
    )
    upper = (
        runtime["vllm"]["startup_timeout"]
        + len(selected) * pair_seconds
        + experiment.session.cleanup_reserve_seconds
    )
    estimates = []
    for record in records:
        metrics = record["metrics"]
        if {"model_startup_seconds", "evaluation_seconds"} <= set(metrics):
            seconds = (
                metrics["model_startup_seconds"]
                + metrics["evaluation_seconds"]
                / len(record["task_ids"])
                * len(selected)
                * VARIANTS_PER_PAIR
                + experiment.session.cleanup_reserve_seconds
            )
            estimates.append(
                {
                    "record_id": record["record_id"],
                    "wall_seconds": seconds,
                    "quota_hours": seconds
                    / SECONDS_PER_HOUR
                    * experiment.session.quota_per_wall_hour,
                }
            )
    return {
        "experiment_id": experiment.experiment_id,
        "task_ids": selected,
        "matches_declared_selection": selected == experiment.task_ids,
        "attempted": sorted(attempted & set(pool.tuning)),
        "completed": sorted(completed & set(pool.tuning)),
        "reserved_elsewhere": sorted(reserved & set(pool.tuning)),
        "confirmation_tasks_used": [],
        "cost_kind": "estimate",
        "startup_and_evaluation_upper_seconds": upper,
        "cpu_setup_seconds": None,
        "fits_without_cpu_setup_at_upper_bound": upper <= experiment.session.job_timeout_seconds,
        "quota_upper_without_cpu_setup_hours": upper
        / SECONDS_PER_HOUR
        * experiment.session.quota_per_wall_hour,
        "historical_extrapolations": estimates,
    }
