"""Project validated Git metadata into a local SQLite store without a server."""

import json
import sys
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import write_json


def deny_network(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect":
        raise RuntimeError("Network access is forbidden during MLflow synchronization")


def synchronize(request: dict[str, Any]) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    database = Path(request["database"])
    database.parent.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str(database.resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    existing = client.get_experiment_by_name(request["experiment_name"])
    experiment_id = (
        existing.experiment_id if existing else client.create_experiment(request["experiment_name"])
    )
    by_id = {}
    token = None
    while True:
        runs = client.search_runs([experiment_id], page_token=token)
        for run in runs:
            key = run.data.tags.get("record_id")
            if key:
                if key in by_id:
                    raise ValueError("Duplicate MLflow record IDs")
                by_id[key] = run
        token = runs.token
        if not token:
            break
    definitions = {
        run.data.tags["experiment_id"]: run.data.tags.get("definition_sha256")
        for run in by_id.values()
    }
    completed, skipped = [], []
    for record in request["records"]:
        prior = by_id.get(record["record_id"])
        previous_definition = definitions.get(record["experiment_id"])
        if previous_definition is not None and previous_definition != record["definition_sha256"]:
            raise ValueError("Stable experiment ID cannot be reused with different conditions")
        definitions[record["experiment_id"]] = record["definition_sha256"]
        if prior and prior.data.tags.get("content_sha256") != record["content_sha256"]:
            raise ValueError("Stable record ID cannot be reused with different content")
        if prior and prior.info.status == "FINISHED":
            skipped.append(record["record_id"])
            continue
        for prompt in record["prompts"]:
            name = prompt["prompt_id"]
            current = mlflow.genai.load_prompt(
                name, version=1, allow_missing=True, link_to_model=False, cache_ttl_seconds=0
            )
            if current is not None and current.tags.get("source_sha256") != prompt["sha256"]:
                raise ValueError("Stable prompt ID cannot be reused with different content")
            if current is None:
                mlflow.genai.register_prompt(
                    name=name, template=prompt["template"], tags={"source_sha256": prompt["sha256"]}
                )
        tags = {
            k: record[k]
            for k in (
                "record_id",
                "experiment_id",
                "hypothesis",
                "definition_sha256",
                "status",
                "content_sha256",
                "interpretation",
            )
        }
        run = prior or client.create_run(experiment_id, tags=tags)
        try:
            for key, value in tags.items():
                client.set_tag(run.info.run_id, key, value)
            for key in ("task_ids", "definition_sha256", "source_sha256", "candidate_hashes"):
                client.log_param(run.info.run_id, key, json.dumps(record[key], sort_keys=True))
            for key, value in record["metrics"].items():
                client.log_metric(run.info.run_id, key, value)
            client.set_terminated(run.info.run_id, "FINISHED")
        except BaseException as error:
            try:
                client.set_terminated(run.info.run_id, "FAILED")
            except BaseException as storage_error:
                error.add_note(f"Cannot mark projection failed: {storage_error}")
            raise
        completed.append(record["record_id"])
    return {
        "status": "completed",
        "created": completed,
        "skipped": skipped,
        "backend": "sqlite",
        "artifacts_uploaded": False,
    }


def main() -> None:
    sys.addaudithook(deny_network)
    output = Path(sys.argv[2])
    try:
        write_json(output, synchronize(json.loads(Path(sys.argv[1]).read_text())))
    except BaseException as error:
        try:
            write_json(output, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        except OSError as storage_error:
            error.add_note(f"Cannot save synchronization failure: {storage_error}")
        raise


if __name__ == "__main__":
    main()
