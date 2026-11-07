"""Import results from one fixed private Notebook version without remote writes."""

import json
import tempfile
from pathlib import Path

from gemma_agent_dev.candidates.preflight import artifact_limit
from gemma_agent_dev.common.artifacts import file_hash, new_run_id, write_json
from gemma_agent_dev.evaluation.experiment_results import summarize
from gemma_agent_dev.evaluation.results import load_run
from gemma_agent_dev.submission.api import KaggleGateway
from gemma_agent_dev.submission.config import SubmissionSettings
from gemma_agent_dev.workflows.records import checked_path

NOTEBOOK_NAME_PARTS = 2


def fetch(
    root: Path, kernel: str, version: int, remote_path: str, policy: Path, api_python: Path
) -> Path:
    parts = kernel.split("/")
    relative = Path(remote_path)
    if (
        len(parts) != NOTEBOOK_NAME_PARTS
        or not all(parts)
        or type(version) is not int
        or version <= 0
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in remote_path
        or relative.name not in {"session.json", "manifest.json"}
        or str(relative) != remote_path
        or str(relative.parent) == "."
    ):
        raise ValueError("Specify a fixed Notebook and its result manifest path")
    settings = SubmissionSettings.load(root, policy)
    gateway = KaggleGateway(root, settings, api_python)
    source = {"owner": parts[0], "kernel": parts[1], "version": version}
    files = gateway.call(
        "files", source=source, max_pages=settings.max_pages, page_size=settings.page_size
    )
    selected = [item for item in files if item["path"].startswith(str(relative.parent) + "/")]
    if not selected or remote_path not in {item["path"] for item in selected}:
        raise ValueError("Requested result manifest is absent from this Notebook version")
    selected.sort(key=lambda item: item["path"] != remote_path)
    directory = root / "artifacts/imports"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary)
        seen: set[str] = set()
        size = 0
        for item in selected:
            name = item["path"]
            target = checked_path(staging, name)
            if name in seen or "\\" in name or ".." in Path(name).parts or str(Path(name)) != name:
                raise ValueError("Duplicate or invalid Notebook output path")
            seen.add(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            receipt = gateway.call(
                "kernel",
                source={**source, "path": name},
                destination=str(target),
                max_bytes=settings.max_archive_bytes - size,
            )
            size += receipt["bytes"]
            if size > settings.max_archive_bytes:
                raise ValueError("Notebook output exceeds the transfer limit")
            if name == remote_path:
                manifest_data = json.loads(target.read_text())
                if manifest_data.get("worker_completion_required"):
                    job = manifest_data.get("job_id")
                    if (
                        not isinstance(job, str)
                        or Path(job).name != job
                        or job in {"", ".", ".."}
                        or "\\" in job
                    ):
                        raise ValueError("Invalid remote worker identity")
                    prefix = f"artifacts/jobs/{job}/"
                    selected.extend(
                        item
                        for item in files
                        if item["path"] in {prefix + "watchdog.json", prefix + "processes.jsonl"}
                    )
        manifest = staging / relative
        data = json.loads(manifest.read_text())
        if data.get("kind") == "experiment-session":
            summarize(manifest, max_bytes=artifact_limit(root))
        else:
            load_run(manifest.parent, require_evaluation=True)
        write_json(
            staging / "import.json",
            {"source": source, "manifest": remote_path, "manifest_sha256": file_hash(manifest)},
        )
        destination = directory / new_run_id()
        staging.rename(destination)
    return destination / relative
