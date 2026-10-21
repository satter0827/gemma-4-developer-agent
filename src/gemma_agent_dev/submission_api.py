"""Isolate Kaggle credentials and SDK calls behind a bounded, explicit worker."""

import contextlib
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_agent_dev.submission_config import SubmissionSettings, timestamp

KAGGLE_VERSION = "2.2.4"
READ_OPERATIONS = frozenset({"history", "limits", "kernel", "download"})
WRITE_OPERATIONS = frozenset({"upload", "create"})
DOWNLOAD_CHUNK_BYTES = 64 * 1024


class KaggleGateway:
    def __init__(self, root: Path, settings: SubmissionSettings, python: Path):
        self.root, self.settings, self.python = root, settings, python

    def call(self, operation: str, *, execute: bool = False, **payload: Any) -> Any:
        if operation not in READ_OPERATIONS | WRITE_OPERATIONS:
            raise ValueError("Unknown Kaggle operation")
        if operation in WRITE_OPERATIONS and not execute:
            raise ValueError("Kaggle writes require explicit execution")
        transfer = operation in {"kernel", "download", "upload"}
        timeout = (
            self.settings.transfer_timeout_seconds
            if transfer
            else self.settings.api_timeout_seconds
        )
        environment = {**os.environ, "PYTHONPATH": str(self.root / "src")}
        try:
            result = subprocess.run(
                [str(self.python), "-m", "gemma_agent_dev.submission_api"],
                input=json.dumps(
                    {
                        "operation": operation,
                        "execute": execute,
                        "competition": self.settings.competition,
                        **payload,
                    }
                ),
                text=True,
                capture_output=True,
                env=environment,
                cwd=self.root,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Kaggle {operation} timed out") from error
        if result.returncode:
            # SDK errors can include signed URLs or credentials. Never forward their text.
            raise RuntimeError(f"Kaggle {operation} failed")
        return json.loads(result.stdout)


def utc_date(value: Any) -> str:
    if not isinstance(value, datetime):
        raise ValueError("Missing submission timestamp")
    return value.replace(tzinfo=value.tzinfo or UTC).astimezone(UTC).isoformat()


def save_download(response: Any, destination: Path, max_bytes: int) -> dict[str, int]:
    # In the pinned SDK, HttpRedirect.prepare_from returns the requests.Response.
    # Its HTTP session already follows redirects and leaves the body streamed.
    with contextlib.closing(response):
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ValueError("Downloaded submission exceeds the size limit")
        with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
            partial = Path(directory) / "submission.zip"
            size = 0
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Downloaded submission exceeds the size limit")
                    output.write(chunk)
            partial.replace(destination)
    return {"bytes": size}


def dispatch(api: Any, client: Any, data: dict[str, Any], types: Any, kernel_types: Any) -> Any:
    operation, competition = data["operation"], data["competition"]
    service = client.competitions.competition_api_client
    if operation == "history":
        rows, tokens = [], set()
        token = ""
        for _ in range(data["max_pages"]):
            request = types.ApiListSubmissionsRequest()
            request.competition_name, request.page_size = competition, data["page_size"]
            request.page_token = token
            response = service.list_submissions(request)
            for row in response.submissions:
                rows.append(
                    {
                        "id": row.ref,
                        "date": utc_date(row.date),
                        "status": row.status.name,
                        "score": row.public_score or None,
                        "description": row.description,
                    }
                )
            token = response.next_page_token
            if not token:
                return rows
            if token in tokens:
                raise ValueError("Repeated history cursor")
            tokens.add(token)
        raise ValueError("Submission history exceeds pagination limit")
    if operation == "limits":
        request = types.ApiGetSubmissionLimitsRequest()
        request.competition_name = competition
        response = service.get_submission_limits(request)
        # Read typed fields: to_dict omits legitimate zero values.
        return {"remaining": response.num_allowed_now, "today": response.num_today}
    if operation in {"kernel", "download"}:
        if operation == "kernel":
            request = kernel_types.ApiDownloadKernelOutputRequest()
            source = data["source"]
            request.owner_slug, request.kernel_slug = source["owner"], source["kernel"]
            request.version_number, request.file_path = source["version"], source["path"]
            response = client.kernels.kernels_api_client.download_kernel_output(request)
        else:
            request = types.ApiDownloadSubmissionRequest()
            request.submission_id = data["id"]
            response = service.download_submission(request)
        return save_download(response, Path(data["destination"]), data["max_bytes"])
    if operation not in WRITE_OPERATIONS or data.get("execute") is not True:
        raise ValueError("Write operation denied")
    if operation == "upload":
        path = Path(data["path"])
        request = types.ApiStartSubmissionUploadRequest()
        request.competition_name, request.file_name = competition, "submission.zip"
        request.content_length = path.stat().st_size
        request.last_modified_epoch_seconds = int(path.stat().st_mtime)
        response = service.start_submission_upload(request)
        status = api.upload_complete(str(path), response.create_url, quiet=True)
        if (
            status
            != importlib.import_module(
                "kaggle.api.kaggle_api_extended"
            ).ResumableUploadResult.COMPLETE
        ):
            raise ValueError("Upload did not complete")
        return {"token": response.token}
    request = types.ApiCreateSubmissionRequest()
    request.competition_name, request.blob_file_tokens = competition, data["token"]
    request.submission_description = data["description"]
    # Check inside the authenticated worker, immediately before the final SDK call.
    now = datetime.now(UTC)
    if now >= timestamp(data["before"]) or now.date().isoformat() != data["utc_day"]:
        return {"expired": True}
    response = service.create_submission(request)
    if type(response.ref) is not int or response.ref <= 0:
        raise ValueError("Submission receipt is unknown")
    return {"id": response.ref}


def worker(data: dict[str, Any]) -> Any:
    operation = data["operation"]
    if operation not in READ_OPERATIONS | WRITE_OPERATIONS or (
        operation in WRITE_OPERATIONS and data.get("execute") is not True
    ):
        raise ValueError("Operation denied before SDK initialization")
    if importlib.metadata.version("kaggle") != KAGGLE_VERSION:
        raise ValueError("Kaggle SDK version changed")
    api = importlib.import_module("kaggle.api.kaggle_api_extended").KaggleApi()
    api.authenticate()
    types = importlib.import_module("kagglesdk.competitions.types.competition_api_service")
    kernel_types = importlib.import_module("kagglesdk.kernels.types.kernels_api_service")
    with api.build_kaggle_client() as client:
        return dispatch(api, client, data, types, kernel_types)


def main() -> int:
    try:
        data = json.load(sys.stdin)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = worker(data)
        print(json.dumps(result))
        return 0
    except Exception:
        print("Kaggle operation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
