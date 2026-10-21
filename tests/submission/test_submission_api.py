"""Verify SDK operation boundaries without uploading or creating a real submission."""

import importlib.metadata
import io
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from gemma_agent_dev.submission import api
from gemma_agent_dev.submission.config import SubmissionSettings

ROOT = Path(__file__).resolve().parents[2]


def types() -> Any:
    return NS(
        **{
            name: NS
            for name in (
                "ApiListSubmissionsRequest",
                "ApiGetSubmissionLimitsRequest",
                "ApiDownloadSubmissionRequest",
                "ApiStartSubmissionUploadRequest",
                "ApiCreateSubmissionRequest",
                "ApiDownloadKernelOutputRequest",
            )
        }
    )


def test_read_operations(tmp_path: Path) -> None:
    calls: list[Any] = []

    def history(request: Any) -> Any:
        calls.append(request)
        return NS(
            submissions=[
                NS(
                    ref=len(calls),
                    date=datetime(2026, 10, 21),
                    status=NS(name="COMPLETE"),
                    public_score="0.01",
                    description="manual",
                )
            ],
            next_page_token="page2" if len(calls) == 1 else "",
        )

    content = NS(
        headers={"Content-Length": "9"},
        iter_content=lambda **_: iter((b"syn", b"thetic")),
        raise_for_status=lambda: None,
        close=lambda: None,
    )
    service = NS(
        list_submissions=history,
        get_submission_limits=lambda _: NS(num_allowed_now=0, num_today=1),
        download_submission=lambda _: content,
    )
    client = NS(
        competitions=NS(competition_api_client=service),
        kernels=NS(kernels_api_client=NS(download_kernel_output=lambda _: content)),
    )
    data = {"operation": "history", "competition": "competition", "page_size": 1, "max_pages": 3}
    rows = api.dispatch(None, client, data, types(), types())
    assert [r["id"] for r in rows] == [1, 2] and calls[1].page_token == "page2"
    assert api.dispatch(None, client, {**data, "operation": "limits"}, types(), types()) == {
        "remaining": 0,
        "today": 1,
    }
    for op in ("kernel", "download"):
        payload = {
            **data,
            "operation": op,
            "id": 1,
            "destination": str(tmp_path / op),
            "max_bytes": 100,
            "source": {
                "owner": "owner",
                "kernel": "kernel",
                "version": 10,
                "path": "run/submission.zip",
            },
        }
        assert api.dispatch(None, client, payload, types(), types())["bytes"] == 9
        with pytest.raises(ValueError):
            api.dispatch(None, client, {**payload, "max_bytes": 1}, types(), types())
    service.list_submissions = lambda _: NS(submissions=[], next_page_token="same")
    with pytest.raises(ValueError, match="Repeated"):
        api.dispatch(None, client, data, types(), types())
    with pytest.raises(ValueError, match="pagination"):
        api.dispatch(None, client, {**data, "max_pages": 1}, types(), types())
    with pytest.raises(ValueError):
        api.utc_date(None)


@pytest.mark.parametrize("failure", ["header", "stream", "read", "status"])
def test_failed_download_is_closed_and_preserves_destination(tmp_path: Path, failure: str) -> None:
    destination = tmp_path / "submission.zip"
    destination.write_bytes(b"existing")
    consumed, closed = [], []

    def chunks(*, chunk_size: int) -> Any:
        assert chunk_size == api.DOWNLOAD_CHUNK_BYTES
        for chunk in (b"123", b"456", b"unread"):
            consumed.append(chunk)
            yield chunk
            if failure == "read":
                raise OSError("Connection interrupted")

    def check_status() -> None:
        if failure == "status":
            raise OSError("HTTP failure")

    response = NS(
        headers={"Content-Length": "100"} if failure == "header" else {},
        iter_content=chunks,
        raise_for_status=check_status,
        close=lambda: closed.append(True),
    )
    with pytest.raises((ValueError, OSError)):
        api.save_download(response, destination, 5)
    assert closed == [True]
    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]
    expected = [] if failure in {"header", "status"} else [b"123"]
    if failure == "stream":
        expected.append(b"456")
    assert consumed == expected


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "5"}, {"Content-Length": "1"}])
def test_download_counts_streamed_bytes_at_the_limit(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    closed = []
    response = NS(
        headers=headers,
        iter_content=lambda **_: iter((b"12", b"", b"345")),
        raise_for_status=lambda: None,
        close=lambda: closed.append(True),
    )
    destination = tmp_path / "submission.zip"
    assert api.save_download(response, destination, 5) == {"bytes": 5}
    assert destination.read_bytes() == b"12345"
    assert closed == [True] and list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("operation", ["kernel", "download"])
def test_pinned_sdk_follows_redirects_and_returns_streamed_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    sdk = pytest.importorskip("kagglesdk.kaggle_client")
    requests = pytest.importorskip("requests")
    assert importlib.metadata.version("kagglesdk") == "0.1.37"
    competition_types = importlib.import_module(
        "kagglesdk.competitions.types.competition_api_service"
    )
    kernel_types = importlib.import_module("kagglesdk.kernels.types.kernels_api_service")
    methods: list[str] = []
    closed: list[Any] = []
    close_response = requests.Response.close

    def close(response: Any) -> None:
        closed.append(response)
        close_response(response)

    def send(adapter: Any, request: Any, **kwargs: Any) -> Any:
        assert kwargs["stream"] is True
        methods.append(request.method)
        response = requests.Response()
        response.request, response.url = request, request.url
        response.headers["Content-Type"] = "application/zip"
        if len(methods) == 1:
            assert request.method == "POST"
            response.status_code = 302
            response.headers["Location"] = "https://storage.invalid/submission.zip"
            response.raw = io.BytesIO()
        else:
            assert len(methods) == 2 and request.method == "GET"
            response.status_code = 200
            response.headers["Content-Length"] = "9"
            response.raw = io.BytesIO(b"synthetic")
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setattr(requests.Response, "close", close)
    save_download = api.save_download

    def save(response: Any, destination: Path, max_bytes: int) -> dict[str, int]:
        assert isinstance(response, requests.Response)
        assert len(response.history) == 1 and response.history[0].status_code == 302
        assert response.raw.tell() == 0
        return save_download(response, destination, max_bytes)

    monkeypatch.setattr(api, "save_download", save)
    destination = tmp_path / "submission.zip"
    with sdk.KaggleClient(api_token="synthetic") as client:
        result = api.dispatch(
            None,
            client,
            {
                "operation": operation,
                "competition": "competition",
                "id": 1,
                "source": {
                    "owner": "owner",
                    "kernel": "kernel",
                    "version": 10,
                    "path": "run/submission.zip",
                },
                "destination": str(destination),
                "max_bytes": 9,
            },
            competition_types,
            kernel_types,
        )
    assert result == {"bytes": 9} and destination.read_bytes() == b"synthetic"
    assert methods == ["POST", "GET"]
    assert any(response.request.method == "GET" for response in closed)


def test_create_cutoff_and_write_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: list[Any] = []

    def create(request: Any) -> Any:
        created.append(request)
        return NS(ref=123)

    service = NS(
        create_submission=create,
        start_submission_upload=lambda _: NS(token="token", create_url="signed-secret"),
    )
    client = NS(competitions=NS(competition_api_client=service))
    data = {
        "operation": "create",
        "execute": True,
        "competition": "competition",
        "token": "token",
        "description": "candidate",
        "before": (datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
        "utc_day": datetime.now(UTC).date().isoformat(),
    }
    assert api.dispatch(None, client, data, types(), types()) == {"id": 123}
    assert len(created) == 1
    assert api.dispatch(None, client, {**data, "before": "2000-01-01T00:00:00Z"}, types(), types())[
        "expired"
    ]
    assert api.dispatch(None, client, {**data, "utc_day": "2000-01-01"}, types(), types())[
        "expired"
    ]
    assert len(created) == 1
    with pytest.raises(ValueError, match="denied"):
        api.dispatch(None, client, {**data, "execute": False}, types(), types())
    service.create_submission = lambda _: NS(ref=0)
    with pytest.raises(ValueError, match="unknown"):
        api.dispatch(None, client, data, types(), types())
    path = tmp_path / "submission.zip"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(
        "gemma_agent_dev.submission.api.importlib.import_module",
        lambda _: NS(ResumableUploadResult=NS(COMPLETE="complete")),
    )
    uploader = NS(upload_complete=lambda *a, **kw: "complete")
    upload = {**data, "operation": "upload", "path": str(path)}
    assert api.dispatch(uploader, client, upload, types(), types()) == {"token": "token"}
    uploader.upload_complete = lambda *a, **kw: "failed"
    with pytest.raises(ValueError):
        api.dispatch(uploader, client, upload, types(), types())


def test_gateway_does_not_leak_worker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SubmissionSettings.load(ROOT, Path("configs/submission/policy.yaml"))
    gateway = api.KaggleGateway(ROOT, settings, Path("python"))
    with pytest.raises(ValueError):
        gateway.call("upload")
    with pytest.raises(ValueError):
        gateway.call("unexpected")

    def run(*args: Any, **kwargs: Any) -> Any:
        assert json.loads(kwargs["input"])["operation"] == "history"
        return NS(returncode=0, stdout="[]")

    monkeypatch.setattr("gemma_agent_dev.submission.api.subprocess.run", run)
    assert gateway.call("history") == []
    monkeypatch.setattr(
        "gemma_agent_dev.submission.api.subprocess.run",
        lambda *a, **kw: NS(returncode=1, stdout="credential", stderr="secret"),
    )
    with pytest.raises(RuntimeError, match="^Kaggle history failed$"):
        gateway.call("history")

    def timeout(*a: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired("command-with-token", 1)

    monkeypatch.setattr("gemma_agent_dev.submission.api.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="^Kaggle history timed out$"):
        gateway.call("history")
    for operation in ("upload", "create", "unknown"):
        with pytest.raises(ValueError):
            api.worker({"operation": operation})


def test_worker_and_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    from contextlib import nullcontext

    monkeypatch.setattr(
        "gemma_agent_dev.submission.api.importlib.metadata.version", lambda _: "wrong"
    )
    with pytest.raises(ValueError):
        api.worker({"operation": "limits"})
    monkeypatch.setattr(
        "gemma_agent_dev.submission.api.importlib.metadata.version", lambda _: api.KAGGLE_VERSION
    )
    fake = NS(authenticate=lambda: None, build_kaggle_client=lambda: nullcontext("client"))
    monkeypatch.setattr(
        "gemma_agent_dev.submission.api.importlib.import_module",
        lambda _: NS(KaggleApi=lambda: fake),
    )
    monkeypatch.setattr(api, "dispatch", lambda *a: {"remaining": 1})
    assert api.worker({"operation": "limits"}) == {"remaining": 1}
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"operation":"limits"}'))
    assert api.main() == 0 and '"remaining": 1' in capsys.readouterr().out
    monkeypatch.setattr(sys, "stdin", io.StringIO("{"))
    assert api.main() == 1 and "failed" in capsys.readouterr().err
