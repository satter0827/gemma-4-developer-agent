"""Serialize a closed source bundle and its private Kaggle launcher deterministically."""

import base64
import hashlib
import io
import lzma
import re
import zipfile
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.archives import zip_content
from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.common.events import policy
from gemma_agent_dev.common.sources import source_paths, validate_bundle_sources

# Kaggle SaveKernel requires the UTF-8 source to be strictly below one megabyte.
KAGGLE_SOURCE_MAX_BYTES = 1_000_000


def payload(
    root: Path, modules: tuple[str, ...], inputs: dict[str, bytes]
) -> tuple[bytes, list[Path]]:
    files = source_paths(root, modules)
    validate_bundle_sources(root, files)
    content = {
        **inputs,
        "logging.json": policy(root),
        **{p.relative_to(root).as_posix(): p.read_bytes() for p in files},
    }
    return zip_content(content), files


def write_notebook(
    destination: Path,
    content: bytes,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    kernel_id: str,
    *,
    verify_only: bool,
    diagnostic: bool,
    bundle_dataset: str | None = None,
) -> str:
    digest = hashlib.sha256(content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entry = archive.read("src/gemma_agent_dev/notebooks/entry.py").decode()
    if bundle_dataset is not None and not re.fullmatch(
        r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", bundle_dataset
    ):
        raise ValueError("Expected an owner/dataset source")
    encoded = None if bundle_dataset else base64.b64encode(lzma.compress(content)).decode()
    source_path = (
        f"/kaggle/input/datasets/{bundle_dataset}/source-bundle.bin" if bundle_dataset else None
    )
    script = entry + (
        "\nif __name__ == '__main__':\n"
        f"    main({encoded!r}, {provenance!r}, payload_path={source_path!r}, "
        f"diagnostic={diagnostic!r}, verify_only={verify_only!r}, "
        f"timeout_seconds={runtime['job_timeout_seconds']!r}, "
        f"progress_interval_seconds={runtime['progress_interval_seconds']!r})\n"
    )
    if len(script.encode("utf-8")) >= KAGGLE_SOURCE_MAX_BYTES:
        raise ValueError(
            "Kaggle source exceeds the upload limit; use --bundle-dataset for private assets"
        )
    destination.mkdir(parents=True)
    (destination / "run.py").write_text(script, encoding="utf-8")
    write_json(
        destination / "kernel-metadata.json",
        {
            "id": kernel_id,
            "title": kernel_id.split("/")[-1].replace("-", " ").title(),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": not verify_only,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": list(
                dict.fromkeys(
                    [runtime["wheelhouse_source"], *([bundle_dataset] if bundle_dataset else [])]
                )
            ),
            "competition_sources": [runtime["competition"]],
            "kernel_sources": [],
            "model_sources": [] if verify_only else [runtime["model_source"]],
            "docker_image": runtime["docker_image"],
            **({} if verify_only else {"machine_shape": runtime["machine_shape"]}),
        },
    )
    return digest
