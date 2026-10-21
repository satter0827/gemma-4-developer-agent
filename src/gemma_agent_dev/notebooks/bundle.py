"""Serialize a closed source bundle and its private Kaggle launcher deterministically."""

import base64
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.archives import zip_content
from gemma_agent_dev.common.artifacts import write_json
from gemma_agent_dev.common.events import policy
from gemma_agent_dev.common.sources import source_paths, validate_bundle_sources


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
) -> str:
    digest = hashlib.sha256(content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entry = archive.read("src/gemma_agent_dev/notebooks/entry.py").decode()
    script = entry + (
        "\nif __name__ == '__main__':\n"
        f"    main({base64.b64encode(content).decode()!r}, {provenance!r}, "
        f"diagnostic={diagnostic!r}, verify_only={verify_only!r}, "
        f"timeout_seconds={runtime['job_timeout_seconds']!r}, "
        f"progress_interval_seconds={runtime['progress_interval_seconds']!r})\n"
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
            "dataset_sources": [runtime["wheelhouse_source"]],
            "competition_sources": [runtime["competition"]],
            "kernel_sources": [],
            "model_sources": [] if verify_only else [runtime["model_source"]],
            "docker_image": runtime["docker_image"],
            **({} if verify_only else {"machine_shape": runtime["machine_shape"]}),
        },
    )
    return digest
