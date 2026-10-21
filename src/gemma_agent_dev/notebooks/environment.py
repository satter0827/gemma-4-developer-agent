"""Install the pinned official wheelhouse inside Kaggle only."""

import glob
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.common.events import logged


@logged("dependencies.install")
def install_wheelhouse(
    directory: Path, output: Path, startup_timeout: int, expected: dict[str, str]
) -> None:
    environment = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "TRANSFORMERS_NO_TF": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": str(startup_timeout),
        "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    wheels = sorted(p for p in directory.glob("*.whl") if "cutlass" not in p.name.lower())
    if not wheels:
        raise FileNotFoundError(f"No official wheels in {directory}")
    actual = {p.name: file_hash(p) for p in wheels}
    if actual != expected:
        raise ValueError("Official wheelhouse checksum mismatch")
    write_json(output / "wheelhouse.json", actual)
    os.environ.update(environment)
    for pattern in (
        "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
        "/usr/local/lib/python*/site-packages/*cutlass*.pth",
    ):
        for name in glob.glob(pattern):
            Path(name).unlink(missing_ok=True)
    staging = output / "wheel-links"
    staging.mkdir()
    for wheel in wheels:
        name = wheel.name.replace("cu128", "+cu128") if "+" not in wheel.name else wheel.name
        (staging / name).symlink_to(wheel)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                *map(str, sorted(staging.glob("*.whl"))),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(staging)
    importlib.invalidate_caches()
