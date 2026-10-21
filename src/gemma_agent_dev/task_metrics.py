"""Run the verified official structural estimator offline in its CPU environment."""

import importlib.metadata
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from gemma_agent_dev.artifacts import file_hash

DENIED_EVENTS = frozenset(
    {"socket.connect", "subprocess.Popen", "os.system", "os.fork", "os.posix_spawn"}
)
GPU_MODULES = frozenset({"torch", "torchvision", "vllm"})


def audit(event: str, args: tuple[Any, ...]) -> None:
    if event in DENIED_EVENTS or (event == "import" and str(args[0]).split(".")[0] in GPU_MODULES):
        raise RuntimeError("External action denied during structural analysis")


def calculate(tasks: Path, wheel: Path, expected: str, version: str) -> list[dict[str, Any]]:
    if file_hash(wheel) != expected or importlib.metadata.version("swegemma") != version:
        raise ValueError("Official difficulty SDK changed")
    distribution = importlib.metadata.distribution("swegemma")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("swegemma/") and name.endswith(".py"):
                if Path(str(distribution.locate_file(name))).read_bytes() != archive.read(name):
                    raise ValueError("Installed official SDK differs from the verified wheel")
    os.environ.update(
        {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    )
    sys.addaudithook(audit)
    from swegemma.difficulty import analyze_task

    fields = (
        "instance_id",
        "repo",
        "num_files",
        "total_churn",
        "test_files",
        "composite_score",
        "difficulty_tier",
    )
    results = []
    for line in tasks.read_text().splitlines():
        result = analyze_task(json.loads(line)).to_dict()
        results.append({name: result[name] for name in fields})
    if GPU_MODULES & sys.modules.keys():
        raise ValueError("Structural analysis imported GPU libraries")
    return results


def main() -> None:
    tasks, wheel, digest, version, output = sys.argv[1:]
    Path(output).write_text(json.dumps(calculate(Path(tasks), Path(wheel), digest, version)))


if __name__ == "__main__":
    main()
