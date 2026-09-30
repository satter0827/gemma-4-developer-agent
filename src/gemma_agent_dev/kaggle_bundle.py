"""Build a private, self-contained Kaggle script from selected project sources."""

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from gemma_agent_dev.artifacts import write_json
from gemma_agent_dev.config import EvaluationSettings


def prepare(root: Path, destination: Path) -> str:
    EvaluationSettings.load(root / "agent/eval_config.yaml")
    config = json.loads((root / "configs/kaggle.json").read_text())
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(
        [
            *[
                root / "src/gemma_agent_dev" / name
                for name in (
                    "__init__.py",
                    "runtime.py",
                    "config.py",
                    "artifacts.py",
                    "official.py",
                    "packaging.py",
                )
            ],
            *root.glob("agent/**/*.yaml"),
            *root.glob("agent/**/*.md"),
            root / "configs/kaggle.json",
            root / "kaggle/entry.py",
        ]
    )
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Symlink in source bundle: {path}")
            info = zipfile.ZipInfo(path.relative_to(root).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = content.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    provenance = {"source_sha256": digest, "git_commit": commit}
    script = (root / "kaggle/entry.py").read_text() + (
        "\nif __name__ == '__main__':\n"
        f"    main({base64.b64encode(payload).decode()!r}, {provenance!r})\n"
    )
    destination.mkdir(parents=True)
    (destination / "run.py").write_text(script, encoding="utf-8")
    write_json(
        destination / "kernel-metadata.json",
        {
            "id": config["kernel_id"],
            "title": "Gemma 4 Developer Agent Baseline",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [config["wheelhouse_source"]],
            "competition_sources": [config["competition"]],
            "kernel_sources": [],
            "model_sources": [config["model_source"]],
            "docker_image": config["docker_image"],
            "machine_shape": config["machine_shape"],
        },
    )
    write_json(
        destination / "source-manifest.json",
        {
            **provenance,
            "files": [p.relative_to(root).as_posix() for p in files],
        },
    )
    return digest
