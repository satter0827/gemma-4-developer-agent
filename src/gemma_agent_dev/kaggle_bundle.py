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


def prepare(
    root: Path,
    destination: Path,
    *,
    verify_only: bool = False,
    eval_config: Path | None = None,
) -> str:
    root = root.resolve()
    selected = root / (eval_config if eval_config is not None else Path("agent/eval_config.yaml"))
    if selected.is_symlink() or not selected.resolve().is_relative_to(root):
        raise ValueError("Evaluation configuration must be a regular file inside the project")
    selected = selected.resolve()
    settings = EvaluationSettings.load(selected)
    selected_content = selected.read_bytes()
    config = json.loads((root / "configs/runtime.json").read_text())
    kernel_id = config["verification_kernel_id"] if verify_only else config["kernel_id"]
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
                    "results.py",
                )
            ],
            *root.glob("agent/**/*.yaml"),
            *root.glob("agent/**/*.md"),
            root / "configs/runtime.json",
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
            archive.writestr(
                info,
                selected_content
                if info.filename == "agent/eval_config.yaml"
                else path.read_bytes(),
            )
    payload = content.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    provenance = {
        "source_sha256": digest,
        "git_commit": commit,
        "evaluation_config": {
            "path": selected.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(selected_content).hexdigest(),
            "values": settings.as_dict(),
        },
    }
    script = (root / "kaggle/entry.py").read_text() + (
        "\nif __name__ == '__main__':\n"
        f"    main({base64.b64encode(payload).decode()!r}, {provenance!r}, "
        f"verify_only={verify_only!r})\n"
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
            "dataset_sources": [config["wheelhouse_source"]],
            "competition_sources": [config["competition"]],
            "kernel_sources": [],
            "model_sources": [] if verify_only else [config["model_source"]],
            "docker_image": config["docker_image"],
            **({} if verify_only else {"machine_shape": config["machine_shape"]}),
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
