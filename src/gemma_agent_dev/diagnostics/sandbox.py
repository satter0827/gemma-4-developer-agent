"""Apply and observe source roots only in the official development subprocess sandbox."""

import json
import re
import shlex
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.diagnostics.origins import (
    HOST_DISPLAY_VARIABLES,
    test_environment,
    validate_origins,
)


def observe_sources(
    sandbox: Any, specifications: dict[str, Any], output: Path, dependency_dir: Path | None = None
) -> Callable[[], None]:
    original = sandbox.exec
    failures: list[str] = []
    prepared: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    lock = threading.RLock()

    def execute(sandbox_id: str, command: str, **kwargs: Any) -> Any:
        paths = sandbox.sandboxes[sandbox_id]
        workspace = Path(paths["workspace"])
        guard = Path(paths["root"]) / "source-guard"
        with lock:
            if sandbox_id not in prepared:
                matches = [
                    spec
                    for spec in specifications.values()
                    if all(
                        any(
                            (workspace / root / module.replace(".", "/"))
                            .with_suffix(".py")
                            .is_file()
                            or (
                                workspace / root / module.replace(".", "/") / "__init__.py"
                            ).is_file()
                            for root in spec["source_roots"]
                        )
                        for module in spec["import_modules"]
                    )
                ]
                if matches:
                    spec = matches[0]
                    if any(m != spec for m in matches):
                        raise ValueError("Ambiguous workspace source configuration")
                    wheels = spec.get("dependency_wheels", {})
                    for name, digest in wheels.items():
                        if dependency_dir is None or file_hash(dependency_dir / name) != digest:
                            raise ValueError("Task dependency wheel changed or is missing")
                    if wheels:
                        assert dependency_dir is not None
                        command_parts = [
                            str(Path(paths["venv"]) / "bin/python"),
                            "-m",
                            "pip",
                            "install",
                            "--no-index",
                            "--no-deps",
                            *[str(dependency_dir / name) for name in wheels],
                        ]
                        install = original(sandbox_id, shlex.join(command_parts), **kwargs)
                        if install.exit_code != 0:
                            raise RuntimeError("Offline sandbox dependency installation failed")
                    prepared[sandbox_id] = (
                        test_environment(
                            workspace, spec["source_roots"], spec["import_modules"], guard
                        ),
                        spec,
                    )
        state = prepared.get(sandbox_id)
        invoked = command
        if state is not None:
            prefix = "export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in state[0].items())
            invoked = "unset " + " ".join(HOST_DISPLAY_VARIABLES) + "; " + prefix + "; " + command
        junit = re.search(r"--junitxml=(/tmp/_swegemma_junit_[a-f0-9]+\.xml)", command)
        if junit:
            # An absent hook must never reuse evidence from an earlier test command.
            (guard / "origins.json").unlink(missing_ok=True)
        result = original(sandbox_id, invoked, **kwargs)
        if junit:
            destination = output / "source-checks" / sandbox_id
            destination.mkdir(parents=True, exist_ok=True)
            xml = original(sandbox_id, f"cat {junit[1]} 2>/dev/null || true")
            (destination / "junit.xml").write_text(xml.stdout)
            try:
                evidence = validate_origins(guard / "origins.json")
                write_json(destination / "origins.json", evidence)
            except (OSError, ValueError, KeyError) as error:
                failures.append(str(error))
            if guard.exists():
                shutil.copyfile(guard / "gemma_source_guard.py", destination / "guard.py")
            write_json(destination / "environment.json", state[1] if state is not None else {})
        return result

    sandbox.exec = execute

    def check() -> None:
        if failures:
            raise ValueError(f"Workspace source verification failed: {json.dumps(failures)}")

    return check
