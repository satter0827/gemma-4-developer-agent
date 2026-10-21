"""Publish metadata using fast-forward Git updates as a cross-host compare-and-swap."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gemma_agent_dev.submission_state import validate_state


class StateStore:
    def __init__(self, root: Path, branch: str, competition: str, timeout: int):
        self.root, self.branch, self.competition, self.timeout = root, branch, competition, timeout
        self.base: str | None = None

    def git(self, *args: str, input: str | None = None) -> str:
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Submission automation",
            "GIT_AUTHOR_EMAIL": "submission@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "Submission automation",
            "GIT_COMMITTER_EMAIL": "submission@users.noreply.github.com",
        }
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=environment,
            input=input,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Submission state Git operation failed; check access or concurrent update"
            )
        return result.stdout.strip()

    def load(self) -> dict[str, Any] | None:
        reference = f"refs/heads/{self.branch}"
        remote = self.git("ls-remote", "--heads", "origin", reference)
        if not remote:
            self.base = None
            return None
        self.base = remote.split()[0]
        self.git("fetch", "--quiet", "origin", self.base)
        return validate_state(
            json.loads(self.git("show", f"{self.base}:state.json")), self.competition
        )

    def save(self, state: dict[str, Any]) -> None:
        validate_state(state, self.competition)
        blob = self.git(
            "hash-object", "-w", "--stdin", input=json.dumps(state, sort_keys=True) + "\n"
        )
        tree = self.git("mktree", input=f"100644 blob {blob}\tstate.json\n")
        parent = ["-p", self.base] if self.base else []
        commit = self.git("commit-tree", tree, *parent, input="Update submission state\n")
        # No force: two writers with the same parent cannot overwrite one another.
        self.git("push", "--quiet", "origin", f"{commit}:refs/heads/{self.branch}")
        self.base = commit
