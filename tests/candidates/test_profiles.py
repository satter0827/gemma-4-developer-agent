from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_profiles_are_explicit_and_unknown_names_do_not_start_preparation(
    preflight_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.candidates.preflight import PreflightSettings
    from gemma_agent_dev.cli import main

    monkeypatch.chdir(preflight_project)
    with pytest.raises(SystemExit) as raised:
        main(["preflight", "--config", "preflight.yaml"])
    assert raised.value.code == 2
    with pytest.raises(ValueError, match="Unknown"):
        PreflightSettings.load(preflight_project, Path("preflight.yaml"), "unknown")
    assert not (preflight_project / "artifacts/preflight").exists()
