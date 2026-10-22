import json
from pathlib import Path
from typing import Any

import pytest

from gemma_agent_dev.common.artifacts import file_hash, write_json
from gemma_agent_dev.notebooks import experiments as bundles


def cpu_proof(root: Path, metadata: dict[str, Any]) -> Path:
    folder = root / "artifacts/fixture-cpu"
    folder.mkdir(parents=True)
    controls = {}
    for variant in ["workflow-v1", "sequential-v1"]:
        p = folder / "controls" / variant
        p.mkdir(parents=True)
        write_json(p / "origins.json", {"valid": True})
        write_json(
            p / "controls.json",
            [
                {
                    "instance_id": "fixture_1",
                    "unmodified": {"resolved": False, "error": None},
                    "reference": {"resolved": True, "error": None},
                }
            ],
        )
        controls[variant] = {
            name: file_hash(p / name) for name in ("origins.json", "controls.json")
        }
    report = folder / "session.json"
    write_json(
        report,
        {
            "kind": "experiment-verification",
            "status": "completed",
            "source_sha256": metadata["source_sha256"],
            "definition_sha256": metadata["definition_sha256"],
            "controls": controls,
            "task_ids": ["fixture_1"],
            "variants": json.loads(
                next((root / "artifacts/experiments").glob("*/preparation.json")).read_text()
            )["variants"],
        },
    )
    data = json.loads(report.read_text())
    data["variants"] = {name: v["sha256"] for name, v in data["variants"].items()}
    write_json(report, data)
    return report


def test_exact_payload_proof_and_quota_required_for_gpu(
    experiment_project: Path, frozen_experiment: Path
) -> None:
    root = experiment_project
    cpu = root / "cpu"
    digest = bundles.prepare(
        root, cpu, report_path=frozen_experiment, verify_only=True, cpu_report=None
    )
    metadata = json.loads((cpu / "source-manifest.json").read_text())
    proof = cpu_proof(root, metadata)
    with pytest.raises(ValueError, match="CPU proof"):
        bundles.prepare(
            root,
            root / "missing-proof",
            report_path=frozen_experiment,
            verify_only=False,
            cpu_report=None,
        )
    with pytest.raises(ValueError, match="quota"):
        bundles.prepare(
            root,
            root / "missing-quota",
            report_path=frozen_experiment,
            verify_only=False,
            cpu_report=proof,
        )
    with pytest.raises(ValueError, match="quota"):
        bundles.prepare(
            root,
            root / "insufficient",
            report_path=frozen_experiment,
            verify_only=False,
            cpu_report=proof,
            quota={"consumed": 2.8303, "remaining": 20, "completed_runs": 2, "active": False},
        )
    assert (
        bundles.prepare(
            root,
            root / "gpu",
            report_path=frozen_experiment,
            verify_only=False,
            cpu_report=proof,
            quota={"consumed": 0, "remaining": 4, "completed_runs": 0, "active": False},
        )
        == digest
    )
    assert (root / "gpu/source-bundle.zip").read_bytes() == (cpu / "source-bundle.zip").read_bytes()
    assert json.loads((root / "gpu/kernel-metadata.json").read_text())["enable_gpu"] is True
    with pytest.raises(FileExistsError):
        bundles.prepare(
            root, root / "gpu", report_path=frozen_experiment, verify_only=True, cpu_report=None
        )


def test_verify_local_uses_fixed_cpu_sdk_and_read_only_input(
    experiment_project: Path, frozen_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = experiment_project
    bundle = root / "bundle"
    bundles.prepare(root, bundle, report_path=frozen_experiment, verify_only=True, cpu_report=None)
    metadata = json.loads((bundle / "source-manifest.json").read_text())
    proof = cpu_proof(root, metadata)
    sdk = root / "artifacts/sdk"
    sdk.mkdir(parents=True)
    state = {"installed": {"packages": {}}, "inputs": {"fingerprint": "a"}}
    write_json(sdk / "prepared.json", state)
    monkeypatch.setattr(
        bundles, "sdk_settings", lambda *a: {"environment": sdk, "timeout_seconds": 10}
    )
    monkeypatch.setattr(bundles, "fingerprint", lambda d: state["inputs"])
    monkeypatch.setattr(bundles, "installed_state", lambda *a: state["installed"])

    def command(args: list[str], **kwargs: Any) -> str:
        assert "-m" in args and "session_worker" in args[2]
        bindings = json.loads(args[-1])
        assert set(bindings) == {"data", "output", "native_controls"}
        return json.dumps({"kind": "experiment-verification-receipt", "manifest": str(proof)})

    monkeypatch.setattr(bundles, "command", command)
    assert bundles.verify_local(root, bundle, Path("sdk.yaml")) == proof
    monkeypatch.setattr(bundles, "command", lambda *a, **kw: "CPU verified\n" + str(proof))
    with pytest.raises(ValueError):
        bundles.verify_local(root, bundle, Path("sdk.yaml"))
    (bundle / "source-bundle.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bundle changed"):
        bundles.verify_local(root, bundle, Path("sdk.yaml"))
    monkeypatch.setattr(bundles, "installed_state", lambda *a: {"changed": True})
    with pytest.raises(ValueError, match="environment changed"):
        bundles.verify_local(root, bundle, Path("sdk.yaml"))


def test_local_cpu_payload_does_not_render_a_remote_notebook(
    frozen_experiment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gemma_agent_dev.notebooks import experiments

    root = frozen_experiment.parents[3]
    destination = root / "local-cpu"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Local CPU proof must not inherit the remote notebook source limit")

    monkeypatch.setattr(experiments, "write_notebook", forbidden)
    digest = experiments.prepare(
        root,
        destination,
        report_path=frozen_experiment,
        verify_only=True,
        cpu_report=None,
        local_only=True,
    )
    assert (destination / "source-bundle.zip").is_file()
    assert json.loads((destination / "source-manifest.json").read_text())["source_sha256"] == digest
    assert not (destination / "run.py").exists()
    with pytest.raises(ValueError, match="remote launch"):
        experiments.prepare(
            root,
            root / "bad-local",
            report_path=frozen_experiment,
            verify_only=False,
            cpu_report=None,
            local_only=True,
        )
