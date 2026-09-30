from collections.abc import Callable
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src/gemma_agent_dev/app.py"


def test_empty_results_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMMA_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    assert "評価結果はまだありません" in app.info[0].value


def test_partial_run_comparison_is_flagged(
    make_run: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_run()
    make_run("run-b", status="running", completed=1)
    monkeypatch.setenv("GEMMA_RESULTS_DIR", str(first.parent))
    app = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not app.exception
    assert any("比較条件が異なる" in warning.value for warning in app.warning)
    app.selectbox[0].select(1).run()
    assert not app.exception
    assert len(app.dataframe) == 2
