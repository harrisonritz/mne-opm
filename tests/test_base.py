"""Tests for preprocessing._base — BaseAnalysis class and constants."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import pytest

from custom.preprocessing._base import (
    SEGMENT_LEN_SEC,
    BaseAnalysis,
    have_qt_browser,
)


# ---------------------------------------------------------------------------
# Concrete subclass for testing the abstract base
# ---------------------------------------------------------------------------

class _DummyAnalysis(BaseAnalysis):
    """Minimal concrete implementation for testing BaseAnalysis."""

    ANALYSIS_KEY = "dummy"
    ANALYSIS_NAME = "dummy_analysis"

    def __init__(self, cfg, *, enabled=True, load_val=None, run_val=None):
        super().__init__(cfg)
        self._enabled = enabled
        self._load_val = load_val or {"data": 42}
        self._run_val = run_val or {"result": 99}
        self.save_called_with = None

    def is_enabled(self) -> bool:
        return self._enabled

    def load_data(self) -> Dict[str, Any]:
        return self._load_val

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_val

    def save_results(self, results: Dict[str, Any]) -> None:
        self.save_called_with = results


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_segment_len_sec(self):
        assert SEGMENT_LEN_SEC == 1.0
        assert isinstance(SEGMENT_LEN_SEC, float)


# ---------------------------------------------------------------------------
# have_qt_browser
# ---------------------------------------------------------------------------

class TestHaveQtBrowser:
    def test_returns_bool(self):
        result = have_qt_browser()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# BaseAnalysis
# ---------------------------------------------------------------------------

class TestBaseAnalysis:
    """Tests for the BaseAnalysis abstract base class via _DummyAnalysis."""

    def test_init_stores_cfg(self):
        cfg = SimpleNamespace(x=1)
        analysis = _DummyAnalysis(cfg)
        assert analysis.cfg is cfg

    def test_analysis_key_and_name(self):
        analysis = _DummyAnalysis(SimpleNamespace())
        assert analysis.ANALYSIS_KEY == "dummy"
        assert analysis.ANALYSIS_NAME == "dummy_analysis"

    def test_log_prints(self, capsys):
        analysis = _DummyAnalysis(SimpleNamespace())
        analysis.log("hello world")
        captured = capsys.readouterr()
        assert "[dummy_analysis] hello world" in captured.out

    def test_is_enabled(self):
        a_on = _DummyAnalysis(SimpleNamespace(), enabled=True)
        a_off = _DummyAnalysis(SimpleNamespace(), enabled=False)
        assert a_on.is_enabled() is True
        assert a_off.is_enabled() is False

    def test_execute_calls_load_run_save(self, capsys):
        data_val = {"raw": "mock_data"}
        result_val = {"cleaned": "mock_result"}
        analysis = _DummyAnalysis(
            SimpleNamespace(), load_val=data_val, run_val=result_val
        )
        analysis.execute()

        # save_results should have been called with the run output
        assert analysis.save_called_with == result_val

        # Verify log messages
        output = capsys.readouterr().out
        assert "Starting analysis" in output
        assert "Loading data" in output
        assert "Running analysis" in output
        assert "Saving results" in output
        assert "Analysis complete" in output

    def test_execute_pipeline_data_flow(self):
        """Verify that data flows load -> run -> save correctly."""
        load_return = {"step1": "loaded"}
        run_return = {"step2": "processed"}

        class _TrackerAnalysis(_DummyAnalysis):
            def run(self, data):
                assert data == load_return
                return run_return

        analysis = _TrackerAnalysis(
            SimpleNamespace(), load_val=load_return
        )
        analysis.execute()
        assert analysis.save_called_with == run_return

    def test_cannot_instantiate_abstract(self):
        """BaseAnalysis itself should not be instantiable."""
        with pytest.raises(TypeError):
            BaseAnalysis(SimpleNamespace())

    def test_subclass_must_implement_all_methods(self):
        """A subclass missing any abstract method should not be instantiable."""

        class _Incomplete(BaseAnalysis):
            ANALYSIS_KEY = "incomplete"
            ANALYSIS_NAME = "incomplete"

            def is_enabled(self):
                return True

            def load_data(self):
                return {}

            # Missing: run, save_results

        with pytest.raises(TypeError):
            _Incomplete(SimpleNamespace())
