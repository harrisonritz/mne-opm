"""Tests for regress.py — general sensor regression.

Extends the existing tests in test_preprocessing_analyses.py by covering:
- load_data / save_results with mocked BIDS I/O
- The module-level run(cfg) entry point
- Time-varying regression with frequency bands
- Standard regression end-to-end flow
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.regress import RegressAnalysis, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def regress_cfg(tmp_path):
    """Config for regression."""
    return SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        _regress=True,
        _regress_preds=["ref_meg"],
        _regress_timevarying=False,
        _regress_window=3.0,
        _regress_freqs=None,
        _regress_plot=False,
        process_empty_room=False,
        find_breaks=False,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# _regress — standard mode
# ---------------------------------------------------------------------------

class TestRegressStandard:
    """Test standard (EOGRegression) regression."""

    def test_standard_reduces_contamination(
        self, raw_with_ref_contamination, regress_cfg
    ):
        """Standard regression should remove ref contamination from MEG."""
        raw, brain = raw_with_ref_contamination
        analysis = RegressAnalysis(regress_cfg)

        var_before = np.var(raw.get_data(picks="mag"))
        raw_clean = analysis._regress(raw.copy())
        var_after = np.var(raw_clean.get_data(picks="mag"))

        assert var_after < var_before, (
            "Standard regression should reduce MEG variance "
            "by removing ref contamination"
        )

    def test_standard_returns_raw(
        self, raw_with_ref_contamination, regress_cfg
    ):
        raw, _ = raw_with_ref_contamination
        analysis = RegressAnalysis(regress_cfg)
        result = analysis._regress(raw.copy())
        assert isinstance(result, mne.io.BaseRaw)


# ---------------------------------------------------------------------------
# _regress — time-varying mode
# ---------------------------------------------------------------------------

class TestRegressTimeVarying:
    """Test time-varying (sliding window QR) regression."""

    def test_timevarying_reduces_contamination(
        self, raw_with_ref_contamination, regress_cfg
    ):
        """Time-varying regression should remove ref contamination."""
        raw, _ = raw_with_ref_contamination
        regress_cfg._regress_timevarying = True
        analysis = RegressAnalysis(regress_cfg)

        var_before = np.var(raw.get_data(picks="mag"))
        raw_clean = analysis._regress(raw.copy())
        var_after = np.var(raw_clean.get_data(picks="mag"))

        assert var_after < var_before

    def test_timevarying_with_freq_bands(
        self, raw_with_ref_contamination, regress_cfg
    ):
        """Time-varying regression with frequency bands should work."""
        raw, _ = raw_with_ref_contamination
        regress_cfg._regress_timevarying = True
        regress_cfg._regress_freqs = [(None, 10.0), (10.0, 50.0)]
        analysis = RegressAnalysis(regress_cfg)

        var_before = np.var(raw.get_data(picks="mag"))
        raw_clean = analysis._regress(raw.copy())
        var_after = np.var(raw_clean.get_data(picks="mag"))

        assert var_after < var_before

    def test_timevarying_output_shape(
        self, raw_with_ref_contamination, regress_cfg
    ):
        """Output should have same number of channels and samples."""
        raw, _ = raw_with_ref_contamination
        regress_cfg._regress_timevarying = True
        analysis = RegressAnalysis(regress_cfg)

        raw_clean = analysis._regress(raw.copy())
        assert raw_clean.get_data().shape == raw.get_data().shape


# ---------------------------------------------------------------------------
# run() method
# ---------------------------------------------------------------------------

class TestRegressRun:
    def test_run_processes_task(
        self, raw_with_ref_contamination, regress_cfg
    ):
        raw, _ = raw_with_ref_contamination
        analysis = RegressAnalysis(regress_cfg)
        data = {regress_cfg.task: raw}
        results = analysis.run(data)

        assert regress_cfg.task in results
        assert isinstance(results[regress_cfg.task], mne.io.BaseRaw)

    def test_run_with_noise_task(
        self, raw_with_ref_contamination, regress_cfg
    ):
        raw, _ = raw_with_ref_contamination
        raw_noise = raw.copy()
        analysis = RegressAnalysis(regress_cfg)
        data = {"noise": raw_noise, regress_cfg.task: raw}
        results = analysis.run(data)

        assert "noise" in results
        assert regress_cfg.task in results


# ---------------------------------------------------------------------------
# load_data — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestRegressLoadData:
    @patch("custom.preprocessing.regress.read_raw_bids_with_retry")
    @patch("custom.preprocessing.regress.mne_bids")
    def test_load_single_task(self, mock_bids, mock_read, raw_meg, regress_cfg):
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = RegressAnalysis(regress_cfg)
        data = analysis.load_data()

        assert regress_cfg.task in data
        mock_read.assert_called_once()

    @patch("custom.preprocessing.regress.read_raw_bids_with_retry")
    @patch("custom.preprocessing.regress.mne_bids")
    def test_load_with_empty_room(self, mock_bids, mock_read, raw_meg, regress_cfg):
        regress_cfg.process_empty_room = True
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = RegressAnalysis(regress_cfg)
        data = analysis.load_data()

        assert "noise" in data
        assert regress_cfg.task in data

    @patch("custom.preprocessing.regress.mne_bids")
    def test_load_no_files_raises(self, mock_bids, regress_cfg):
        mock_bids.find_matching_paths.return_value = []
        analysis = RegressAnalysis(regress_cfg)
        with pytest.raises(FileNotFoundError):
            analysis.load_data()


# ---------------------------------------------------------------------------
# save_results — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestRegressSaveResults:
    @patch("custom.preprocessing.regress.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.regress.mne_bids")
    def test_save_writes_data(self, mock_bids, mock_write, raw_meg, regress_cfg):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = RegressAnalysis(regress_cfg)
        results = {regress_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        mock_write.assert_called_once()

    @patch("custom.preprocessing.regress.mne_bids")
    def test_save_no_paths_raises(self, mock_bids, raw_meg, regress_cfg):
        mock_bids.find_matching_paths.return_value = []
        analysis = RegressAnalysis(regress_cfg)
        results = {regress_cfg.task: raw_meg, "bads": []}
        with pytest.raises(FileNotFoundError):
            analysis.save_results(results)

    @patch("custom.preprocessing.regress.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.regress.mne_bids")
    def test_save_with_empty_room_association(
        self, mock_bids, mock_write, raw_meg, regress_cfg
    ):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = RegressAnalysis(regress_cfg)
        results = {"noise": raw_meg, regress_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        assert mock_write.call_count == 2
        task_call_kwargs = mock_write.call_args_list[-1][1]
        assert "empty_room" in task_call_kwargs


# ---------------------------------------------------------------------------
# Module-level run(cfg) entry point
# ---------------------------------------------------------------------------

class TestRegressModuleRun:
    def test_disabled_exits_early(self, capsys):
        cfg = SimpleNamespace(_regress=False)
        run(cfg)
        captured = capsys.readouterr()
        assert "Disabled" in captured.out

    @patch("custom.preprocessing.regress.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.regress.read_raw_bids_with_retry")
    @patch("custom.preprocessing.regress.mne_bids")
    def test_run_end_to_end(
        self, mock_bids, mock_read, mock_write, raw_with_ref_contamination, regress_cfg
    ):
        raw, _ = raw_with_ref_contamination
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw

        run(regress_cfg)

        mock_read.assert_called()
        mock_write.assert_called()
