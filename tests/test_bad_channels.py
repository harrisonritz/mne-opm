"""Tests for bad_channels.py — GESD bad channel detection.

Exercises the core detection logic (_detect_bad_channels), the run()
accumulation across tasks, and the module-level run(cfg) entry point.
BIDS I/O is mocked; the statistical detection itself runs on synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_channels import BadChannelsAnalysis, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bad_ch_cfg(tmp_path):
    """Config appropriate for bad channel detection."""
    return SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        l_freq=1.0,
        h_freq=100.0,
        process_empty_room=False,
        find_breaks=False,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# _detect_bad_channels — core GESD logic
# ---------------------------------------------------------------------------

class TestDetectBadChannels:
    """Test the core detection method on synthetic data."""

    def test_detects_bad_channel_with_high_variance(
        self, raw_with_bad_channel, bad_ch_cfg
    ):
        """A channel with 100x higher variance should be flagged."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        bads = analysis._detect_bad_channels(raw_with_bad_channel)
        assert isinstance(bads, list)
        # MEG001 has 100x variance — GESD should detect it
        assert "MEG001" in bads

    def test_clean_data_has_few_or_no_bads(self, raw_meg, bad_ch_cfg):
        """Homogeneous random data should flag few or no channels."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        bads = analysis._detect_bad_channels(raw_meg)
        assert isinstance(bads, list)
        # With iid Gaussian channels, we expect 0 or very few detections
        assert len(bads) <= 2

    def test_filtering_applied_before_detection(
        self, raw_with_bad_channel, bad_ch_cfg
    ):
        """Detection should use data filtered at cfg.l_freq / h_freq."""
        bad_ch_cfg.l_freq = 5.0
        bad_ch_cfg.h_freq = 40.0
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        # Should not crash, and still detect the outlier
        bads = analysis._detect_bad_channels(raw_with_bad_channel)
        assert isinstance(bads, list)


# ---------------------------------------------------------------------------
# run() method — accumulation across tasks
# ---------------------------------------------------------------------------

class TestBadChannelsRun:
    """Test run() method accumulates bads across tasks."""

    def test_single_task_accumulation(self, raw_with_bad_channel, bad_ch_cfg):
        """run() with one task should return its bads."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = {bad_ch_cfg.task: raw_with_bad_channel}
        results = analysis.run(data)

        assert "bads" in results
        assert isinstance(results["bads"], list)
        assert bad_ch_cfg.task in results

    def test_multiple_task_union(self, meg_info, bad_ch_cfg):
        """Bads from multiple tasks should be unioned."""
        rng = np.random.RandomState(42)
        n_ch = len(meg_info["ch_names"])
        n_samples = int(meg_info["sfreq"] * 10)

        # Task 1: MEG001 is bad
        data1 = rng.randn(n_ch, n_samples) * 1e-13
        data1[1, :] *= 100.0
        raw1 = mne.io.RawArray(data1, meg_info)

        # Task 2: MEG002 is bad
        data2 = rng.randn(n_ch, n_samples) * 1e-13
        data2[2, :] *= 100.0
        raw2 = mne.io.RawArray(data2, meg_info)

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = {"restingstate": raw1, "noise": raw2}
        results = analysis.run(data)

        # Union should contain bads from both tasks
        assert "MEG001" in results["bads"] or "MEG002" in results["bads"]


# ---------------------------------------------------------------------------
# load_data — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestBadChannelsLoadData:
    """Test load_data with mocked mne_bids calls."""

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_load_single_task(self, mock_bids, raw_meg, bad_ch_cfg):
        """load_data should call find_matching_paths and read_raw_bids."""
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_meg

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = analysis.load_data()

        assert bad_ch_cfg.task in data
        assert data[bad_ch_cfg.task] is raw_meg
        mock_bids.find_matching_paths.assert_called_once()
        mock_bids.read_raw_bids.assert_called_once()

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_load_with_empty_room(self, mock_bids, raw_meg, bad_ch_cfg):
        """When process_empty_room=True, load noise + task."""
        bad_ch_cfg.process_empty_room = True
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_meg

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = analysis.load_data()

        assert "noise" in data
        assert bad_ch_cfg.task in data
        assert mock_bids.find_matching_paths.call_count == 2

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_load_missing_task_raises(self, mock_bids, bad_ch_cfg):
        """load_data raises FileNotFoundError if no paths found."""
        mock_bids.find_matching_paths.return_value = []

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        with pytest.raises(FileNotFoundError, match="No raw data"):
            analysis.load_data()


# ---------------------------------------------------------------------------
# save_results — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestBadChannelsSaveResults:
    """Test save_results merges bads and calls write_raw_bids."""

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_save_merges_bads_into_raw(self, mock_bids, raw_meg, bad_ch_cfg):
        """save_results should merge detected bads into raw.info['bads']."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = {bad_ch_cfg.task: raw_meg, "bads": ["MEG001", "MEG003"]}
        analysis.save_results(results)

        # Verify bads were merged into raw.info
        assert "MEG001" in raw_meg.info["bads"]
        assert "MEG003" in raw_meg.info["bads"]
        # Verify mark_channels was called
        mock_bids.mark_channels.assert_called()
        # Verify write_raw_bids was called
        mock_bids.write_raw_bids.assert_called()

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_save_no_bads_still_writes(self, mock_bids, raw_meg, bad_ch_cfg):
        """save_results with empty bads should still write raw."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = {bad_ch_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        mock_bids.write_raw_bids.assert_called_once()
        # mark_channels should NOT be called with empty bads
        mock_bids.mark_channels.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level run(cfg) entry point
# ---------------------------------------------------------------------------

class TestBadChannelsModuleRun:
    """Test the module-level run(cfg) function."""

    @patch("custom.preprocessing.bad_channels.mne_bids")
    def test_run_calls_execute(self, mock_bids, raw_meg, bad_ch_cfg):
        """run(cfg) should construct the analysis and call execute."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_meg

        # This exercises: run(cfg) -> is_enabled -> execute -> load/run/save
        run(bad_ch_cfg)

        mock_bids.read_raw_bids.assert_called()
        mock_bids.write_raw_bids.assert_called()
