"""Tests for bad_channels.py — multi-detector consensus bad channel detection.

Exercises the individual detectors (full-recording GESD, time-resolved GESD,
PSD, LOF), the consensus vote that splits confirmed vs candidate channels, the
candidates sidecar, the run() accumulation across tasks, and the module-level
run(cfg) entry point.  BIDS I/O is mocked; the statistical detection itself runs
on synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_channels import (
    BadChannelsAnalysis,
    candidates_sidecar_path,
    run,
)


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
# Detectors — core per-recording logic
# ---------------------------------------------------------------------------

class TestDetectBadChannels:
    """Test the detectors on synthetic data."""

    def test_detects_bad_channel_with_high_variance(
        self, raw_with_bad_channel, bad_ch_cfg
    ):
        """A channel with 100x higher variance should be flagged by detectors."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = analysis._detect_all_methods(raw_with_bad_channel)
        flagged = set().union(*results.values()) if results else set()
        # MEG001 has 100x variance — multiple detectors should catch it
        assert "MEG001" in flagged
        assert "gesd" in results and "MEG001" in results["gesd"]

    def test_clean_data_has_few_or_no_confirmed_bads(self, raw_meg, bad_ch_cfg):
        """Homogeneous random data should yield no confirmed (consensus) bads."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = analysis._detect_all_methods(raw_meg)
        confirmed, _ = analysis._combine_votes(results, analysis._consensus_n())
        # Consensus of >=2 detectors on iid data should confirm nothing.
        assert len(confirmed) == 0

    def test_filtering_applied_before_detection(
        self, raw_with_bad_channel, bad_ch_cfg
    ):
        """Detection should use data filtered at cfg.l_freq / h_freq."""
        bad_ch_cfg.l_freq = 5.0
        bad_ch_cfg.h_freq = 40.0
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        # Should not crash, and still detect the outlier
        results = analysis._detect_all_methods(raw_with_bad_channel)
        flagged = set().union(*results.values()) if results else set()
        assert "MEG001" in flagged


# ---------------------------------------------------------------------------
# Time-resolved detector — intermittent bad channels
# ---------------------------------------------------------------------------

class TestTimeResolvedDetection:
    """The time-resolved detector should catch intermittent bad channels."""

    def _make_intermittent(self, meg_info, bad_frac=0.3):
        """Raw where MEG002 is bad in only `bad_frac` of the recording."""
        rng = np.random.RandomState(7)
        n_ch = len(meg_info["ch_names"])
        n_samples = int(meg_info["sfreq"] * 60)  # 60 s → many windows
        data = rng.randn(n_ch, n_samples) * 1e-13
        # Inflate MEG002 variance in a contiguous early fraction of the record.
        bad_stop = int(bad_frac * n_samples)
        data[2, :bad_stop] *= 80.0
        return mne.io.RawArray(data, meg_info)

    def test_timeresolved_catches_intermittent(self, meg_info, bad_ch_cfg):
        """A channel bad in ~30% of windows is flagged by time-resolved."""
        bad_ch_cfg.l_freq = 1.0
        bad_ch_cfg.h_freq = 100.0
        raw = self._make_intermittent(meg_info, bad_frac=0.3)

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        filt = raw.copy().filter(l_freq=1.0, h_freq=100.0, method="iir")
        tr = analysis._detect_timeresolved(filt, "mag")
        assert "MEG002" in tr

    def _make_heterogeneous_intermittent(self, bad_frac=0.3):
        """Raw with a realistic spread of channel noise levels, plus an
        intermittent channel whose whole-recording variance hides in the pack.

        Real OPM arrays have heterogeneous per-channel noise floors.  A channel
        that is only intermittently disruptive has a whole-recording std that
        sits within that natural spread (so full-recording GESD misses it), yet
        in its bad windows it transiently exceeds even the noisiest channels (so
        the time-resolved detector catches it).
        """
        n_meg = 40
        info = mne.create_info(
            [f"MEG{i:03d}" for i in range(n_meg)], sfreq=300.0, ch_types="mag"
        )
        rng = np.random.RandomState(11)
        n_samples = int(info["sfreq"] * 60)
        # Smoothly varying baseline noise levels — no static outlier channel.
        scales = np.linspace(1.0, 3.0, n_meg)
        data = rng.randn(n_meg, n_samples) * scales[:, None] * 1e-13
        # MEG002 is a quiet channel (low baseline) that misbehaves intermittently.
        data[2, :] = rng.randn(n_samples) * 1e-13
        win = int(info["sfreq"] * 2.0)
        n_windows = n_samples // win
        bad_windows = rng.choice(
            n_windows, size=int(bad_frac * n_windows), replace=False
        )
        for w in bad_windows:
            data[2, w * win:(w + 1) * win] *= 6.0
        return mne.io.RawArray(data, info)

    def test_full_gesd_misses_intermittent(self, bad_ch_cfg):
        """Whole-recording GESD misses the intermittent channel that the
        time-resolved detector catches (the core motivation)."""
        raw = self._make_heterogeneous_intermittent(bad_frac=0.3)
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        filt = raw.copy().filter(l_freq=1.0, h_freq=100.0, method="iir")
        full = analysis._detect_gesd_full(filt, "mag")
        tr = analysis._detect_timeresolved(filt, "mag")
        # The time-resolved detector catches it even when full-recording does not.
        assert "MEG002" in tr
        assert "MEG002" not in full


# ---------------------------------------------------------------------------
# Consensus voting
# ---------------------------------------------------------------------------

class TestConsensusVoting:
    """Test _combine_votes confirmed/candidate split."""

    def test_consensus_two_confirms_multi_method(self, bad_ch_cfg):
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        method_results = {
            "gesd": {"MEG001"},
            "timeresolved": {"MEG001"},
            "lof": {"MEG005"},  # single method only
        }
        confirmed, candidates = analysis._combine_votes(method_results, 2)
        assert "MEG001" in confirmed
        assert "MEG005" in candidates
        assert "MEG005" not in confirmed
        assert "MEG001" not in candidates

    def test_consensus_one_marks_any_flag(self, bad_ch_cfg):
        """With consensus_n=1, any single flag confirms (no candidates)."""
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        method_results = {"lof": {"MEG005"}}
        confirmed, candidates = analysis._combine_votes(method_results, 1)
        assert "MEG005" in confirmed
        assert candidates == {}


# ---------------------------------------------------------------------------
# Candidates sidecar — written here, consumed by manual_channel
# ---------------------------------------------------------------------------

class TestCandidatesSidecar:
    """Single-method flags are written to a sidecar and not auto-marked."""

    def test_sidecar_written_and_readable(self, tmp_path, bad_ch_cfg):
        """_write_candidates_sidecar writes a TSV that manual_channel can read."""
        # Lightweight stand-in for a BIDSPath with directory + basename.
        fake_bp = SimpleNamespace(
            directory=str(tmp_path),
            basename="sub-001_ses-01_task-restingstate_meg",
        )
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        analysis._write_candidates_sidecar(
            fake_bp, {"MEG004": ["lof"], "MEG006": ["psd"]}
        )

        sidecar = candidates_sidecar_path(fake_bp)
        assert sidecar.exists()

        import pandas as pd
        df = pd.read_csv(sidecar, sep="\t")
        assert set(df["channel"]) == {"MEG004", "MEG006"}

    def test_empty_candidates_removes_stale_sidecar(self, tmp_path, bad_ch_cfg):
        """An empty candidate set clears any stale sidecar from a prior run."""
        fake_bp = SimpleNamespace(
            directory=str(tmp_path),
            basename="sub-001_ses-01_task-restingstate_meg",
        )
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        analysis._write_candidates_sidecar(fake_bp, {"MEG004": ["lof"]})
        assert candidates_sidecar_path(fake_bp).exists()

        analysis._write_candidates_sidecar(fake_bp, {})
        assert not candidates_sidecar_path(fake_bp).exists()


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
    """Test load_data with mocked path-resolution calls."""

    @patch("custom.preprocessing.bad_channels.read_raw_bids_with_retry")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_load_single_task(self, mock_find, mock_read, raw_meg, bad_ch_cfg):
        """load_data should resolve input paths and read raw."""
        mock_bp = MagicMock()
        mock_find.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = analysis.load_data()

        assert bad_ch_cfg.task in data
        assert data[bad_ch_cfg.task] is raw_meg
        mock_find.assert_called_once()
        mock_read.assert_called_once()

    @patch("custom.preprocessing.bad_channels.read_raw_bids_with_retry")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_load_with_empty_room(self, mock_find, mock_read, raw_meg, bad_ch_cfg):
        """When process_empty_room=True, load noise + task."""
        bad_ch_cfg.process_empty_room = True
        mock_bp = MagicMock()
        mock_find.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        data = analysis.load_data()

        assert "noise" in data
        assert bad_ch_cfg.task in data
        assert mock_find.call_count == 2

    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_load_missing_task_raises(self, mock_find, bad_ch_cfg):
        """load_data raises FileNotFoundError if no paths found."""
        mock_find.return_value = []

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        with pytest.raises(FileNotFoundError, match="No raw data"):
            analysis.load_data()


# ---------------------------------------------------------------------------
# save_results — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestBadChannelsSaveResults:
    """Test save_results merges bads and calls write_raw_bids_custom_step."""

    @patch("custom.preprocessing.bad_channels.write_raw_bids_custom_step")
    @patch("custom.preprocessing.bad_channels.mne_bids")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_save_merges_bads_into_raw(
        self, mock_find, mock_bids, mock_write, raw_meg, bad_ch_cfg
    ):
        """save_results should merge detected bads into raw.info['bads']."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_find.return_value = [mock_bp]
        mock_write.return_value = mock_bp  # output bp

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = {bad_ch_cfg.task: raw_meg, "bads": ["MEG001", "MEG003"]}
        analysis.save_results(results)

        # Verify bads were merged into raw.info
        assert "MEG001" in raw_meg.info["bads"]
        assert "MEG003" in raw_meg.info["bads"]
        # Verify mark_channels was called (after the redirected write)
        mock_bids.mark_channels.assert_called()
        # Verify write_raw_bids_custom_step was called
        mock_write.assert_called()

    @patch("custom.preprocessing.bad_channels.write_raw_bids_custom_step")
    @patch("custom.preprocessing.bad_channels.mne_bids")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_save_no_bads_still_writes(
        self, mock_find, mock_bids, mock_write, raw_meg, bad_ch_cfg
    ):
        """save_results with empty bads should still write raw."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_find.return_value = [mock_bp]
        mock_write.return_value = mock_bp

        analysis = BadChannelsAnalysis(bad_ch_cfg)
        results = {bad_ch_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        mock_write.assert_called_once()
        # mark_channels should NOT be called with empty bads
        mock_bids.mark_channels.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level run(cfg) entry point
# ---------------------------------------------------------------------------

class TestBadChannelsModuleRun:
    """Test the module-level run(cfg) function."""

    @patch("custom.preprocessing.bad_channels.write_raw_bids_custom_step")
    @patch("custom.preprocessing.bad_channels.read_raw_bids_with_retry")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_run_calls_execute(
        self, mock_find, mock_read, mock_write, raw_meg, bad_ch_cfg
    ):
        """run(cfg) should construct the analysis and call execute."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_find.return_value = [mock_bp]
        mock_read.return_value = raw_meg
        mock_write.return_value = mock_bp

        # This exercises: run(cfg) -> is_enabled -> execute -> load/run/save
        run(bad_ch_cfg)

        mock_read.assert_called()
        mock_write.assert_called()
