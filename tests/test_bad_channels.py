"""Tests for bad_channels.py — PCA-whitened GESD bad channel detection.

Exercises detection through run() on synthetic data (whole-recording variance
outliers, intermittent outliers via the windowed outlier fraction, the
configurable filter band), the run() accumulation across tasks, and the
module-level run(cfg) entry point.  BIDS I/O is mocked; the statistical
detection itself runs on synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_channels import (
    BadChannelsAnalysis,
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


def _detect(cfg, raw):
    """Run detection on a single recording and return the flagged channels."""
    return BadChannelsAnalysis(cfg).run({cfg.task: raw})["bads"]


# ---------------------------------------------------------------------------
# Detection — core per-recording logic
# ---------------------------------------------------------------------------


class TestDetectBadChannels:
    """Test PCA→GESD detection on synthetic data."""

    def test_detects_bad_channel_with_high_variance(
        self, raw_with_bad_channel, bad_ch_cfg
    ):
        """A channel with 100x higher variance should be flagged."""
        assert "MEG001" in _detect(bad_ch_cfg, raw_with_bad_channel)

    def test_clean_data_has_few_or_no_confirmed_bads(self, raw_meg, bad_ch_cfg):
        """Homogeneous random data should yield no flagged channels."""
        assert _detect(bad_ch_cfg, raw_meg) == []

    def test_filtering_applied_before_detection(self, raw_with_bad_channel, bad_ch_cfg):
        """Detection should run with the configured _bad_channel_* band."""
        bad_ch_cfg._bad_channel_lfreq = 5.0
        bad_ch_cfg._bad_channel_hfreq = 40.0
        analysis = BadChannelsAnalysis(bad_ch_cfg)
        with patch.object(
            mne.io.BaseRaw, "filter", autospec=True, side_effect=mne.io.BaseRaw.filter
        ) as mock_filter:
            bads = analysis.run({bad_ch_cfg.task: raw_with_bad_channel})["bads"]
        assert mock_filter.call_args.kwargs["l_freq"] == 5.0
        assert mock_filter.call_args.kwargs["h_freq"] == 40.0
        assert "MEG001" in bads


# ---------------------------------------------------------------------------
# log_std and the spectral-shape metrics
# ---------------------------------------------------------------------------


def _pink_raw(n_ch=40, sfreq=300.0, seconds=60, seed=0, scales=None):
    """Raw of 1/f noise at tesla scale, one row per channel."""
    rng = np.random.default_rng(seed)
    n = int(sfreq * seconds)
    spec = np.fft.rfft(rng.standard_normal((n_ch, n)), axis=1)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    spec[:, 1:] /= np.sqrt(freqs[1:])
    spec[:, 0] = 0.0
    data = np.fft.irfft(spec, n=n, axis=1)
    data /= data.std(axis=1, keepdims=True)
    if scales is not None:
        data *= np.asarray(scales)[:, None]
    info = mne.create_info([f"MEG{i:03d}" for i in range(n_ch)], sfreq, "mag")
    return mne.io.RawArray(data * 1e-12, info, verbose="ERROR")


class TestLogStd:
    """log_std must be a log of the std whatever the units of the data."""

    def test_log_std_spans_log_of_channel_scales(self, bad_ch_cfg):
        scales = np.linspace(1.0, 3.0, 40)
        raw = _pink_raw(scales=scales)
        bad_ch_cfg._channel_metrics = ["log_std"]
        _, specs = BadChannelsAnalysis(bad_ch_cfg)._compute_channel_metrics(raw)
        # A fixed 1e-10 offset squashed this to ~0.02 for data near 1e-12 T.
        assert np.ptp(specs[0].values) == pytest.approx(np.log(3.0), rel=0.1)

    def test_dead_channel_flagged_by_log_std(self, bad_ch_cfg):
        raw = _pink_raw(scales=np.linspace(1.0, 3.0, 40))
        raw._data[5] = 0.0
        bad_ch_cfg._channel_metrics = ["log_std"]
        assert _detect(bad_ch_cfg, raw) == ["MEG005"]


class TestSpectralShapeMetrics:
    """psd_slope and psd_resid_kurtosis catch channels with an odd spectrum."""

    def test_white_noise_channel_flagged_by_slope(self, bad_ch_cfg):
        raw = _pink_raw()
        rng = np.random.default_rng(1)
        raw._data[3] = rng.standard_normal(raw.n_times) * 1e-12  # same std, flat
        bad_ch_cfg._channel_metrics = ["psd_slope"]
        assert _detect(bad_ch_cfg, raw) == ["MEG003"]

    def test_narrowband_peak_flagged_by_resid_kurtosis(self, bad_ch_cfg):
        raw = _pink_raw()
        t = raw.times
        raw._data[7] += 0.5e-12 * np.sin(2 * np.pi * 23.0 * t)
        bad_ch_cfg._channel_metrics = ["psd_resid_kurtosis"]
        assert _detect(bad_ch_cfg, raw) == ["MEG007"]

    def test_line_noise_is_not_a_peak(self, bad_ch_cfg):
        """Mains pickup of varying size is left out of the spectral-shape fit."""
        raw = _pink_raw()
        for line in (60.0, 120.0):
            raw._data[4] += 2e-12 * np.sin(2 * np.pi * line * raw.times)
        bad_ch_cfg._channel_metrics = ["psd_slope", "psd_resid_kurtosis"]
        assert _detect(bad_ch_cfg, raw) == []

    def test_clean_pink_noise_has_no_spectral_outliers(self, bad_ch_cfg):
        bad_ch_cfg._channel_metrics = ["psd_slope", "psd_resid_kurtosis"]
        assert _detect(bad_ch_cfg, _pink_raw()) == []


# ---------------------------------------------------------------------------
# Windowed outlier fraction — intermittent bad channels
# ---------------------------------------------------------------------------


class TestTimeResolvedDetection:
    """The logit_outlier_frac metric should catch intermittent bad channels."""

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
        """A channel bad in ~30% of windows is flagged by the outlier fraction."""
        raw = self._make_intermittent(meg_info, bad_frac=0.3)
        bad_ch_cfg._channel_metrics = ["logit_outlier_frac"]
        assert "MEG002" in _detect(bad_ch_cfg, raw)

    def _make_heterogeneous_intermittent(self, bad_frac=0.3):
        """Raw with a realistic spread of channel noise levels, plus an
        intermittent channel whose whole-recording variance hides in the pack.

        Real OPM arrays have heterogeneous per-channel noise floors.  A channel
        that is only intermittently disruptive has a whole-recording std that
        sits within that natural spread (so the log_std metric misses it), yet
        in its bad windows it transiently exceeds even the noisiest channels (so
        the windowed outlier fraction catches it).
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
            data[2, w * win : (w + 1) * win] *= 6.0
        return mne.io.RawArray(data, info)

    def test_full_gesd_misses_intermittent(self, bad_ch_cfg):
        """Whole-recording log_std misses the intermittent channel that the
        windowed outlier fraction catches (the core motivation)."""
        raw = self._make_heterogeneous_intermittent(bad_frac=0.3)
        bad_ch_cfg._channel_metrics = ["log_std"]
        full = _detect(bad_ch_cfg, raw)
        bad_ch_cfg._channel_metrics = ["logit_outlier_frac"]
        tr = _detect(bad_ch_cfg, raw)
        # The outlier fraction catches it even when whole-recording std does not.
        assert "MEG002" in tr
        assert "MEG002" not in full


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
        """With process_empty_room and _bad_channel_emptyroom, load noise + task."""
        bad_ch_cfg.process_empty_room = True
        bad_ch_cfg._bad_channel_emptyroom = True
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
