"""Tests for bad_epochs.py — GESD bad epoch detection.

Exercises the core detection logic (_drop_bad_epochs), load_data,
save_results, and module-level run(cfg). Epochs file I/O is mocked
but the GESD detection runs on real synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_epochs import BadEpochsAnalysis, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bad_epo_cfg(tmp_path):
    """Config appropriate for bad epoch detection."""
    return SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        n_jobs=1,
    )


@pytest.fixture()
def epochs_with_bad_trial(meg_info):
    """Create 20 epochs where the last one has 100x amplitude (outlier)."""
    rng = np.random.RandomState(99)
    sfreq = meg_info["sfreq"]
    n_ch = len(meg_info["ch_names"])
    n_samples = int(sfreq * 50)  # 50 seconds

    data = rng.randn(n_ch, n_samples) * 1e-13
    raw = mne.io.RawArray(data, meg_info)

    events = np.array([[int(i * 2 * sfreq), 0, 1] for i in range(20)])
    epochs = mne.Epochs(
        raw, events, event_id={"stim": 1},
        tmin=0, tmax=1.0 - 1 / sfreq,
        baseline=None, preload=True,
    )

    # Inject a large artifact into the last epoch
    n_meg = sum(1 for t in meg_info.get_channel_types() if t == "mag")
    epochs._data[-1, :n_meg, :] *= 100.0

    return epochs


# ---------------------------------------------------------------------------
# _drop_bad_epochs — core GESD logic
# ---------------------------------------------------------------------------

class TestDropBadEpochs:
    """Test the core epoch rejection method."""

    def test_drops_outlier_epoch(self, epochs_with_bad_trial, bad_epo_cfg):
        """The 100x amplitude epoch should be dropped."""
        analysis = BadEpochsAnalysis(bad_epo_cfg)
        n_before = len(epochs_with_bad_trial)
        clean = analysis._drop_bad_epochs(epochs_with_bad_trial)
        n_after = len(clean)

        assert n_after < n_before, "GESD should drop the outlier epoch"
        assert isinstance(clean, mne.Epochs)

    def test_preserves_clean_epochs(self, epochs_meg, bad_epo_cfg):
        """Homogeneous epochs should lose few or none."""
        analysis = BadEpochsAnalysis(bad_epo_cfg)
        n_before = len(epochs_meg)
        clean = analysis._drop_bad_epochs(epochs_meg)
        n_after = len(clean)

        # With iid Gaussian data, GESD should drop 0 or very few
        assert n_after >= n_before - 1


# ---------------------------------------------------------------------------
# run() method
# ---------------------------------------------------------------------------

class TestBadEpochsRun:
    """Test run() method."""

    def test_run_returns_cleaned_epochs(
        self, epochs_with_bad_trial, bad_epo_cfg
    ):
        n_before = len(epochs_with_bad_trial)
        analysis = BadEpochsAnalysis(bad_epo_cfg)
        data = {bad_epo_cfg.task: epochs_with_bad_trial}
        results = analysis.run(data)

        assert bad_epo_cfg.task in results
        assert isinstance(results[bad_epo_cfg.task], mne.Epochs)
        assert len(results[bad_epo_cfg.task]) < n_before


# ---------------------------------------------------------------------------
# load_data — mocked file I/O
# ---------------------------------------------------------------------------

class TestBadEpochsLoadData:
    @patch("custom.preprocessing.bad_epochs.mne.read_epochs")
    def test_load_constructs_bids_path(
        self, mock_read, epochs_meg, bad_epo_cfg
    ):
        """load_data should construct a BIDSPath and call mne.read_epochs."""
        mock_read.return_value = epochs_meg

        analysis = BadEpochsAnalysis(bad_epo_cfg)
        data = analysis.load_data()

        assert bad_epo_cfg.task in data
        assert data[bad_epo_cfg.task] is epochs_meg
        mock_read.assert_called_once()


# ---------------------------------------------------------------------------
# save_results — mocked file I/O
# ---------------------------------------------------------------------------

class TestBadEpochsSaveResults:
    def test_save_calls_epochs_save(self, epochs_meg, bad_epo_cfg):
        """save_results should call epochs.save() with the BIDS path."""
        analysis = BadEpochsAnalysis(bad_epo_cfg)

        mock_epochs = MagicMock(spec=mne.Epochs)
        results = {bad_epo_cfg.task: mock_epochs}
        analysis.save_results(results)

        mock_epochs.save.assert_called_once()
        call_kwargs = mock_epochs.save.call_args[1]
        assert call_kwargs["overwrite"] is True


# ---------------------------------------------------------------------------
# Module-level run(cfg)
# ---------------------------------------------------------------------------

class TestBadEpochsModuleRun:
    @patch("custom.preprocessing.bad_epochs.mne.read_epochs")
    def test_run_entry_point(self, mock_read, epochs_with_bad_trial, bad_epo_cfg):
        mock_read.return_value = epochs_with_bad_trial

        # Mock epochs.save so it doesn't try to write to disk
        with patch.object(type(epochs_with_bad_trial), "save", autospec=True):
            run(bad_epo_cfg)

        mock_read.assert_called_once()
