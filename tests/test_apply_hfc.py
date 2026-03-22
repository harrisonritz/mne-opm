"""Tests for apply_hfc.py — Homogeneous Field Correction.

Exercises the core HFC logic (_apply_hfc) which computes and applies
SSP projections, load_data/save_results with mocked BIDS I/O, and the
module-level run(cfg) entry point.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.apply_hfc import ApplyHFCAnalysis, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hfc_cfg(tmp_path):
    """Config for HFC analysis."""
    return SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        _do_HFC=True,
        _hfc_order=1,
        process_empty_room=False,
        find_breaks=False,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# _apply_hfc — core logic
# ---------------------------------------------------------------------------

class TestApplyHFC:
    """Test the core HFC projection computation and application."""

    def test_applies_projections_to_raw(self, raw_meg, hfc_cfg):
        """HFC should add and apply SSP projections to raw data."""
        analysis = ApplyHFCAnalysis(hfc_cfg)
        raw_before = raw_meg.copy()
        raw_out, noise_out = analysis._apply_hfc(raw_meg)

        # Projections should be applied
        assert len(raw_out.info["projs"]) > 0
        assert all(p["active"] for p in raw_out.info["projs"])
        assert noise_out is None

    def test_hfc_reduces_homogeneous_field(self, meg_info, hfc_cfg):
        """HFC should remove a spatially uniform field component."""
        rng = np.random.RandomState(42)
        n_ch = len(meg_info["ch_names"])
        n_samples = int(meg_info["sfreq"] * 10)

        # Create data with a large homogeneous field (same signal on all MEG)
        brain = rng.randn(n_ch, n_samples) * 1e-14
        n_meg = sum(1 for t in meg_info.get_channel_types() if t == "mag")
        homogeneous = rng.randn(1, n_samples) * 1e-12  # 100x brain
        brain[:n_meg, :] += homogeneous  # uniform field on all MEG channels
        raw = mne.io.RawArray(brain, meg_info)

        var_before = np.var(raw.get_data(picks="mag"))
        analysis = ApplyHFCAnalysis(hfc_cfg)
        raw_out, _ = analysis._apply_hfc(raw)
        var_after = np.var(raw_out.get_data(picks="mag"))

        # Removing the homogeneous field should reduce variance
        assert var_after < var_before, (
            "HFC should reduce variance by removing the homogeneous field"
        )

    def test_applies_same_projs_to_noise(self, raw_meg, meg_info, rng, hfc_cfg):
        """When noise is provided, same projections applied to both."""
        n_ch = len(meg_info["ch_names"])
        n_samples = int(meg_info["sfreq"] * 5)
        noise = mne.io.RawArray(
            rng.randn(n_ch, n_samples) * 1e-13, meg_info
        )

        analysis = ApplyHFCAnalysis(hfc_cfg)
        raw_out, noise_out = analysis._apply_hfc(raw_meg, noise)

        assert noise_out is not None
        # Both should have the same number of projections
        assert len(raw_out.info["projs"]) == len(noise_out.info["projs"])
        # Both should have active projections
        assert all(p["active"] for p in noise_out.info["projs"])

    def test_hfc_order_2(self, raw_meg, hfc_cfg):
        """Higher order HFC should produce more projections."""
        hfc_cfg._hfc_order = 1
        analysis = ApplyHFCAnalysis(hfc_cfg)
        raw1 = raw_meg.copy()
        raw1_out, _ = analysis._apply_hfc(raw1)
        n_projs_order1 = len(raw1_out.info["projs"])

        hfc_cfg._hfc_order = 2
        analysis2 = ApplyHFCAnalysis(hfc_cfg)
        raw2 = raw_meg.copy()
        raw2_out, _ = analysis2._apply_hfc(raw2)
        n_projs_order2 = len(raw2_out.info["projs"])

        assert n_projs_order2 > n_projs_order1, (
            "Higher HFC order should produce more projections"
        )


# ---------------------------------------------------------------------------
# run() method
# ---------------------------------------------------------------------------

class TestHFCRun:
    def test_run_returns_corrected_data(self, raw_meg, hfc_cfg):
        analysis = ApplyHFCAnalysis(hfc_cfg)
        data = {hfc_cfg.task: raw_meg}
        results = analysis.run(data)

        assert hfc_cfg.task in results
        assert len(results[hfc_cfg.task].info["projs"]) > 0

    def test_run_with_noise(self, raw_meg, meg_info, rng, hfc_cfg):
        hfc_cfg.process_empty_room = True
        n_ch = len(meg_info["ch_names"])
        n_samples = int(meg_info["sfreq"] * 5)
        noise = mne.io.RawArray(
            rng.randn(n_ch, n_samples) * 1e-13, meg_info
        )

        analysis = ApplyHFCAnalysis(hfc_cfg)
        data = {hfc_cfg.task: raw_meg, "noise": noise}
        results = analysis.run(data)

        assert "noise" in results


# ---------------------------------------------------------------------------
# load_data — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestHFCLoadData:
    @patch("custom.preprocessing.apply_hfc.read_raw_bids_with_retry")
    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_load_task_data(self, mock_bids, mock_read, raw_meg, hfc_cfg):
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = ApplyHFCAnalysis(hfc_cfg)
        data = analysis.load_data()

        assert hfc_cfg.task in data
        mock_read.assert_called_once()

    @patch("custom.preprocessing.apply_hfc.read_raw_bids_with_retry")
    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_load_with_noise(self, mock_bids, mock_read, raw_meg, hfc_cfg):
        hfc_cfg.process_empty_room = True
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        analysis = ApplyHFCAnalysis(hfc_cfg)
        data = analysis.load_data()

        assert "noise" in data
        assert hfc_cfg.task in data

    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_load_no_files_raises(self, mock_bids, hfc_cfg):
        mock_bids.find_matching_paths.return_value = []
        analysis = ApplyHFCAnalysis(hfc_cfg)
        with pytest.raises(FileNotFoundError):
            analysis.load_data()


# ---------------------------------------------------------------------------
# save_results — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestHFCSaveResults:
    @patch("custom.preprocessing.apply_hfc.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_save_writes_data(self, mock_bids, mock_write, raw_meg, hfc_cfg):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = ApplyHFCAnalysis(hfc_cfg)
        results = {hfc_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        mock_write.assert_called_once()

    @patch("custom.preprocessing.apply_hfc.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_save_with_empty_room(self, mock_bids, mock_write, raw_meg, hfc_cfg):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = ApplyHFCAnalysis(hfc_cfg)
        results = {"noise": raw_meg, hfc_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        assert mock_write.call_count == 2


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------

class TestHFCEnabled:
    def test_disabled_by_default(self):
        cfg = SimpleNamespace()
        assert ApplyHFCAnalysis(cfg).is_enabled() is False

    def test_disabled_exits_early(self, capsys):
        """run(cfg) with _do_HFC=False should print disabled message."""
        cfg = SimpleNamespace(_do_HFC=False)
        run(cfg)
        captured = capsys.readouterr()
        assert "Disabled" in captured.out


# ---------------------------------------------------------------------------
# Module-level run(cfg)
# ---------------------------------------------------------------------------

class TestHFCModuleRun:
    @patch("custom.preprocessing.apply_hfc.write_raw_bids_preserve_events")
    @patch("custom.preprocessing.apply_hfc.read_raw_bids_with_retry")
    @patch("custom.preprocessing.apply_hfc.mne_bids")
    def test_run_end_to_end(self, mock_bids, mock_read, mock_write, raw_meg, hfc_cfg):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_read.return_value = raw_meg

        run(hfc_cfg)

        mock_read.assert_called()
        mock_write.assert_called()
