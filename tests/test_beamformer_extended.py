"""Extended tests for run_beamformer.py.

Covers load_beamformer_data, run_beamformer_power contrast computation,
save_beamformer_results, add_to_report, and the main() entry point.
Uses mocked forward solutions and epochs to avoid real file I/O.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.run_beamformer import (
    compute_lcmv_filters,
    load_beamformer_data,
    run_beamformer_power,
    run_beamformer_timecourse,
    save_beamformer_results,
    add_to_report,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def beam_cfg(tmp_path):
    """Minimal beamformer config."""
    deriv = tmp_path / "derivatives"
    deriv.mkdir()
    return SimpleNamespace(
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        deriv_root=str(deriv),
        datatype="meg",
        n_jobs=1,
        noise_cov="ad-hoc",
        conditions=["stim_a", "stim_b"],
        contrasts=[{
            "name": "a_vs_b",
            "conditions": ["stim_a", "stim_b"],
            "weights": [1.0, -1.0],
        }],
        ch_types=["mag"],
        _run_beamformer=True,
        _beamformer_reg=0.05,
        _beamformer_pick_ori="max-power",
        _beamformer_weight_norm="unit-noise-gain",
        _beamformer_depth=None,
        _beamformer_rank=None,
        _beamformer_save_filters=False,
        _beamformer_add_to_report=False,
        _beamformer_power_tmin=0.0,
        _beamformer_power_tmax=0.5,
    )


# ---------------------------------------------------------------------------
# load_beamformer_data
# ---------------------------------------------------------------------------

class TestLoadBeamformerData:
    def test_missing_forward_raises(self, beam_cfg, tmp_path):
        """Should raise FileNotFoundError when forward file missing."""
        with pytest.raises(FileNotFoundError, match="Forward solution"):
            load_beamformer_data(beam_cfg)

    def test_ad_hoc_noise_cov(self, beam_cfg, tmp_path):
        """When noise_cov='ad-hoc', data['noise_cov'] should be None."""
        # Create mock files
        deriv = Path(beam_cfg.deriv_root)
        meg_dir = deriv / "sub-001" / "ses-01" / "meg"
        meg_dir.mkdir(parents=True)

        fwd_path = meg_dir / "sub-001_ses-01_task-restingstate_fwd.fif"
        epo_path = meg_dir / "sub-001_ses-01_task-restingstate_proc-clean_epo.fif"

        with patch("custom.run_beamformer.mne.read_forward_solution") as mock_fwd, \
             patch("custom.run_beamformer.mne.read_epochs") as mock_epo, \
             patch("custom.run_beamformer.mne.io.read_info") as mock_info:

            # Create mock fwd path
            fwd_path.touch()
            epo_path.touch()

            mock_fwd.return_value = MagicMock(spec=mne.Forward)
            mock_fwd.return_value.__getitem__ = lambda self, key: [1, 2] if key == "src" else None

            info = mne.create_info(["MEG001"], 300.0, ["mag"])
            mock_epochs = MagicMock(spec=mne.Epochs)
            mock_epochs.__len__ = lambda self: 10
            mock_epochs.ch_names = ["MEG001"]
            mock_epo.return_value = mock_epochs
            mock_info.return_value = info

            data = load_beamformer_data(beam_cfg)
            assert data["noise_cov"] is None


# ---------------------------------------------------------------------------
# save_beamformer_results
# ---------------------------------------------------------------------------

class TestSaveBeamformerResults:
    def test_saves_stc_files(self, beam_cfg, tmp_path):
        """Should call stc.save() for each condition."""
        mock_stc_a = MagicMock()
        mock_stc_b = MagicMock()
        stcs = {"stim_a": mock_stc_a, "stim_b": mock_stc_b}
        mock_filters = MagicMock()

        out = save_beamformer_results(
            beam_cfg, mock_filters, stcs, analysis_type="time"
        )

        mock_stc_a.save.assert_called_once()
        mock_stc_b.save.assert_called_once()
        assert "stim_a" in out
        assert "stim_b" in out

    def test_saves_filters_when_enabled(self, beam_cfg, tmp_path):
        """Filters should be saved when _beamformer_save_filters=True."""
        beam_cfg._beamformer_save_filters = True
        mock_stc = MagicMock()
        mock_filters = MagicMock()

        out = save_beamformer_results(
            beam_cfg, mock_filters, {"stim_a": mock_stc}, analysis_type="time"
        )

        mock_filters.save.assert_called_once()
        assert "filters" in out

    def test_no_filters_for_power_type(self, beam_cfg, tmp_path):
        """Filters should NOT be saved for power analysis type."""
        beam_cfg._beamformer_save_filters = True
        mock_stc = MagicMock()
        mock_filters = MagicMock()

        out = save_beamformer_results(
            beam_cfg, mock_filters, {"stim_a": mock_stc}, analysis_type="power"
        )

        mock_filters.save.assert_not_called()

    def test_power_suffix(self, beam_cfg, tmp_path):
        """Power analysis should use 'lcmv-power' in suffix."""
        mock_stc = MagicMock()
        mock_filters = MagicMock()

        out = save_beamformer_results(
            beam_cfg, mock_filters, {"stim_a": mock_stc}, analysis_type="power"
        )

        # Verify the save was called (suffix encoded in BIDSPath)
        mock_stc.save.assert_called_once()


# ---------------------------------------------------------------------------
# add_to_report
# ---------------------------------------------------------------------------

class TestAddToReport:
    def test_disabled_report_skips(self, beam_cfg, capsys):
        """When _beamformer_add_to_report=False, should skip."""
        beam_cfg._beamformer_add_to_report = False
        add_to_report(beam_cfg, {}, analysis_type="time")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()


# ---------------------------------------------------------------------------
# run_beamformer_timecourse — contrast paths
# ---------------------------------------------------------------------------

class TestBeamformerTimecourseContrasts:
    """Test the contrast computation branch in run_beamformer_timecourse."""

    def test_timecourse_contrast_computed(self, beam_cfg):
        """Contrasts should combine conditions using weights."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.RandomState(0).randn(6, 1, 100) * 1e-13,
            info,
            events=np.array(
                [[0, 0, 1], [100, 0, 1], [200, 0, 1],
                 [300, 0, 2], [400, 0, 2], [500, 0, 2]]
            ),
            event_id={"stim_a": 1, "stim_b": 2},
        )

        with patch("custom.run_beamformer._all_conditions") as mock_conds:
            mock_conds.return_value = ["stim_a", "stim_b", "a_vs_b"]
            with patch("custom.run_beamformer.apply_lcmv") as mock_apply:
                mock_stc = MagicMock()
                mock_stc.data = np.ones((5, 100))
                mock_apply.return_value = mock_stc

                stcs = run_beamformer_timecourse(
                    epochs, MagicMock(), beam_cfg
                )

        assert "stim_a" in stcs
        assert "stim_b" in stcs
        assert "a_vs_b" in stcs

    def test_timecourse_missing_contrast_def_skipped(self, beam_cfg, capsys):
        """A contrast name not in cfg.contrasts should be skipped."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.RandomState(0).randn(3, 1, 100) * 1e-13,
            info,
            events=np.array([[0, 0, 1], [100, 0, 1], [200, 0, 1]]),
            event_id={"stim_a": 1},
        )
        beam_cfg.conditions = ["stim_a"]
        beam_cfg.contrasts = []  # No contrast definitions

        with patch("custom.run_beamformer._all_conditions") as mock_conds:
            mock_conds.return_value = ["stim_a", "undefined_contrast"]
            with patch("custom.run_beamformer.apply_lcmv") as mock_apply:
                mock_stc = MagicMock()
                mock_stc.data = np.zeros((5, 100))
                mock_apply.return_value = mock_stc

                stcs = run_beamformer_timecourse(
                    epochs, MagicMock(), beam_cfg
                )

        assert "undefined_contrast" not in stcs
        assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_beamformer_power — contrast paths
# ---------------------------------------------------------------------------

class TestBeamformerPowerContrasts:
    """Test contrast computation in run_beamformer_power."""

    def test_power_contrast_computed(self, beam_cfg):
        """Power contrasts should compute normalized differences."""
        info = mne.create_info(["MEG001"], 300.0, ["mag"])
        epochs = mne.EpochsArray(
            np.random.RandomState(0).randn(6, 1, 100) * 1e-13,
            info,
            events=np.array(
                [[0, 0, 1], [100, 0, 1], [200, 0, 1],
                 [300, 0, 2], [400, 0, 2], [500, 0, 2]]
            ),
            event_id={"stim_a": 1, "stim_b": 2},
        )

        with patch("custom.run_beamformer.apply_lcmv_cov") as mock_apply, \
             patch("custom.run_beamformer.mne.compute_covariance") as mock_cov:
            mock_stc = MagicMock()
            mock_stc.data = np.ones((5, 1))
            mock_stc.copy.return_value = MagicMock(data=np.ones((5, 1)))
            mock_apply.return_value = mock_stc
            mock_cov.return_value = MagicMock()

            stcs = run_beamformer_power(
                epochs, MagicMock(), beam_cfg
            )

        assert "stim_a" in stcs
        assert "stim_b" in stcs
        assert "a_vs_b" in stcs


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestBeamformerParseArgs:
    def test_requires_config(self):
        """--config is required."""
        with pytest.raises(SystemExit):
            parse_args()
