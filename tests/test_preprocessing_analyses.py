"""Tests for preprocessing analysis modules — is_enabled, class attributes, run logic.

These tests exercise the analysis classes *without* real BIDS data by testing
the non-I/O parts: is_enabled(), class constants, and (where feasible) the
core computation methods with synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_segments import BadSegmentsAnalysis
from custom.preprocessing.bad_channels import BadChannelsAnalysis
from custom.preprocessing.bad_epochs import BadEpochsAnalysis
from custom.preprocessing.manual_channel import ManualChannelAnalysis
from custom.preprocessing.regress import RegressAnalysis
from custom.preprocessing.apply_hfc import ApplyHFCAnalysis
from custom.preprocessing.zca_filter import ZCAFilterAnalysis
from custom.preprocessing.bad_ICs import BadICAnalysis
from custom.preprocessing.manual_ica import ManualICAAnalysis
from custom.preprocessing.coreg import CoregAnalysis


# ---------------------------------------------------------------------------
# Class constants
# ---------------------------------------------------------------------------


class TestAnalysisClassConstants:
    """Verify ANALYSIS_KEY and ANALYSIS_NAME on every analysis class."""

    @pytest.mark.parametrize(
        "cls, expected_key, expected_name",
        [
            (BadSegmentsAnalysis, "badsegments", "bad_segments"),
            (BadChannelsAnalysis, "badchannels", "bad_channels"),
            (BadEpochsAnalysis, "badepochs", "bad_epochs"),
            (ManualChannelAnalysis, "manualchannel", "manual_channel"),
            (RegressAnalysis, "regress", "regress"),
            (ApplyHFCAnalysis, "applyhfc", "apply_hfc"),
            (ZCAFilterAnalysis, "zcafilter", "zca_filter"),
            (BadICAnalysis, "autoica", "auto_ica"),
            (ManualICAAnalysis, "manualica", "manual_ica"),
            (CoregAnalysis, "coreg", "coreg"),
        ],
    )
    def test_keys_and_names(self, cls, expected_key, expected_name):
        assert cls.ANALYSIS_KEY == expected_key
        assert cls.ANALYSIS_NAME == expected_name


# ---------------------------------------------------------------------------
# is_enabled — always-on analyses
# ---------------------------------------------------------------------------


class TestAlwaysEnabled:
    """bad_segments, bad_channels, bad_epochs, coreg are always enabled."""

    @pytest.mark.parametrize(
        "cls",
        [BadSegmentsAnalysis, BadChannelsAnalysis, BadEpochsAnalysis, CoregAnalysis],
    )
    def test_always_enabled(self, cls):
        cfg = SimpleNamespace()
        analysis = cls(cfg)
        assert analysis.is_enabled() is True


# ---------------------------------------------------------------------------
# is_enabled — flag-gated analyses
# ---------------------------------------------------------------------------


class TestFlagGatedEnabled:
    """Analyses gated by a config flag."""

    def test_regress_enabled(self):
        cfg = SimpleNamespace(_regress=True)
        assert RegressAnalysis(cfg).is_enabled() is True

    def test_regress_disabled(self):
        cfg = SimpleNamespace(_regress=False)
        assert RegressAnalysis(cfg).is_enabled() is False

    def test_regress_missing_defaults_false(self):
        cfg = SimpleNamespace()
        assert RegressAnalysis(cfg).is_enabled() is False

    def test_hfc_enabled(self):
        cfg = SimpleNamespace(_do_HFC=True)
        assert ApplyHFCAnalysis(cfg).is_enabled() is True

    def test_hfc_disabled(self):
        cfg = SimpleNamespace(_do_HFC=False)
        assert ApplyHFCAnalysis(cfg).is_enabled() is False

    def test_zca_enabled(self):
        cfg = SimpleNamespace(_do_ZCA=True)
        assert ZCAFilterAnalysis(cfg).is_enabled() is True

    def test_zca_disabled(self):
        cfg = SimpleNamespace()
        assert ZCAFilterAnalysis(cfg).is_enabled() is False

    def test_manual_channel_enabled(self):
        cfg = SimpleNamespace(_manual_channels=True)
        assert ManualChannelAnalysis(cfg).is_enabled() is True

    def test_manual_channel_disabled(self):
        cfg = SimpleNamespace(_manual_channels=False)
        assert ManualChannelAnalysis(cfg).is_enabled() is False


# ---------------------------------------------------------------------------
# is_enabled — ICA analyses (require spatial_filter='ica' + flag)
# ---------------------------------------------------------------------------


class TestICAEnabled:
    """auto_ica and manual_ica require both flag AND spatial_filter='ica'."""

    def test_auto_ica_both_set(self):
        cfg = SimpleNamespace(_auto_ica=True, spatial_filter="ica")
        assert AutoICAAnalysis(cfg).is_enabled() is True

    def test_auto_ica_flag_missing(self):
        cfg = SimpleNamespace(spatial_filter="ica")
        assert AutoICAAnalysis(cfg).is_enabled() is False

    def test_auto_ica_spatial_filter_missing(self):
        cfg = SimpleNamespace(_auto_ica=True)
        assert AutoICAAnalysis(cfg).is_enabled() is False

    def test_auto_ica_spatial_filter_wrong(self):
        cfg = SimpleNamespace(_auto_ica=True, spatial_filter="ssp")
        assert AutoICAAnalysis(cfg).is_enabled() is False

    def test_manual_ica_both_set(self):
        cfg = SimpleNamespace(_manual_ica=True, spatial_filter="ica")
        assert ManualICAAnalysis(cfg).is_enabled() is True

    def test_manual_ica_flag_missing(self):
        cfg = SimpleNamespace(spatial_filter="ica")
        assert ManualICAAnalysis(cfg).is_enabled() is False


# ---------------------------------------------------------------------------
# RegressAnalysis — core regression logic (time-varying)
# ---------------------------------------------------------------------------


class TestRegressCore:
    """Test the time-varying sliding-window regression on synthetic data."""

    @pytest.fixture()
    def regression_setup(self):
        """Create raw data where ref channels are correlated with MEG."""
        rng = np.random.RandomState(42)
        sfreq = 300.0
        n_samples = int(sfreq * 10)  # 10 seconds
        n_meg = 10
        n_ref = 3

        # Reference signals
        ref_signals = rng.randn(n_ref, n_samples) * 1e-12

        # MEG = brain signal + linear mixing of ref (interference)
        mixing = rng.randn(n_meg, n_ref)
        brain = rng.randn(n_meg, n_samples) * 1e-14  # weaker brain signal
        meg_data = brain + mixing @ ref_signals

        info = mne.create_info(
            ch_names=[f"MEG{i:03d}" for i in range(n_meg)]
            + [f"REF{i:03d}" for i in range(n_ref)],
            sfreq=sfreq,
            ch_types=["mag"] * n_meg + ["ref_meg"] * n_ref,
        )
        data = np.vstack([meg_data, ref_signals])
        raw = mne.io.RawArray(data, info)
        return raw, brain

    def test_prepare_predictor_data_shape(self, regression_setup):
        raw, _ = regression_setup
        cfg = SimpleNamespace(
            _regress=True,
            _regress_preds=["ref_meg"],
            ch_types=["mag"],
            _regress_freqs=None,
        )
        analysis = RegressAnalysis(cfg)
        pred_data = analysis._prepare_predictor_data(raw)
        # Without freq bands: raw + squared = 2 * n_ref features
        assert pred_data.shape[0] == 6  # 3 ref * 2 (raw + squared)
        assert pred_data.shape[1] == raw.n_times

    def test_prepare_predictor_data_with_freqs(self, regression_setup):
        raw, _ = regression_setup
        cfg = SimpleNamespace(
            _regress=True,
            _regress_preds=["ref_meg"],
            ch_types=["mag"],
            _regress_freqs=[(None, 10.0), (10.0, 50.0)],
        )
        analysis = RegressAnalysis(cfg)
        pred_data = analysis._prepare_predictor_data(raw)
        # 2 freq bands * 3 ref channels = 6 features
        assert pred_data.shape[0] == 6
        assert pred_data.shape[1] == raw.n_times

    def test_standard_regression_reduces_variance(self, regression_setup):
        """Standard regression should reduce MEG variance."""
        raw, _ = regression_setup
        cfg = SimpleNamespace(
            _regress=True,
            _regress_preds=["ref_meg"],
            _regress_timevarying=False,
            _regress_plot=False,
            ch_types=["mag"],
            find_breaks=False,
        )
        analysis = RegressAnalysis(cfg)

        var_before = np.var(raw.get_data(picks="mag"))
        raw_clean = analysis._regress(raw.copy())
        var_after = np.var(raw_clean.get_data(picks="mag"))

        assert var_after < var_before, (
            "Variance should decrease after regressing out reference signals"
        )

    def test_timevarying_regression_reduces_variance(self, regression_setup):
        """Time-varying regression should reduce MEG variance."""
        raw, _ = regression_setup
        cfg = SimpleNamespace(
            _regress=True,
            _regress_preds=["ref_meg"],
            _regress_timevarying=True,
            _regress_window=3.0,  # short windows for 10s data
            _regress_freqs=None,
            _regress_plot=False,
            ch_types=["mag"],
            find_breaks=False,
        )
        analysis = RegressAnalysis(cfg)

        var_before = np.var(raw.get_data(picks="mag"))
        raw_clean = analysis._regress(raw.copy())
        var_after = np.var(raw_clean.get_data(picks="mag"))

        assert var_after < var_before


# ---------------------------------------------------------------------------
# ZCA / Coreg — FreeSurfer subject name and BEM/source space helpers
# ---------------------------------------------------------------------------


class TestZCAAndCoregHelpers:
    """ZCAFilterAnalysis has _get_fs_subject, _find_bem_solution, _find_source_space."""

    def test_get_fs_subject_with_session(self):
        cfg = SimpleNamespace(subjects=["001"], sessions=["02"])
        analysis = ZCAFilterAnalysis(cfg)
        assert analysis._get_fs_subject() == "sub-001_ses-02"

    def test_get_fs_subject_without_session(self):
        cfg = SimpleNamespace(subjects=["001"], sessions=[])
        analysis = ZCAFilterAnalysis(cfg)
        assert analysis._get_fs_subject() == "sub-001"

    def test_find_bem_missing_dir(self, tmp_path):
        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        with pytest.raises(FileNotFoundError, match="BEM directory"):
            analysis._find_bem_solution("sub-001_ses-01")

    def test_find_source_space_missing_dir(self, tmp_path):
        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        with pytest.raises(FileNotFoundError, match="BEM directory"):
            analysis._find_source_space("sub-001_ses-01")

    def test_find_bem_no_files(self, tmp_path):
        bem_dir = tmp_path / "sub-001_ses-01" / "bem"
        bem_dir.mkdir(parents=True)
        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        with pytest.raises(FileNotFoundError, match="No BEM solution"):
            analysis._find_bem_solution("sub-001_ses-01")

    def test_find_source_space_no_files(self, tmp_path):
        bem_dir = tmp_path / "sub-001_ses-01" / "bem"
        bem_dir.mkdir(parents=True)
        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        with pytest.raises(FileNotFoundError, match="No source space"):
            analysis._find_source_space("sub-001_ses-01")

    def test_find_bem_returns_path(self, tmp_path):
        fs_subj = "sub-001_ses-01"
        bem_dir = tmp_path / fs_subj / "bem"
        bem_dir.mkdir(parents=True)
        bem_file = bem_dir / f"{fs_subj}-5120-5120-5120-bem-sol.fif"
        bem_file.touch()

        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        result = analysis._find_bem_solution(fs_subj)
        assert result == bem_file

    def test_find_source_space_returns_path(self, tmp_path):
        fs_subj = "sub-001_ses-01"
        bem_dir = tmp_path / fs_subj / "bem"
        bem_dir.mkdir(parents=True)
        src_file = bem_dir / f"{fs_subj}-oct6-src.fif"
        src_file.touch()

        cfg = SimpleNamespace(
            subjects=["001"],
            sessions=["01"],
            subjects_dir=str(tmp_path),
        )
        analysis = ZCAFilterAnalysis(cfg)
        result = analysis._find_source_space(fs_subj)
        assert result == src_file

    def test_coreg_is_always_enabled(self):
        cfg = SimpleNamespace()
        assert CoregAnalysis(cfg).is_enabled() is True


# ---------------------------------------------------------------------------
# ZCA — helper methods
# ---------------------------------------------------------------------------


class TestZCAHelpers:
    def test_get_fs_subject_with_session(self):
        cfg = SimpleNamespace(subjects=["002"], sessions=["03"])
        analysis = ZCAFilterAnalysis(cfg)
        assert analysis._get_fs_subject() == "sub-002_ses-03"

    def test_get_fs_subject_no_session(self):
        cfg = SimpleNamespace(subjects=["002"], sessions=[])
        analysis = ZCAFilterAnalysis(cfg)
        assert analysis._get_fs_subject() == "sub-002"
