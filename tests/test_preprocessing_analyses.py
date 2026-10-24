"""Tests for preprocessing analysis modules — is_enabled, class attributes, run logic.

These tests exercise the analysis classes *without* real BIDS data by testing
the non-I/O parts: is_enabled(), class constants, and (where feasible) the
core computation methods with synthetic data.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import mne
import pytest

from custom.preprocessing._io import get_ica_bids_path
from custom.preprocessing.bad_segments import BadSegmentsAnalysis
from custom.preprocessing.bad_channels import BadChannelsAnalysis
from custom.preprocessing.bad_epochs import BadEpochsAnalysis
from custom.preprocessing.manual_channel import ManualChannelAnalysis
from custom.preprocessing.apply_hfc import ApplyHFCAnalysis
from custom.preprocessing.zca_filter import ZCAFilterAnalysis
from custom.preprocessing.bad_ICs import BadICAnalysis
from custom.preprocessing.manual_ica import ManualICAAnalysis
from custom.preprocessing.coreg import CoregAnalysis
from custom.preprocessing.ica_convergence import ICAConvergenceAnalysis


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
            (ApplyHFCAnalysis, "applyhfc", "apply_hfc"),
            (ZCAFilterAnalysis, "zcafilter", "zca_filter"),
            (BadICAnalysis, "badICs", "bad_ICs"),
            (ManualICAAnalysis, "manualica", "manual_ica"),
            (CoregAnalysis, "coreg", "coreg"),
            (ICAConvergenceAnalysis, "icaconvergence", "ica_convergence"),
        ],
    )
    def test_keys_and_names(self, cls, expected_key, expected_name):
        assert cls.ANALYSIS_KEY == expected_key
        assert cls.ANALYSIS_NAME == expected_name


# ---------------------------------------------------------------------------
# is_enabled — always-on analyses
# ---------------------------------------------------------------------------


class TestAlwaysEnabled:
    """bad_segments, bad_channels, coreg are always enabled."""

    @pytest.mark.parametrize(
        "cls",
        [BadSegmentsAnalysis, BadChannelsAnalysis, CoregAnalysis],
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

    @pytest.mark.parametrize(
        "flags, expected",
        [({}, True), ({"_bad_epochs": True}, True), ({"_bad_epochs": False}, False)],
    )
    def test_bad_epochs_flag_defaults_on(self, flags, expected):
        cfg = SimpleNamespace(**flags)
        assert BadEpochsAnalysis(cfg).is_enabled() is expected


# ---------------------------------------------------------------------------
# is_enabled — ICA analyses (require spatial_filter='ica' + flag)
# ---------------------------------------------------------------------------


class TestICAEnabled:
    """bad_ICs and manual_ica require both flag AND spatial_filter='ica'."""

    def test_bad_ICs_both_set(self):
        cfg = SimpleNamespace(_bad_ICs=True, spatial_filter="ica")
        assert BadICAnalysis(cfg).is_enabled() is True

    def test_bad_ICs_flag_missing(self):
        cfg = SimpleNamespace(spatial_filter="ica")
        assert BadICAnalysis(cfg).is_enabled() is False

    def test_bad_ICs_spatial_filter_missing(self):
        cfg = SimpleNamespace(_bad_ICs=True)
        assert BadICAnalysis(cfg).is_enabled() is False

    def test_bad_ICs_spatial_filter_wrong(self):
        cfg = SimpleNamespace(_bad_ICs=True, spatial_filter="ssp")
        assert BadICAnalysis(cfg).is_enabled() is False

    def test_manual_ica_both_set(self):
        cfg = SimpleNamespace(_manual_ica=True, spatial_filter="ica")
        assert ManualICAAnalysis(cfg).is_enabled() is True

    def test_manual_ica_flag_missing(self):
        cfg = SimpleNamespace(spatial_filter="ica")
        assert ManualICAAnalysis(cfg).is_enabled() is False


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


# ---------------------------------------------------------------------------
# ICA convergence check
# ---------------------------------------------------------------------------


class TestICAConvergence:
    @pytest.fixture()
    def setup(self, tmp_path, meg_info, rng):
        # mixed Laplace sources, so that ICA can converge
        n_ch = len(meg_info["ch_names"])
        data = rng.randn(n_ch, n_ch) @ rng.laplace(size=(n_ch, 3000))
        raw_meg = mne.io.RawArray(data * 1e-13, meg_info, verbose=False)
        cfg = SimpleNamespace(
            deriv_root=tmp_path,
            subjects=["001"],
            sessions=["01"],
            task="TSX",
            spatial_filter="ica",
        )
        bp = get_ica_bids_path(cfg)
        bp.directory.mkdir(parents=True)

        def fit(max_iter):
            ica = mne.preprocessing.ICA(5, max_iter=max_iter, random_state=0)
            with pytest.warns(UserWarning) if max_iter < 50 else nullcontext():
                ica.fit(raw_meg, verbose=False)
            ica.save(bp.fpath, overwrite=True, verbose=False)

        return cfg, bp, fit

    def test_enabled_only_for_ica(self):
        assert ICAConvergenceAnalysis(
            SimpleNamespace(spatial_filter="ica")
        ).is_enabled()
        assert not ICAConvergenceAnalysis(
            SimpleNamespace(spatial_filter="ssp")
        ).is_enabled()

    def test_warns_and_plots_when_not_converged(self, setup):
        cfg, bp, fit = setup
        fit(max_iter=2)
        with pytest.warns(RuntimeWarning, match="did not converge after 2 of 2"):
            ICAConvergenceAnalysis(cfg).execute()
        assert (bp.directory / "ICA" / f"{bp.fpath.stem}_convergence.png").exists()
        cfg._ica_fail_on_nonconvergence = True
        with pytest.raises(RuntimeError, match="did not converge"):
            ICAConvergenceAnalysis(cfg).execute()

    def test_converged_is_silent(self, setup, recwarn):
        cfg, bp, fit = setup
        fit(max_iter=1000)
        ICAConvergenceAnalysis(cfg).execute()
        assert not [w for w in recwarn if "did not converge" in str(w.message)]
        assert (bp.directory / "ICA" / f"{bp.fpath.stem}_convergence.png").exists()
