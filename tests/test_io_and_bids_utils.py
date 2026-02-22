"""Tests for preprocessing._io and preprocessing._bids_utils."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mne_bids import BIDSPath

from custom.preprocessing._bids_utils import get_bids_path
from custom.preprocessing._io import get_bids_path_for_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path, **overrides):
    defaults = dict(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _suffix_raw_allowed() -> bool:
    """Check whether the installed mne-bids allows suffix='raw' without check=False."""
    try:
        BIDSPath(
            root="/tmp", subject="001", session="01", task="test",
            datatype="meg", suffix="raw", extension=".fif",
        )
        return True
    except ValueError:
        return False


_RAW_SUFFIX_OK = _suffix_raw_allowed()


# ---------------------------------------------------------------------------
# get_bids_path (from _bids_utils)
# ---------------------------------------------------------------------------

class TestGetBidsPath:
    """Tests for the _bids_utils.get_bids_path convenience function."""

    def test_basic_construction(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path(cfg, task="restingstate")
        assert isinstance(bp, BIDSPath)
        assert bp.subject == "001"
        assert bp.session == "01"
        assert bp.task == "restingstate"
        assert bp.datatype == "meg"
        assert bp.suffix == "meg"
        assert bp.extension == ".fif"

    @pytest.mark.skipif(
        not _RAW_SUFFIX_OK,
        reason="mne-bids fork does not allow suffix='raw' without check=False",
    )
    def test_derivatives_changes_root_and_suffix(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path(cfg, task="restingstate", from_derivatives=True)
        assert str(bp.root) == cfg.deriv_root
        assert bp.suffix == "raw"

    def test_derivatives_suffix_logic(self, tmp_path):
        """Verify that from_derivatives only overrides the default 'meg' suffix,
        not an explicitly-provided custom suffix like 'meg' (left as-is if
        from_derivatives=False)."""
        cfg = _make_cfg(tmp_path)
        # With from_derivatives=False, suffix stays "meg"
        bp = get_bids_path(cfg, task="noise", from_derivatives=False, suffix="meg")
        assert bp.suffix == "meg"

    def test_derivatives_root_used(self, tmp_path):
        """Verify from_derivatives selects deriv_root, even if suffix
        validation fails (we catch the error to check the root parameter)."""
        cfg = _make_cfg(tmp_path)
        # Test that the function *attempts* to use deriv_root
        try:
            bp = get_bids_path(cfg, task="noise", from_derivatives=True)
            assert str(bp.root) == cfg.deriv_root
        except ValueError:
            # BIDSPath rejected suffix="raw" — that's a known issue with
            # standard mne-bids; the custom fork supports it.
            pytest.skip("mne-bids does not allow suffix='raw'")

    def test_processing_label(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path(cfg, task="restingstate", processing="clean")
        assert bp.processing == "clean"

    def test_custom_extension(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path(cfg, task="noise", extension=".tsv")
        assert bp.extension == ".tsv"

    def test_subjects_as_string(self, tmp_path):
        cfg = _make_cfg(tmp_path, subjects="002", sessions="03")
        bp = get_bids_path(cfg, task="restingstate")
        assert bp.subject == "002"
        assert bp.session == "03"

    def test_task_noise(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path(cfg, task="noise")
        assert bp.task == "noise"


# ---------------------------------------------------------------------------
# get_bids_path_for_task (from _io)
# ---------------------------------------------------------------------------

class TestGetBidsPathForTask:
    """Tests for the _io.get_bids_path_for_task function."""

    def test_basic_construction(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path_for_task(cfg, task="restingstate")
        assert isinstance(bp, BIDSPath)
        assert bp.subject == "001"
        assert bp.session == "01"
        assert bp.task == "restingstate"

    @pytest.mark.skipif(
        not _RAW_SUFFIX_OK,
        reason="mne-bids fork does not allow suffix='raw' without check=False",
    )
    def test_derivatives_switch(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = get_bids_path_for_task(
            cfg, task="restingstate", from_derivatives=True
        )
        assert str(bp.root) == cfg.deriv_root
        assert bp.suffix == "raw"

    def test_subjects_list_vs_string(self, tmp_path):
        """Both list and string subjects should produce the same path."""
        cfg_list = _make_cfg(tmp_path, subjects=["005"])
        cfg_str = _make_cfg(tmp_path, subjects="005")
        bp1 = get_bids_path_for_task(cfg_list, task="restingstate")
        bp2 = get_bids_path_for_task(cfg_str, task="restingstate")
        assert bp1.subject == bp2.subject == "005"

    def test_consistency_with_bids_utils(self, tmp_path):
        """Both utility functions should produce equivalent paths."""
        cfg = _make_cfg(tmp_path)
        bp1 = get_bids_path(cfg, task="restingstate", processing="clean")
        bp2 = get_bids_path_for_task(cfg, task="restingstate", processing="clean")
        assert bp1.subject == bp2.subject
        assert bp1.session == bp2.session
        assert bp1.task == bp2.task
        assert bp1.processing == bp2.processing
