"""Tests for the preprocessing re-run safeguards.

Covers three behaviours added to make custom preprocessing reproducible
across re-runs:

1. ``init_derivatives.run`` clears stale ``proc-<custom_proc>`` files from a
   previous run while preserving everything else in the ``meg`` folder
   (notably the ``*_trans.fif`` coregistration).
2. ``assert_not_raw_bids_write`` refuses writes into the raw BIDS data
   directory but allows writes into the derivatives tree.
3. ``_seed_sidecars`` names the data JSON sidecar to match the output FIF
   suffix (``_raw.json`` in derivative mode, not a stray ``_meg.json``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mne_bids import BIDSPath

from custom.preprocessing import init_derivatives
from custom.preprocessing._io import (
    _seed_sidecars,
    assert_not_raw_bids_write,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path, **overrides):
    bids_root = tmp_path / "bids"
    deriv_root = bids_root / "derivatives" / "analysis"
    defaults = dict(
        bids_root=str(bids_root),
        deriv_root=str(deriv_root),
        subjects=["011"],
        sessions=["01"],
        custom_proc="init",
        task="TSX",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _meg_dir(cfg, subject="011", session="01"):
    d = Path(cfg.deriv_root) / f"sub-{subject}" / f"ses-{session}" / "meg"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# init_derivatives.run
# ---------------------------------------------------------------------------


class TestInitDerivatives:
    """Tests for clearing stale proc-<custom_proc> derivatives."""

    def test_clears_proc_init_preserves_others(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        meg = _meg_dir(cfg)

        stem = "sub-011_ses-01_task-TSX_run-01"
        stale = [
            meg / f"{stem}_proc-init_raw.fif",
            meg / f"{stem}_proc-init_split-01_raw.fif",
            meg / f"{stem}_proc-init_meg.json",
            meg / f"{stem}_proc-init_raw.json",
            meg / f"{stem}_proc-init_channels.tsv",
            meg / f"{stem}_proc-init_events.tsv",
            meg / f"{stem}_proc-init_bads.tsv",
            meg / f"{stem}_proc-init_scores.json",
        ]
        preserved = [
            meg / "sub-011_ses-01_task-TSX_trans.fif",
            meg / f"{stem}_proc-filt_raw.fif",
            meg / f"{stem}_proc-clean_raw.fif",
            meg / "sub-011_ses-01_task-TSX_epo.fif",
        ]
        for f in stale + preserved:
            f.write_text("x")

        init_derivatives.run(cfg)

        for f in stale:
            assert not f.exists(), f"{f.name} should have been removed"
        for f in preserved:
            assert f.exists(), f"{f.name} should have been preserved"

    def test_noise_task_proc_init_cleared(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        meg = _meg_dir(cfg)
        noise = meg / "sub-011_ses-01_task-noise_proc-init_raw.fif"
        noise.write_text("x")

        init_derivatives.run(cfg)

        assert not noise.exists()

    def test_only_configured_subject_cleared(self, tmp_path):
        cfg = _make_cfg(tmp_path, subjects=["011"])
        meg_011 = _meg_dir(cfg, subject="011")
        meg_042 = _meg_dir(cfg, subject="042")
        f_011 = meg_011 / "sub-011_ses-01_task-TSX_run-01_proc-init_raw.fif"
        f_042 = meg_042 / "sub-042_ses-01_task-TSX_run-01_proc-init_raw.fif"
        f_011.write_text("x")
        f_042.write_text("x")

        init_derivatives.run(cfg)

        assert not f_011.exists()
        assert f_042.exists(), "other subjects must not be touched"

    def test_no_custom_proc_is_noop(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc=None)
        meg = _meg_dir(cfg)
        f = meg / "sub-011_ses-01_task-TSX_run-01_proc-init_raw.fif"
        f.write_text("x")

        init_derivatives.run(cfg)

        assert f.exists(), "without custom_proc nothing should be cleared"

    def test_missing_deriv_root_is_noop(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        # deriv_root never created — should not raise.
        init_derivatives.run(cfg)


# ---------------------------------------------------------------------------
# assert_not_raw_bids_write
# ---------------------------------------------------------------------------


class TestAssertNotRawBidsWrite:
    """Tests for the raw-BIDS write guard."""

    def test_allows_deriv_root_write(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        target = (
            tmp_path
            / "bids"
            / "derivatives"
            / "analysis"
            / "sub-011"
            / "ses-01"
            / "meg"
            / "sub-011_ses-01_task-TSX_run-01_proc-init_raw.fif"
        )
        # Should not raise.
        assert_not_raw_bids_write(target, cfg, context="test")

    def test_blocks_raw_bids_write(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        target = (
            tmp_path
            / "bids"
            / "sub-011"
            / "ses-01"
            / "meg"
            / "sub-011_ses-01_task-TSX_run-01_meg.fif"
        )
        with pytest.raises(RuntimeError, match="raw BIDS"):
            assert_not_raw_bids_write(target, cfg, context="test")

    def test_accepts_bidspath(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        bp = BIDSPath(
            root=cfg.bids_root,
            subject="011",
            session="01",
            task="TSX",
            run="01",
            datatype="meg",
            suffix="meg",
            extension=".fif",
            check=False,
        )
        with pytest.raises(RuntimeError, match="raw BIDS"):
            assert_not_raw_bids_write(bp, cfg, context="test")

    def test_outside_bids_root_allowed(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        target = tmp_path / "somewhere_else" / "file.fif"
        assert_not_raw_bids_write(target, cfg, context="test")


# ---------------------------------------------------------------------------
# _seed_sidecars JSON suffix
# ---------------------------------------------------------------------------


class TestSeedSidecarsJsonSuffix:
    """The data JSON sidecar must follow the output FIF suffix."""

    def test_meg_json_becomes_raw_json_in_derivative_mode(self, tmp_path):
        cfg = _make_cfg(tmp_path)

        source_bp = BIDSPath(
            root=cfg.bids_root,
            subject="011",
            session="01",
            task="TSX",
            run="01",
            datatype="meg",
            suffix="meg",
            extension=".fif",
            check=False,
        )
        output_bp = source_bp.copy().update(
            root=cfg.deriv_root,
            processing="init",
            suffix="raw",
            check=False,
        )

        # Create the source meg.json (and a channels.tsv) to be seeded.
        src_json = (
            source_bp.copy().update(suffix="meg", extension=".json", check=False).fpath
        )
        src_json.parent.mkdir(parents=True, exist_ok=True)
        src_json.write_text("{}")

        _seed_sidecars(source_bp, output_bp)

        raw_json = (
            output_bp.copy().update(suffix="raw", extension=".json", check=False).fpath
        )
        meg_json = (
            output_bp.copy().update(suffix="meg", extension=".json", check=False).fpath
        )

        assert raw_json.exists(), "derivative JSON should be named *_raw.json"
        assert not meg_json.exists(), "no stray *_meg.json should be created"
