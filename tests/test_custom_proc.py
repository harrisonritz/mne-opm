"""Tests for the ``custom_proc`` deriv routing in ``preprocessing._io``.

When ``cfg.custom_proc`` is set (e.g. ``'init'``), custom preprocessing
steps must:

1. Read inputs from ``deriv_root`` with ``proc-<custom_proc>`` if files
   exist there, otherwise fall back to ``bids_root``.
2. Write outputs to ``deriv_root`` with ``proc-<custom_proc>`` instead
   of overwriting the BIDS data files.

When ``cfg.custom_proc`` is unset / ``None``, behaviour falls back to
the legacy contract of reading from / writing to ``bids_root``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import mne_bids
import numpy as np
import pytest
from mne_bids import BIDSPath

from custom.preprocessing._io import (
    find_custom_input_paths,
    get_custom_output_path,
    get_custom_proc,
    write_raw_bids_custom_step,
)


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


def _make_bids_path(root, **overrides):
    defaults = dict(
        root=str(root),
        subject="001",
        session="01",
        task="restingstate",
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    defaults.update(overrides)
    return BIDSPath(**defaults)


def _write_minimal_raw_to_bids(bids_root: Path, raw, task: str = "restingstate"):
    """Write a minimal raw file to a BIDS dataset for testing.

    Returns the BIDSPath that was written.
    """
    bp = BIDSPath(
        root=str(bids_root),
        subject="001",
        session="01",
        task=task,
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    raw.set_meas_date(None)
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bp,
        allow_preload=True,
        overwrite=True,
        format="FIF",
    )
    return bp


# ---------------------------------------------------------------------------
# get_custom_proc
# ---------------------------------------------------------------------------


class TestGetCustomProc:
    """get_custom_proc unwraps the cfg attribute, returning None if unset."""

    def test_missing_attribute_returns_none(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        assert get_custom_proc(cfg) is None

    def test_none_attribute_returns_none(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc=None)
        assert get_custom_proc(cfg) is None

    def test_empty_string_returns_none(self, tmp_path):
        """Empty string should be treated as unset."""
        cfg = _make_cfg(tmp_path, custom_proc="")
        assert get_custom_proc(cfg) is None

    def test_string_value_returned_verbatim(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc="init")
        assert get_custom_proc(cfg) == "init"

    def test_other_string_value(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc="cleaned")
        assert get_custom_proc(cfg) == "cleaned"


# ---------------------------------------------------------------------------
# get_custom_output_path
# ---------------------------------------------------------------------------


class TestGetCustomOutputPath:
    """get_custom_output_path redirects writes when custom_proc is set."""

    def test_no_custom_proc_returns_source_unchanged(self, tmp_path):
        cfg = _make_cfg(tmp_path)  # no custom_proc
        source_bp = _make_bids_path(tmp_path / "bids")

        output_bp = get_custom_output_path(cfg, source_bp)
        assert str(output_bp.root) == str(source_bp.root)
        assert output_bp.processing == source_bp.processing  # both None

    def test_custom_proc_redirects_to_deriv_root(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc="init")
        source_bp = _make_bids_path(tmp_path / "bids")

        output_bp = get_custom_output_path(cfg, source_bp)
        assert str(output_bp.root) == cfg.deriv_root
        assert output_bp.processing == "init"

    def test_custom_proc_preserves_subject_session_task(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc="init")
        source_bp = _make_bids_path(
            tmp_path / "bids", subject="002", session="03", task="other"
        )

        output_bp = get_custom_output_path(cfg, source_bp)
        assert output_bp.subject == "002"
        assert output_bp.session == "03"
        assert output_bp.task == "other"

    def test_custom_proc_filename_includes_proc_label(self, tmp_path):
        cfg = _make_cfg(tmp_path, custom_proc="init")
        source_bp = _make_bids_path(tmp_path / "bids")

        output_bp = get_custom_output_path(cfg, source_bp)
        assert "proc-init" in str(output_bp.fpath)

    def test_returns_independent_copy(self, tmp_path):
        """Mutating the returned bp should not affect the source bp."""
        cfg = _make_cfg(tmp_path, custom_proc="init")
        source_bp = _make_bids_path(tmp_path / "bids")

        output_bp = get_custom_output_path(cfg, source_bp)
        output_bp.update(task="otherTask", check=False)
        assert source_bp.task == "restingstate"

    def test_returns_copy_when_proc_unset(self, tmp_path):
        """Even with no custom_proc, the returned bp should be a copy."""
        cfg = _make_cfg(tmp_path)
        source_bp = _make_bids_path(tmp_path / "bids")

        output_bp = get_custom_output_path(cfg, source_bp)
        output_bp.update(task="otherTask")
        assert source_bp.task == "restingstate"


# ---------------------------------------------------------------------------
# find_custom_input_paths
# ---------------------------------------------------------------------------


class TestFindCustomInputPaths:
    """find_custom_input_paths prefers deriv proc-* files when custom_proc is set."""

    def test_falls_back_to_bids_root_when_proc_unset(
        self, tmp_path, raw_meg
    ):
        """Without custom_proc, only the BIDS root is searched."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path)  # no custom_proc

        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == str(bids_root)
        assert paths[0].processing is None

    def test_falls_back_to_bids_root_when_no_deriv_files(
        self, tmp_path, raw_meg
    ):
        """With custom_proc set but no deriv files yet, returns bids_root."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path, custom_proc="init")

        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == str(bids_root)
        assert paths[0].processing is None

    def test_prefers_deriv_when_proc_files_exist(self, tmp_path, raw_meg):
        """With custom_proc set and deriv files present, those are used."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        # Also write a proc-init copy under deriv_root
        deriv_root = tmp_path / "derivatives"
        deriv_root.mkdir()
        deriv_bp = BIDSPath(
            root=str(deriv_root),
            subject="001",
            session="01",
            task="restingstate",
            datatype="meg",
            suffix="meg",
            processing="init",
            extension=".fif",
            check=False,
        )
        mne_bids.write_raw_bids(
            raw=raw_meg,
            bids_path=deriv_bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        cfg = _make_cfg(tmp_path, custom_proc="init")
        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == str(deriv_root)
        assert paths[0].processing == "init"

    def test_returns_empty_when_no_files(self, tmp_path):
        """If neither deriv nor bids has files, returns []."""
        (tmp_path / "bids").mkdir()
        cfg = _make_cfg(tmp_path)
        paths = find_custom_input_paths(cfg, task="restingstate")
        assert paths == []

    def test_handles_missing_deriv_root(self, tmp_path, raw_meg):
        """A non-existent deriv_root doesn't raise; we fall back to bids."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path, custom_proc="init")
        # deriv_root has not been created
        assert not Path(cfg.deriv_root).exists()

        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == str(bids_root)


# ---------------------------------------------------------------------------
# write_raw_bids_custom_step
# ---------------------------------------------------------------------------


class TestWriteRawBidsCustomStep:
    """End-to-end behaviour of write_raw_bids_custom_step."""

    def test_writes_to_bids_root_when_proc_unset(self, tmp_path, raw_meg):
        """With custom_proc unset, writes back to the source BIDS file."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(
            bids_root, raw_meg, task="restingstate"
        )

        cfg = _make_cfg(tmp_path)  # no custom_proc

        # Modify raw and write back
        raw_meg.info["bads"] = ["MEG001"]
        output_bp = write_raw_bids_custom_step(raw_meg, cfg, source_bp)

        assert str(output_bp.root) == str(bids_root)
        assert output_bp.processing is None
        assert output_bp.fpath.exists()

    def test_writes_to_deriv_with_proc_when_set(self, tmp_path, raw_meg):
        """With custom_proc set, writes go to deriv_root with proc-* tag."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(
            bids_root, raw_meg, task="restingstate"
        )

        cfg = _make_cfg(tmp_path, custom_proc="init")
        output_bp = write_raw_bids_custom_step(raw_meg, cfg, source_bp)

        assert str(output_bp.root) == cfg.deriv_root
        assert output_bp.processing == "init"
        assert output_bp.fpath.exists()
        # The proc-init filename should be used
        assert "proc-init" in str(output_bp.fpath)

    def test_does_not_modify_bids_root_when_redirected(
        self, tmp_path, raw_meg
    ):
        """A redirected write must NOT touch the BIDS data file."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(
            bids_root, raw_meg, task="restingstate"
        )
        original_mtime = source_bp.fpath.stat().st_mtime
        original_size = source_bp.fpath.stat().st_size

        cfg = _make_cfg(tmp_path, custom_proc="init")
        # Mark a channel bad so the write would change the file content if
        # it did go to bids_root.
        raw_meg.info["bads"] = ["MEG001"]
        write_raw_bids_custom_step(raw_meg, cfg, source_bp)

        # bids_root file should be untouched
        assert source_bp.fpath.stat().st_mtime == original_mtime
        assert source_bp.fpath.stat().st_size == original_size

    def test_seeds_events_files_on_first_redirected_write(
        self, tmp_path, raw_with_stim
    ):
        """Events files are copied from source on first redirected write."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()

        # Write raw + events to BIDS
        events = mne.find_events(raw_with_stim, stim_channel="STI001")
        bp = BIDSPath(
            root=str(bids_root), subject="001", session="01",
            task="restingstate", datatype="meg", suffix="meg",
            extension=".fif",
        )
        mne_bids.write_raw_bids(
            raw=raw_with_stim,
            bids_path=bp,
            events=events,
            event_id={"trig1": 1, "trig2": 2},
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        source_events = bp.copy().update(suffix="events", extension=".tsv").fpath
        assert source_events.exists()
        source_event_lines = source_events.read_text().splitlines()

        cfg = _make_cfg(tmp_path, custom_proc="init")
        output_bp = write_raw_bids_custom_step(raw_with_stim, cfg, bp)

        # The redirected write should have an events.tsv next to it that
        # matches the source (the seed plus preserve+restore).
        deriv_events = output_bp.copy().update(
            suffix="events", extension=".tsv"
        ).fpath
        assert deriv_events.exists()
        assert deriv_events.read_text().splitlines() == source_event_lines

    def test_forwards_empty_room(self, tmp_path, raw_meg):
        """The empty_room kwarg is forwarded to write_raw_bids."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(
            bids_root, raw_meg, task="restingstate"
        )
        er_bp = _write_minimal_raw_to_bids(bids_root, raw_meg, task="noise")

        cfg = _make_cfg(tmp_path)
        # Should accept empty_room without raising
        output_bp = write_raw_bids_custom_step(
            raw_meg, cfg, source_bp, empty_room=er_bp
        )
        assert output_bp.fpath.exists()


# ---------------------------------------------------------------------------
# Module integration: bad_channels.save_results respects custom_proc
# ---------------------------------------------------------------------------


class TestBadChannelsRespectsCustomProc:
    """Verify bad_channels.save_results writes to deriv when custom_proc set."""

    def test_save_redirects_when_custom_proc_set(self, tmp_path, raw_meg):
        """End-to-end: custom_proc='init' writes to deriv_root proc-init."""
        from custom.preprocessing.bad_channels import BadChannelsAnalysis

        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = SimpleNamespace(
            bids_root=str(bids_root),
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
            custom_proc="init",
        )

        analysis = BadChannelsAnalysis(cfg)
        results = {cfg.task: raw_meg, "bads": ["MEG001"]}
        analysis.save_results(results)

        # Output file must exist in deriv_root with proc-init label
        deriv_file = (
            Path(cfg.deriv_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_meg.fif"
        )
        assert deriv_file.exists()

        # bids_root file should not have the bad channels marked (untouched)
        bids_channels_tsv = (
            Path(cfg.bids_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_channels.tsv"
        )
        # The original write set MEG001 as good; after our redirect, the
        # bids_root channels.tsv must still mark it as good.
        text = bids_channels_tsv.read_text()
        # find MEG001 row
        for line in text.splitlines():
            if line.startswith("MEG001\t"):
                assert "good" in line.lower()
                break

        # The deriv channels.tsv should mark MEG001 as bad
        deriv_channels_tsv = (
            Path(cfg.deriv_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_channels.tsv"
        )
        text = deriv_channels_tsv.read_text()
        for line in text.splitlines():
            if line.startswith("MEG001\t"):
                assert "bad" in line.lower()
                break

    def test_save_overwrites_bids_when_custom_proc_unset(
        self, tmp_path, raw_meg
    ):
        """Without custom_proc, save_results overwrites bids_root (legacy)."""
        from custom.preprocessing.bad_channels import BadChannelsAnalysis

        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = SimpleNamespace(
            bids_root=str(bids_root),
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

        analysis = BadChannelsAnalysis(cfg)
        results = {cfg.task: raw_meg, "bads": ["MEG001"]}
        analysis.save_results(results)

        # No deriv file should appear
        deriv_file = (
            Path(cfg.deriv_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_meg.fif"
        )
        assert not deriv_file.exists()

        # bids_root channels.tsv should now mark MEG001 as bad
        bids_channels_tsv = (
            Path(cfg.bids_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_channels.tsv"
        )
        text = bids_channels_tsv.read_text()
        marked_bad = False
        for line in text.splitlines():
            if line.startswith("MEG001\t"):
                marked_bad = "bad" in line.lower()
                break
        assert marked_bad


# ---------------------------------------------------------------------------
# Module integration: bad_segments.save_results respects custom_proc
# ---------------------------------------------------------------------------


class TestBadSegmentsRespectsCustomProc:
    """bad_segments.save_results redirects to deriv when custom_proc set."""

    def test_save_writes_annotations_to_deriv(self, tmp_path, raw_meg):
        from custom.preprocessing.bad_segments import BadSegmentsAnalysis

        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = SimpleNamespace(
            bids_root=str(bids_root),
            deriv_root=str(tmp_path / "derivatives"),
            subjects=["001"],
            sessions=["01"],
            task="restingstate",
            ch_types=["mag"],
            l_freq=0.1,
            h_freq=30.0,
            process_empty_room=False,
            find_breaks=False,
            n_jobs=1,
            custom_proc="init",
        )

        # Add a bad annotation to verify it's written
        raw_meg.set_annotations(
            mne.Annotations(onset=[1.0], duration=[0.5], description=["BAD_test"])
        )

        analysis = BadSegmentsAnalysis(cfg)
        results = {cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        deriv_file = (
            Path(cfg.deriv_root)
            / "sub-001" / "ses-01" / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_meg.fif"
        )
        assert deriv_file.exists()

        # Read back and confirm the annotation is preserved
        loaded = mne.io.read_raw_fif(deriv_file, preload=True)
        descriptions = list(loaded.annotations.description)
        assert any(d.startswith("BAD_test") for d in descriptions)


# ---------------------------------------------------------------------------
# load_data prefers deriv proc-* on subsequent steps
# ---------------------------------------------------------------------------


class TestLoadDataPrefersDeriv:
    """Once a custom step has written to deriv proc-init, the next step
    in the chain reads from there rather than re-reading bids_root."""

    def test_apply_hfc_load_finds_deriv_after_first_step(
        self, tmp_path, raw_meg
    ):
        from custom.preprocessing.apply_hfc import ApplyHFCAnalysis
        from custom.preprocessing.bad_channels import BadChannelsAnalysis

        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = SimpleNamespace(
            bids_root=str(bids_root),
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
            custom_proc="init",
            _do_HFC=True,
            _hfc_order=1,
        )

        # Step 1: bad_channels writes to deriv proc-init
        analysis = BadChannelsAnalysis(cfg)
        analysis.save_results({cfg.task: raw_meg, "bads": ["MEG001"]})

        # Step 2: apply_hfc should now find the deriv proc-init file
        hfc = ApplyHFCAnalysis(cfg)
        data = hfc.load_data()

        assert cfg.task in data
        # The loaded raw should reflect the bads tagged by step 1
        assert "MEG001" in data[cfg.task].info["bads"]


# ---------------------------------------------------------------------------
# noise + task ordering when custom_proc is set
# ---------------------------------------------------------------------------


class TestNoiseTaskOrdering:
    """When both noise and task are saved, noise must be saved first so the
    task save can use its newly-written deriv path as the empty_room."""

    @patch("custom.preprocessing.bad_channels.write_raw_bids_custom_step")
    @patch("custom.preprocessing.bad_channels.find_custom_input_paths")
    def test_noise_saved_before_task(
        self, mock_find, mock_write, raw_meg, tmp_path
    ):
        from custom.preprocessing.bad_channels import BadChannelsAnalysis

        cfg = SimpleNamespace(
            bids_root=str(tmp_path / "bids"),
            deriv_root=str(tmp_path / "derivatives"),
            subjects=["001"],
            sessions=["01"],
            task="restingstate",
            ch_types=["mag"],
            l_freq=1.0,
            h_freq=100.0,
            process_empty_room=True,
            find_breaks=False,
            n_jobs=1,
            custom_proc="init",
        )

        noise_source_bp = MagicMock(name="noise_src")
        task_source_bp = MagicMock(name="task_src")
        noise_output_bp = MagicMock(name="noise_out")
        task_output_bp = MagicMock(name="task_out")
        # First find call is for noise, second for the task
        mock_find.side_effect = [[noise_source_bp], [task_source_bp]]
        mock_write.side_effect = [noise_output_bp, task_output_bp]

        analysis = BadChannelsAnalysis(cfg)
        results = {"noise": raw_meg, cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        # Two writes, noise first, then task
        assert mock_write.call_count == 2

        # Task write should reference the noise output as its empty_room
        task_call = mock_write.call_args_list[-1]
        assert task_call.kwargs.get("empty_room") is noise_output_bp
