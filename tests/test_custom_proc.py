"""Tests for the ``custom_proc`` deriv routing in ``preprocessing._io``.

When ``cfg.custom_proc`` is set (e.g. ``'init'``), custom preprocessing
steps must:

1. Read inputs from ``deriv_root`` with ``proc-<custom_proc>`` if files
   exist there, otherwise fall back to ``bids_root``.
2. Write outputs to ``deriv_root`` with ``proc-<custom_proc>`` instead
   of overwriting the BIDS data files.

When ``cfg.custom_proc`` is unset / ``None``, inputs are read from
``bids_root`` and any write raises, so the raw BIDS data are never overwritten.

A branch run (``custom.trunk``) seeds its ``deriv_root`` with a trunk's
``proc-<custom_proc>`` files, so its custom steps continue from there.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import mne
import mne_bids
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

    def test_no_custom_proc_raises(self, tmp_path):
        cfg = _make_cfg(tmp_path)  # no custom_proc
        source_bp = _make_bids_path(tmp_path / "bids")
        with pytest.raises(ValueError, match="custom_proc must be set"):
            get_custom_output_path(cfg, source_bp)

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


# ---------------------------------------------------------------------------
# find_custom_input_paths
# ---------------------------------------------------------------------------


class TestFindCustomInputPaths:
    """find_custom_input_paths prefers deriv proc-* files when custom_proc is set."""

    def test_falls_back_to_bids_root_when_proc_unset(self, tmp_path, raw_meg):
        """Without custom_proc, only the BIDS root is searched."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path)  # no custom_proc

        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == str(bids_root)
        assert paths[0].processing is None

    def test_falls_back_to_bids_root_when_no_deriv_files(self, tmp_path, raw_meg):
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
        source_bp = _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path, custom_proc="init")

        # Produce a proc-init derivative through the real write helper so it is
        # named sub-..._proc-init_raw.fif (suffix 'raw'), matching what the
        # custom steps actually write and what find_custom_input_paths looks
        # for.
        write_raw_bids_custom_step(raw_meg, cfg, source_bp)

        paths = find_custom_input_paths(cfg, task="restingstate")
        assert len(paths) == 1
        assert str(paths[0].root) == cfg.deriv_root
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

    def test_raises_when_proc_unset_would_overwrite_bids(self, tmp_path, raw_meg):
        """With custom_proc unset, writing is refused so the raw BIDS data is
        never overwritten."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path)  # no custom_proc

        raw_meg.info["bads"] = ["MEG001"]
        with pytest.raises(ValueError, match="custom_proc must be set"):
            write_raw_bids_custom_step(raw_meg, cfg, source_bp)

    def test_writes_to_deriv_with_proc_when_set(self, tmp_path, raw_meg):
        """With custom_proc set, writes go to deriv_root with proc-* tag."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")

        cfg = _make_cfg(tmp_path, custom_proc="init")
        output_bp = write_raw_bids_custom_step(raw_meg, cfg, source_bp)

        assert str(output_bp.root) == cfg.deriv_root
        assert output_bp.processing == "init"
        assert output_bp.fpath.exists()
        # The proc-init filename should be used
        assert "proc-init" in str(output_bp.fpath)

        # Maxwell filtering reads the reference derivative via MNE-BIDS.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = mne_bids.read_raw_bids(output_bp, verbose="ERROR")
        assert restored.n_times == raw_meg.n_times
        assert not any(
            "meg.json associated" in str(w.message)
            or "participants.tsv file not found" in str(w.message)
            for w in caught
        )

    def test_does_not_modify_bids_root_when_redirected(self, tmp_path, raw_meg):
        """A redirected write must NOT touch the BIDS data file."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        source_bp = _write_minimal_raw_to_bids(bids_root, raw_meg, task="restingstate")
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
            root=str(bids_root),
            subject="001",
            session="01",
            task="restingstate",
            datatype="meg",
            suffix="meg",
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
        deriv_events = output_bp.copy().update(suffix="events", extension=".tsv").fpath
        assert deriv_events.exists()
        assert deriv_events.read_text().splitlines() == source_event_lines


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
            / "sub-001"
            / "ses-01"
            / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_raw.fif"
        )
        assert deriv_file.exists()

        # bids_root file should not have the bad channels marked (untouched)
        bids_channels_tsv = (
            Path(cfg.bids_root)
            / "sub-001"
            / "ses-01"
            / "meg"
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
            / "sub-001"
            / "ses-01"
            / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_channels.tsv"
        )
        text = deriv_channels_tsv.read_text()
        for line in text.splitlines():
            if line.startswith("MEG001\t"):
                assert "bad" in line.lower()
                break

    def test_save_raises_when_custom_proc_unset(self, tmp_path, raw_meg):
        """Without custom_proc the save would overwrite bids_root, which is
        now refused; the BIDS channels.tsv must stay untouched."""
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
        with pytest.raises(ValueError, match="custom_proc must be set"):
            analysis.save_results(results)

        # The refusal must happen before any mutation of the BIDS data, so
        # MEG001 stays good in the raw BIDS channels.tsv.
        bids_channels_tsv = (
            Path(cfg.bids_root)
            / "sub-001"
            / "ses-01"
            / "meg"
            / "sub-001_ses-01_task-restingstate_channels.tsv"
        )
        for line in bids_channels_tsv.read_text().splitlines():
            if line.startswith("MEG001\t"):
                assert "good" in line.lower()
                break


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
            / "sub-001"
            / "ses-01"
            / "meg"
            / "sub-001_ses-01_task-restingstate_proc-init_raw.fif"
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

    def test_apply_hfc_load_finds_deriv_after_first_step(self, tmp_path, raw_meg):
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
# Trunk / branch: state file and seeding a variant (custom.trunk)
# ---------------------------------------------------------------------------


def _make_trunk(tmp_path, raw, monkeypatch):
    """Build a trunk deriv_root the way the trunk steps and save-state do."""
    from custom.trunk import save_trunk_state

    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    source_bp = _write_minimal_raw_to_bids(bids_root, raw, task="restingstate")
    cfg = _make_cfg(tmp_path, deriv_root=str(tmp_path / "trunk"), custom_proc="init")
    written = write_raw_bids_custom_step(raw, cfg, source_bp)
    meg = written.fpath.parent
    for split in ("01", "02"):  # empty-room derivative split across two files
        (meg / f"sub-001_ses-01_task-noise_proc-init_split-{split}_raw.fif").write_text(
            split
        )
    monkeypatch.setenv("MNE_OPM_BREAKS_ALREADY_ANNOTATED", "1")
    monkeypatch.setenv("MNE_OPM_REQUIRE_RESPONSE_METADATA", "1")
    monkeypatch.delenv("MNE_OPM_STAGE1_BREAKS_SAVED", raising=False)
    (state_path,) = save_trunk_state(cfg)
    return cfg, meg, state_path


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


class TestTrunkState:
    """The trunk state file records the env flags run_preproc.sh exported."""

    def test_round_trip_atomic_and_cleared_by_init(self, tmp_path):
        from custom.preprocessing import init_derivatives
        from custom.trunk import read_trunk_state, trunk_state_path, write_trunk_state

        cfg = _make_cfg(tmp_path, custom_proc="init")
        path = trunk_state_path(cfg.deriv_root, "001", "01", "restingstate", "init")
        assert (
            path.name
            == "sub-001_ses-01_task-restingstate_proc-init_desc-trunk_state.json"
        )
        env = {
            "MNE_OPM_BREAKS_ALREADY_ANNOTATED": "1",
            "MNE_OPM_STAGE1_BREAKS_SAVED": "0",
        }
        state = write_trunk_state(path, "init", environ=env)
        assert state == {
            "custom_proc": "init",
            "flags": {
                "MNE_OPM_BREAKS_ALREADY_ANNOTATED": True,
                "MNE_OPM_REQUIRE_RESPONSE_METADATA": False,
                "MNE_OPM_STAGE1_BREAKS_SAVED": False,
            },
        }
        assert read_trunk_state(path) == state
        assert list(path.parent.iterdir()) == [path]  # no temp file left behind
        # Re-running the trunk's INIT clears it with the files it describes.
        init_derivatives.run(cfg)
        assert not path.exists()

    def test_read_errors(self, tmp_path):
        from custom.trunk import read_trunk_state

        with pytest.raises(FileNotFoundError, match="PREPROC_STAGE=trunk"):
            read_trunk_state(tmp_path / "missing.json")
        bad = tmp_path / "bad.json"
        bad.write_text('{"custom_proc": "init", "flags": {}}')
        with pytest.raises(ValueError, match="Malformed"):
            read_trunk_state(bad)


class TestSeedFromTrunk:
    """A branch copies the trunk's proc-init files into its own deriv_root."""

    def test_copies_proc_files_and_overwrites_stale(
        self, tmp_path, raw_meg, monkeypatch
    ):
        from custom.trunk import seed_from_trunk

        trunk_cfg, trunk_meg, state_path = _make_trunk(tmp_path, raw_meg, monkeypatch)
        trunk_root = Path(trunk_cfg.deriv_root)
        before = _snapshot(trunk_root)
        cfg = _make_cfg(
            tmp_path, deriv_root=str(tmp_path / "variant"), custom_proc="init"
        )
        meg = Path(cfg.deriv_root) / trunk_meg.relative_to(trunk_root)
        meg.mkdir(parents=True)
        task_fif = meg / "sub-001_ses-01_task-restingstate_proc-init_raw.fif"
        task_fif.write_text("stale")
        stale_split = (
            meg / "sub-001_ses-01_task-restingstate_proc-init_split-03_raw.fif"
        )
        stale_split.write_text("stale")
        keep = meg / "sub-001_ses-01_task-restingstate_proc-filt_raw.fif"
        keep.write_text("keep")

        flags = seed_from_trunk(cfg, trunk_root)

        assert flags == {
            "MNE_OPM_BREAKS_ALREADY_ANNOTATED": True,
            "MNE_OPM_REQUIRE_RESPONSE_METADATA": True,
            "MNE_OPM_STAGE1_BREAKS_SAVED": False,
        }
        assert _snapshot(trunk_root) == before  # the trunk is only read
        copied = {rel for rel in before if rel.name != state_path.name}
        names = {rel.name for rel in copied}
        for suffix in ("raw.fif", "channels.tsv", "meg.json"):
            assert f"sub-001_ses-01_task-restingstate_proc-init_{suffix}" in names
        assert "sub-001_ses-01_task-noise_proc-init_split-02_raw.fif" in names
        assert Path("participants.tsv") in copied
        variant = Path(cfg.deriv_root)
        for rel in copied:
            assert (variant / rel).read_bytes() == before[rel][0], rel
        assert not (meg / state_path.name).exists()
        assert not stale_split.exists() and keep.read_text() == "keep"
        assert not list(variant.rglob(".*.tmp"))
        (found,) = find_custom_input_paths(cfg, task="restingstate")
        assert found.fpath == task_fif
        assert mne.io.read_raw_fif(task_fif).n_times == raw_meg.n_times

        task_fif.write_text("rewritten by a branch step")
        seed_from_trunk(cfg, trunk_root)
        assert task_fif.read_bytes() == before[task_fif.relative_to(variant)][0]

    @pytest.mark.parametrize(
        "case, exc, match",
        [
            ("no_state", FileNotFoundError, "PREPROC_STAGE=trunk"),
            ("no_fif", FileNotFoundError, "no proc-init raw FIF"),
            ("unset", ValueError, "MNE_OPM_TRUNK_DERIV must be set"),
            ("relative", ValueError, "must be absolute"),
            ("same_root", ValueError, "its own deriv_root"),
            ("other_proc", ValueError, "custom_proc"),
        ],
    )
    def test_refuses_before_touching_variant(
        self, tmp_path, raw_meg, monkeypatch, case, exc, match
    ):
        from custom.trunk import seed_from_trunk, write_trunk_state

        trunk_cfg, trunk_meg, state_path = _make_trunk(tmp_path, raw_meg, monkeypatch)
        trunk_root = Path(trunk_cfg.deriv_root)
        cfg = _make_cfg(
            tmp_path, deriv_root=str(tmp_path / "variant"), custom_proc="init"
        )
        stale = Path(cfg.deriv_root) / "sub-001" / "ses-01" / "meg"
        stale.mkdir(parents=True)
        stale = stale / "sub-001_ses-01_task-restingstate_proc-init_raw.fif"
        stale.write_text("stale")
        if case == "no_state":
            state_path.unlink()
        elif case == "no_fif":
            for fif in trunk_meg.glob("*.fif"):
                fif.unlink()
        elif case == "unset":
            trunk_root = None
        elif case == "relative":
            trunk_root = Path(os.path.relpath(trunk_root))
        elif case == "same_root":
            cfg.deriv_root = str(trunk_root)
        elif case == "other_proc":
            write_trunk_state(state_path, "other")

        with pytest.raises(exc, match=match):
            seed_from_trunk(cfg, trunk_root)
        assert stale.read_text() == "stale"

    def test_cli_prints_only_set_flags(self, tmp_path, raw_meg, monkeypatch, capsys):
        from custom import trunk

        trunk_cfg, _, _ = _make_trunk(tmp_path, raw_meg, monkeypatch)
        cfg = _make_cfg(
            tmp_path, deriv_root=str(tmp_path / "variant"), custom_proc="init"
        )
        seen = []
        monkeypatch.setattr(
            trunk,
            "load_config",
            lambda path: seen.append(os.environ.get("MNE_OPM_SKIP_METADATA")) or cfg,
        )
        monkeypatch.delenv("MNE_OPM_SKIP_METADATA", raising=False)
        monkeypatch.setenv("MNE_OPM_TRUNK_DERIV", trunk_cfg.deriv_root)
        capsys.readouterr()

        assert trunk.main(["seed", "--config", "variant.py"]) == 0
        out = capsys.readouterr().out
        assert out.splitlines() == [
            "MNE_OPM_BREAKS_ALREADY_ANNOTATED",
            "MNE_OPM_REQUIRE_RESPONSE_METADATA",
        ]
        assert seen == ["1"] and "MNE_OPM_SKIP_METADATA" not in os.environ

        monkeypatch.setenv("MNE_OPM_TRUNK_DERIV", str(tmp_path / "nowhere"))
        assert trunk.main(["seed", "--config", "variant.py"]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "[trunk] ERROR" in captured.err
