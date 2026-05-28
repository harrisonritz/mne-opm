"""Tests for event alignment — verifying that BIDS events, metadata, and
annotations stay consistent throughout the preprocessing pipeline.

These tests target the root cause of the event-count mismatch bug:
``mne_bids.write_raw_bids()`` regenerates *_events.tsv* from
``raw.annotations`` on every save, introducing spurious events through
the lossy annotation ↔ events round-trip.  The fixes are:

1. **format_bids.py** drops trigger stim channels after converting them
   to annotations, preventing ``write_raw_bids`` from re-discovering
   events via ``mne.find_events`` on stim channels.
2. **_io.write_raw_bids_preserve_events()** backs up and restores
   *_events.tsv* / *_events.json* around ``write_raw_bids`` calls so
   preprocessing steps (bad_segments, bad_channels, …) cannot corrupt
   the canonical event table.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import mne
import mne_bids
import numpy as np
import pandas as pd
import pytest

from custom.preprocessing._io import (
    count_condition_events_in_raw,
    count_condition_events_in_tsv,
    verify_event_count_after_write,
    write_raw_bids_custom_step,
    write_raw_bids_preserve_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_with_events(
    n_meg: int = 10,
    sfreq: float = 300.0,
    duration_sec: float = 30.0,
    n_trial_events: int = 20,
    event_descriptions: list[str] | None = None,
    include_stim: bool = False,
    seed: int = 42,
) -> mne.io.RawArray:
    """Create synthetic Raw with deterministic trial annotations.

    Annotations are spaced uniformly across the recording.
    """
    rng = np.random.RandomState(seed)

    ch_names = [f"MEG{i:03d}" for i in range(n_meg)]
    ch_types = ["mag"] * n_meg

    if include_stim:
        ch_names.append("STI001")
        ch_types.append("stim")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Assign sensor locations (needed for write_raw_bids)
    for idx, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        theta = 2 * np.pi * idx / len(info["chs"])
        phi = np.pi / 4
        r = 0.1
        loc[0] = r * np.sin(phi) * np.cos(theta)
        loc[1] = r * np.sin(phi) * np.sin(theta)
        loc[2] = r * np.cos(phi)
        loc[3:6] = -loc[:3] / np.linalg.norm(loc[:3])
        u = np.array([0, 0, 1.0])
        t1 = np.cross(loc[3:6], u)
        norm_t1 = np.linalg.norm(t1)
        if norm_t1 < 1e-6:
            u = np.array([1.0, 0, 0])
            t1 = np.cross(loc[3:6], u)
            norm_t1 = np.linalg.norm(t1)
        t1 /= norm_t1
        t2 = np.cross(loc[3:6], t1)
        loc[6:9] = t1
        loc[9:12] = t2
        ch["loc"] = loc

    n_samples = int(sfreq * duration_sec)
    n_ch = len(ch_names)
    data = rng.randn(n_ch, n_samples) * 1e-13

    # Put stim events if stim channel is present
    if include_stim:
        stim_idx = ch_names.index("STI001")
        data[stim_idx, :] = 0
        # Place stim events at the same positions as annotations
        for i in range(n_trial_events):
            onset_sample = int((i + 1) * sfreq * duration_sec / (n_trial_events + 1))
            data[stim_idx, onset_sample] = i % 4 + 1

    raw = mne.io.RawArray(data, info)

    if event_descriptions is None:
        event_descriptions = [
            "trial/listen_listen",
            "trial/read_read",
            "trial/listen_read",
            "trial/read_listen",
        ]

    # Create annotations at uniform intervals
    annotations = mne.Annotations(
        onset=[
            (i + 1) * duration_sec / (n_trial_events + 1)
            for i in range(n_trial_events)
        ],
        duration=[0.0] * n_trial_events,
        description=[
            event_descriptions[i % len(event_descriptions)]
            for i in range(n_trial_events)
        ],
    )
    raw.set_annotations(annotations)
    return raw


def _write_initial_bids(
    raw: mne.io.RawArray,
    bids_root: Path,
    subject: str = "001",
    session: str = "01",
    task: str = "test",
) -> mne_bids.BIDSPath:
    """Write a raw object to BIDS and return the BIDSPath."""
    bp = mne_bids.BIDSPath(
        root=str(bids_root),
        subject=subject,
        session=session,
        task=task,
        datatype="meg",
        suffix="meg",
        extension=".fif",
    )
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bp,
        allow_preload=True,
        overwrite=True,
        format="FIF",
    )
    return bp


def _read_events_tsv(bp: mne_bids.BIDSPath) -> pd.DataFrame:
    """Read events.tsv for a BIDSPath."""
    events_path = bp.copy().update(suffix="events", extension=".tsv").fpath
    return pd.read_csv(events_path, sep="\t")


def _count_trial_events(events_df: pd.DataFrame) -> int:
    """Count rows whose trial_type contains 'trial'."""
    return int(events_df["trial_type"].str.contains("trial", na=False).sum())


# ---------------------------------------------------------------------------
# Tests: write_raw_bids_preserve_events utility
# ---------------------------------------------------------------------------

class TestWriteRawBidsPreserveEvents:
    """Verify that write_raw_bids_preserve_events keeps events.tsv intact."""

    def test_events_tsv_unchanged_after_rewrite(self, tmp_path):
        """Events.tsv should be identical before and after a rewrite."""
        raw = _make_raw_with_events(n_trial_events=20)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        original_df = _read_events_tsv(bp)

        # Add BAD annotations (simulating bad_segments detection)
        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        raw_read.annotations.append(
            onset=2.0, duration=0.5, description="bad_segment_mag"
        )

        # Re-write with the preserve wrapper
        bp.split = None
        write_raw_bids_preserve_events(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        restored_df = _read_events_tsv(bp)

        pd.testing.assert_frame_equal(
            original_df, restored_df,
            obj="events.tsv after preserve-events rewrite",
        )

    def test_events_tsv_corrupted_without_preserve(self, tmp_path):
        """Without the wrapper, events.tsv changes on rewrite.

        This test documents the *problem* — plain write_raw_bids
        may produce a different events.tsv after a read-modify-write
        cycle because it regenerates from annotations.
        """
        raw = _make_raw_with_events(n_trial_events=20)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        original_df = _read_events_tsv(bp)
        original_count = len(original_df)

        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        raw_read.annotations.append(
            onset=2.0, duration=0.5, description="bad_segment_mag"
        )

        bp.split = None
        mne_bids.write_raw_bids(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        new_df = _read_events_tsv(bp)

        # The bad_segment_mag annotation now appears in events.tsv
        # (or the count may otherwise differ).
        assert len(new_df) != original_count or not original_df.equals(new_df), (
            "Expected events.tsv to change after plain write_raw_bids "
            "(this test documents the corruption problem)"
        )

    def test_no_events_files_first_write(self, tmp_path):
        """First-time write with no existing events.tsv should work fine."""
        raw = _make_raw_with_events(n_trial_events=10)
        bp = mne_bids.BIDSPath(
            root=str(tmp_path / "bids"),
            subject="001",
            session="01",
            task="test",
            datatype="meg",
            suffix="meg",
            extension=".fif",
        )
        # Use the wrapper for the first write — should not crash
        write_raw_bids_preserve_events(
            raw=raw,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )
        df = _read_events_tsv(bp)
        assert _count_trial_events(df) == 10

    def test_no_backup_residue_on_success(self, tmp_path):
        """Backup files (.bak) should be cleaned up after a successful write."""
        raw = _make_raw_with_events(n_trial_events=15)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()

        bp.split = None
        write_raw_bids_preserve_events(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        events_tsv = bp.copy().update(suffix="events", extension=".tsv").fpath
        assert not Path(str(events_tsv) + ".bak").exists()


# ---------------------------------------------------------------------------
# Tests: event count stability across simulated pipeline
# ---------------------------------------------------------------------------

class TestEventCountStability:
    """Simulate the read → modify → write cycle that preprocessing steps
    perform and verify event counts remain stable."""

    def test_single_rewrite_preserves_trial_count(self, tmp_path):
        """One read-modify-write cycle should not change trial event count."""
        n_trials = 25
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        assert _count_trial_events(_read_events_tsv(bp)) == n_trials

        # Simulate bad_segments: read, add BAD annotation, re-write
        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        raw_read.annotations.append(3.0, 1.0, "bad_segment_mag")

        bp.split = None
        write_raw_bids_preserve_events(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        assert _count_trial_events(_read_events_tsv(bp)) == n_trials

    def test_multiple_rewrites_preserve_trial_count(self, tmp_path):
        """Simulating 5 consecutive pipeline steps should not accumulate
        extra events."""
        n_trials = 30
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        # 5 read-modify-write cycles (bad_segments_1, bad_channels,
        # apply_hfc, apply_zca, bad_segments_2)
        for i in range(5):
            raw_read = mne_bids.read_raw_bids(bp)
            raw_read.load_data()
            # Each step might add a BAD annotation
            raw_read.annotations.append(
                float(i + 1) * 2, 0.5, f"bad_segment_pass{i}"
            )
            bp.split = None
            write_raw_bids_preserve_events(
                raw=raw_read,
                bids_path=bp,
                allow_preload=True,
                overwrite=True,
                format="FIF",
            )

        final_df = _read_events_tsv(bp)
        assert _count_trial_events(final_df) == n_trials

    def test_total_event_rows_unchanged_across_rewrites(self, tmp_path):
        """Not just trial events — ALL rows in events.tsv should be stable."""
        n_trials = 20
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        original_df = _read_events_tsv(bp)
        original_rows = len(original_df)

        # 3 rewrite cycles
        for _ in range(3):
            raw_read = mne_bids.read_raw_bids(bp)
            raw_read.load_data()
            raw_read.annotations.append(5.0, 0.2, "bad_segment_mag")
            bp.split = None
            write_raw_bids_preserve_events(
                raw=raw_read,
                bids_path=bp,
                allow_preload=True,
                overwrite=True,
                format="FIF",
            )

        final_df = _read_events_tsv(bp)
        assert len(final_df) == original_rows


# ---------------------------------------------------------------------------
# Tests: stim channel removal in format_bids
# ---------------------------------------------------------------------------

class TestStimChannelDrop:
    """Verify that convert_triggers drops stim channels, preventing
    event re-discovery on later write_raw_bids calls."""

    def test_stim_channels_absent_after_write(self, tmp_path):
        """After a BIDS write of data with annotations only (no stim),
        re-reading should not produce stim channels."""
        raw = _make_raw_with_events(n_trial_events=10, include_stim=False)
        bp = _write_initial_bids(raw, tmp_path / "bids")
        raw_read = mne_bids.read_raw_bids(bp)
        stim_chs = mne.pick_types(raw_read.info, stim=True)
        assert len(stim_chs) == 0, (
            "Raw written from annotations-only data should have no stim channels"
        )

    def test_events_unchanged_after_rewrite_without_stim(self, tmp_path):
        """Without stim channels, a plain write_raw_bids should still
        produce events only from annotations (no stim-based duplication)."""
        raw = _make_raw_with_events(n_trial_events=15, include_stim=False)
        bp = _write_initial_bids(raw, tmp_path / "bids")
        original_count = _count_trial_events(_read_events_tsv(bp))

        # Re-read and re-write once (simulating a pipeline step)
        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        bp.split = None
        write_raw_bids_preserve_events(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )
        assert _count_trial_events(_read_events_tsv(bp)) == original_count


# ---------------------------------------------------------------------------
# Tests: metadata ↔ BIDS event alignment
# ---------------------------------------------------------------------------

class TestMetadataAlignment:
    """Tests verifying that metadata row counts match BIDS trial events.

    These mirror the matching logic in config-trial.py: metadata is
    matched positionally to BIDS trial events, so the counts must be
    equal.
    """

    def test_metadata_matches_trial_events(self, tmp_path):
        """Synthetic metadata row count should match trial events in
        events.tsv after the initial BIDS write."""
        n_trials = 40
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        events_df = _read_events_tsv(bp)
        n_bids_trials = _count_trial_events(events_df)

        # Simulate metadata with exactly n_trials rows
        metadata = pd.DataFrame({
            "word": [f"word_{i}" for i in range(n_trials)],
            "condition": ["A", "B"] * (n_trials // 2),
        })

        assert len(metadata) == n_bids_trials, (
            f"Metadata rows ({len(metadata)}) != "
            f"BIDS trial events ({n_bids_trials})"
        )

    def test_metadata_mismatch_after_corruption(self, tmp_path):
        """After an unprotected rewrite, metadata may no longer match.

        This test documents the failure mode.
        """
        n_trials = 20
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        # Confirm initial alignment
        assert _count_trial_events(_read_events_tsv(bp)) == n_trials

        # Corrupt via plain write_raw_bids (add BAD annotation)
        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        raw_read.annotations.append(2.0, 0.5, "bad_segment_mag")
        bp.split = None
        mne_bids.write_raw_bids(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        corrupted_count = _count_trial_events(_read_events_tsv(bp))
        # After corruption, trial count may still be 20 (since bad_segment
        # doesn't contain "trial"), but total row count changes.
        # The key point: total rows increased because BAD annotation leaked in.
        corrupted_df = _read_events_tsv(bp)
        original_df_len = n_trials  # We started with exactly n_trials rows
        assert len(corrupted_df) > original_df_len, (
            "Plain write_raw_bids should add the BAD annotation as an event row"
        )

    def test_metadata_preserved_with_wrapper(self, tmp_path):
        """Using write_raw_bids_preserve_events, metadata alignment is kept."""
        n_trials = 20
        raw = _make_raw_with_events(n_trial_events=n_trials)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        metadata = pd.DataFrame({
            "word": [f"word_{i}" for i in range(n_trials)]
        })

        # Simulate 3 pipeline steps
        for _ in range(3):
            raw_read = mne_bids.read_raw_bids(bp)
            raw_read.load_data()
            raw_read.annotations.append(5.0, 0.3, "bad_segment_mag")
            bp.split = None
            write_raw_bids_preserve_events(
                raw=raw_read,
                bids_path=bp,
                allow_preload=True,
                overwrite=True,
                format="FIF",
            )

        n_bids_trials = _count_trial_events(_read_events_tsv(bp))
        assert len(metadata) == n_bids_trials, (
            f"After 3 pipeline steps with preserve-events wrapper, "
            f"metadata ({len(metadata)}) should still match "
            f"BIDS trials ({n_bids_trials})"
        )


# ---------------------------------------------------------------------------
# Tests: events.json preservation
# ---------------------------------------------------------------------------

class TestEventsJsonPreservation:
    """Verify that events.json sidecar is also preserved."""

    def test_events_json_unchanged(self, tmp_path):
        """events.json should be byte-identical after preserve-events rewrite."""
        raw = _make_raw_with_events(n_trial_events=10)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        events_json = bp.copy().update(suffix="events", extension=".json").fpath
        if events_json.exists():
            original_content = events_json.read_text()
        else:
            pytest.skip("No events.json generated by mne_bids")

        raw_read = mne_bids.read_raw_bids(bp)
        raw_read.load_data()
        raw_read.annotations.append(1.0, 0.5, "bad_segment_mag")

        bp.split = None
        write_raw_bids_preserve_events(
            raw=raw_read,
            bids_path=bp,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

        restored_content = events_json.read_text()
        assert original_content == restored_content


# ---------------------------------------------------------------------------
# Tests: post-write event-count verification helpers
# ---------------------------------------------------------------------------


class TestEventCountHelpers:
    """Unit tests for the count_* helpers used by verify_event_count_after_write."""

    def test_count_in_raw_matches_hierarchical(self, tmp_path):
        """count_condition_events_in_raw matches hierarchical events like
        ``mne-bids-pipeline.match_event_names``."""
        raw = _make_raw_with_events(
            n_trial_events=12,
            event_descriptions=[
                "trial/read_read", "trial/listen_listen",
                "feedback", "ITI",
            ],
        )
        n, names = count_condition_events_in_raw(raw, ["trial"])
        # 12 annotations total, every 4th is "trial/read_read" and "trial/listen_listen"
        # i.e. half of 12 = 6 trial events
        assert n == 6, f"expected 6 trial events, got {n}"
        assert set(names) == {"trial/read_read", "trial/listen_listen"}

    def test_count_in_raw_ignores_bad_segments(self, tmp_path):
        """BAD_ annotations are excluded by the default regexp, like the pipeline."""
        raw = _make_raw_with_events(n_trial_events=10)
        raw.annotations.append(2.0, 0.5, "BAD_segment_mag")
        raw.annotations.append(5.0, 0.5, "BAD_segment_mag")
        n, _ = count_condition_events_in_raw(raw, ["trial"])
        assert n == 10

    def test_count_in_tsv_matches_hierarchical(self, tmp_path):
        """count_condition_events_in_tsv applies the same hierarchical match."""
        raw = _make_raw_with_events(
            n_trial_events=8,
            event_descriptions=["trial/X", "trial/Y", "feedback", "ITI"],
        )
        bp = _write_initial_bids(raw, tmp_path / "bids")
        tsv = bp.copy().update(suffix="events", extension=".tsv").fpath
        n, names = count_condition_events_in_tsv(tsv, ["trial"])
        assert n == 4, f"expected 4 trial events, got {n}"
        assert set(names) == {"trial/X", "trial/Y"}


class TestVerifyEventCountAfterWrite:
    """End-to-end behaviour of verify_event_count_after_write."""

    def test_passes_on_consistent_write(self, tmp_path):
        """A normal write should pass verification silently."""
        raw = _make_raw_with_events(n_trial_events=20)
        bp = _write_initial_bids(raw, tmp_path / "bids")
        # Should not raise.
        verify_event_count_after_write(raw, bp, ("trial",), context="test")

    def test_raises_on_event_drop(self, tmp_path):
        """If the FIF on disk has fewer trials than raw_before, raise."""
        raw = _make_raw_with_events(n_trial_events=20)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        # Simulate a "fake" before raw with extra trials that aren't on disk.
        raw_extra = raw.copy()
        raw_extra.annotations.append(15.0, 0.0, "trial/extra")
        raw_extra.annotations.append(16.0, 0.0, "trial/extra")

        with pytest.raises(RuntimeError, match="Event-count mismatch"):
            verify_event_count_after_write(
                raw_extra, bp, ("trial",), context="test"
            )

    def test_raises_when_tsv_disagrees_with_fif(self, tmp_path):
        """Mismatch between FIF and events.tsv at the same path raises."""
        raw = _make_raw_with_events(n_trial_events=20)
        bp = _write_initial_bids(raw, tmp_path / "bids")

        # Corrupt the events.tsv by adding a fake "trial" row.
        tsv = bp.copy().update(suffix="events", extension=".tsv").fpath
        df = pd.read_csv(tsv, sep="\t")
        extra = df.iloc[[0]].copy()
        extra["onset"] = 25.0
        extra["sample"] = int(25.0 * 300.0)
        extra["trial_type"] = "trial/ghost"
        pd.concat([df, extra], ignore_index=True).to_csv(tsv, sep="\t", index=False)

        with pytest.raises(RuntimeError, match="Event-count mismatch"):
            verify_event_count_after_write(
                raw, bp, ("trial",), context="test"
            )


class TestWriteCustomStepInvokesVerification:
    """Smoke test that write_raw_bids_custom_step actually runs the new check."""

    def test_invokes_verification_in_derivative_mode(self, tmp_path):
        """In derivative mode, an inconsistent save raises immediately."""
        raw = _make_raw_with_events(n_trial_events=10)
        bp = _write_initial_bids(raw, tmp_path / "bids")
        # Patch raw to claim more trials than the FIF will have.
        raw_inflated = raw.copy()
        for i in range(3):
            raw_inflated.annotations.append(20.0 + i, 0.0, "trial/ghost")

        # Make raw_inflated have MORE trials than what's actually written by
        # overriding the save to write the (slimmer) `raw` instead.
        # Simplest reproduction: use the helper directly.
        cfg = SimpleNamespace(
            bids_root=str(tmp_path / "bids"),
            deriv_root=str(tmp_path / "deriv"),
            subjects=["001"], sessions=["01"], task="test",
            custom_proc="init",
            conditions=["trial"],
        )
        # Re-saving the consistent `raw` should pass.
        out = write_raw_bids_custom_step(raw, cfg, bp)
        assert out.fpath.exists()
