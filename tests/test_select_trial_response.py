"""Tests for the trial-paired response selection used by the response-locked
``config-trialResponse.py`` analysis.

Covers the two shared helpers in :mod:`custom.preprocessing._io`:

* :func:`first_response_per_trial` — pairs each trial annotation with the first
  response in its window ``[trial_onset, next_trial_onset)``, flagging extra
  presses and orphan responses for removal.
* :func:`drop_response_rows_from_events_tsv` — trims a BIDS events.tsv to the
  kept response onsets, leaving non-response rows untouched.

Together these guarantee that, after the ``select_trial_response`` step runs,
the number of response epochs equals the number of responded trials so the
per-trial metadata aligns 1:1.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace

import mne
import mne_bids
import numpy as np
import pandas as pd

from custom.preprocessing._io import (
    count_condition_events_in_raw,
    count_condition_events_in_tsv,
    drop_response_rows_from_events_tsv,
    first_response_per_trial,
    write_raw_bids_custom_step,
)
from custom.preprocessing.select_trial_response import (
    SelectTrialResponseAnalysis,
)
from custom.preprocessing.select_trial_response import run as select_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_from_events(events, sfreq: float = 300.0, pad: float = 1.0):
    """Build a tiny RawArray carrying the given ``(onset, description)`` events.

    Parameters
    ----------
    events : list of (float, str)
        Annotation onsets (seconds) and descriptions, in any order.
    sfreq : float
        Sampling frequency.
    pad : float
        Extra seconds appended after the last onset so all events fit.
    """
    onsets = [float(o) for o, _ in events]
    descs = [str(d) for _, d in events]
    duration_sec = (max(onsets) if onsets else 0.0) + pad
    n_samples = int(sfreq * duration_sec) + 1
    info = mne.create_info(["MEG001"], sfreq, ["mag"])
    raw = mne.io.RawArray(np.zeros((1, n_samples)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=[0.0] * len(onsets),
            description=descs,
        )
    )
    return raw


def _onsets_at(raw, idx):
    """Return the annotation onsets at the given indices (sorted ascending)."""
    return sorted(float(raw.annotations.onset[i]) for i in idx)


def _write_events_tsv(path, rows):
    """Write a minimal BIDS events.tsv from ``rows`` (list of dicts)."""
    df = pd.DataFrame(rows, columns=["onset", "duration", "trial_type", "value", "sample"])
    df.to_csv(path, sep="\t", index=False)
    return path


def _make_meg_info(n_meg: int = 10, sfreq: float = 300.0) -> mne.Info:
    """Minimal MEG Info with sensor locations (needed by write_raw_bids)."""
    info = mne.create_info(
        [f"MEG{i:03d}" for i in range(n_meg)], sfreq, ["mag"] * n_meg
    )
    for idx, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        theta = 2 * np.pi * idx / n_meg
        r = 0.1
        loc[0] = r * np.sin(np.pi / 4) * np.cos(theta)
        loc[1] = r * np.sin(np.pi / 4) * np.sin(theta)
        loc[2] = r * np.cos(np.pi / 4)
        loc[3:6] = -loc[:3] / np.linalg.norm(loc[:3])
        t1 = np.cross(loc[3:6], np.array([0, 0, 1.0]))
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(loc[3:6], np.array([1.0, 0, 0]))
        t1 /= np.linalg.norm(t1)
        loc[6:9] = t1
        loc[9:12] = np.cross(loc[3:6], t1)
        ch["loc"] = loc
    info.set_meas_date(None)
    return info


# ---------------------------------------------------------------------------
# first_response_per_trial
# ---------------------------------------------------------------------------

class TestFirstResponsePerTrial:
    """Per-trial first-response pairing."""

    def test_one_response_per_trial(self):
        raw = _raw_from_events([
            (1.0, "trial/a"), (1.5, "response/left"),
            (3.0, "trial/b"), (3.4, "response/right"),
            (5.0, "trial/a"), (5.2, "response/left"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True, True, True]
        assert len(keep) == 3
        assert drop == []
        np.testing.assert_allclose(sorted(keep_onsets), [1.5, 3.4, 5.2])

    def test_multiple_presses_keep_first(self):
        raw = _raw_from_events([
            (1.0, "trial/a"), (1.5, "response/left"),
            (3.0, "trial/b"), (3.4, "response/right"), (3.6, "response/left"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True, True]
        # The first response in trial b (3.4) is kept; the second (3.6) dropped.
        np.testing.assert_allclose(sorted(keep_onsets), [1.5, 3.4])
        assert _onsets_at(raw, drop) == [3.6]

    def test_no_response_trial_is_false_and_does_not_steal_next(self):
        raw = _raw_from_events([
            (1.0, "trial/a"), (1.5, "response/left"),
            (3.0, "trial/b"),                       # no response in [3.0, 5.0)
            (5.0, "trial/c"), (5.2, "response/right"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True, False, True]
        np.testing.assert_allclose(sorted(keep_onsets), [1.5, 5.2])
        assert drop == []

    def test_orphan_response_before_first_trial_is_dropped(self):
        raw = _raw_from_events([
            (0.5, "response/left"),                 # orphan: precedes all trials
            (1.0, "trial/a"), (1.5, "response/right"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True]
        np.testing.assert_allclose(sorted(keep_onsets), [1.5])
        assert _onsets_at(raw, drop) == [0.5]

    def test_last_trial_response_has_no_upper_bound(self):
        raw = _raw_from_events([
            (1.0, "trial/a"), (10.0, "response/left"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True]
        np.testing.assert_allclose(sorted(keep_onsets), [10.0])
        assert drop == []

    def test_hierarchical_trial_matching(self):
        # 'trial' must match 'trial/read_read'; non-trial labels ignored.
        raw = _raw_from_events([
            (1.0, "CSI"),
            (1.2, "trial/read_read"), (1.6, "response/right"),
            (2.0, "ITI"),
            (3.0, "trial/listen_read"), (3.3, "response/left"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == [True, True]
        np.testing.assert_allclose(sorted(keep_onsets), [1.6, 3.3])
        assert drop == []

    def test_kept_count_matches_response_count_after_drop(self):
        # Mirrors the config invariant: after dropping, the FIF's response count
        # equals the number of responded trials (sum of the mask).
        raw = _raw_from_events([
            (1.0, "trial/a"), (1.5, "response/left"),
            (3.0, "trial/b"), (3.4, "response/right"), (3.6, "response/right"),
            (5.0, "trial/c"),
            (7.0, "trial/d"), (7.1, "response/left"),
        ])
        mask, keep, drop, _ = first_response_per_trial(raw)

        # Apply the drop and recount.
        ann = raw.annotations
        keep_mask = np.array([i not in set(drop) for i in range(len(ann))])
        raw.set_annotations(mne.Annotations(
            onset=ann.onset[keep_mask],
            duration=ann.duration[keep_mask],
            description=ann.description[keep_mask],
            orig_time=ann.orig_time,
        ))
        n_resp, _ = count_condition_events_in_raw(
            raw, ("response/left", "response/right")
        )
        assert n_resp == int(mask.sum()) == len(keep)

    def test_no_trials_drops_all_responses_as_orphans(self):
        raw = _raw_from_events([
            (1.0, "response/left"), (2.0, "response/right"),
        ])
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)

        assert mask.tolist() == []
        assert keep == []
        assert len(keep_onsets) == 0
        assert _onsets_at(raw, drop) == [1.0, 2.0]

    def test_empty_annotations(self):
        raw = _raw_from_events([(1.0, "trial/a")])
        raw.set_annotations(mne.Annotations([], [], []))
        mask, keep, drop, keep_onsets = first_response_per_trial(raw)
        assert mask.tolist() == []
        assert keep == [] and drop == []
        assert len(keep_onsets) == 0


# ---------------------------------------------------------------------------
# drop_response_rows_from_events_tsv
# ---------------------------------------------------------------------------

class TestDropResponseRowsFromEventsTsv:
    """Trimming response rows from a BIDS events.tsv."""

    def _base_rows(self):
        return [
            {"onset": 1.0, "duration": 0, "trial_type": "trial/a", "value": 9, "sample": 300},
            {"onset": 1.5, "duration": 0, "trial_type": "response/left", "value": 5, "sample": 450},
            {"onset": 3.0, "duration": 0, "trial_type": "trial/b", "value": 12, "sample": 900},
            {"onset": 3.4, "duration": 0, "trial_type": "response/right", "value": 6, "sample": 1020},
            {"onset": 3.6, "duration": 0, "trial_type": "response/left", "value": 5, "sample": 1080},
        ]

    def test_drops_unkept_response_rows(self, tmp_path):
        tsv = _write_events_tsv(tmp_path / "events.tsv", self._base_rows())
        n_removed = drop_response_rows_from_events_tsv(tsv, keep_onsets=[1.5, 3.4])
        assert n_removed == 1

        df = pd.read_csv(tsv, sep="\t")
        resp = df[df["trial_type"].str.startswith("response")]
        trials = df[df["trial_type"].str.startswith("trial")]
        # Only the two kept responses remain; both trials untouched.
        np.testing.assert_allclose(sorted(resp["onset"]), [1.5, 3.4])
        assert len(trials) == 2

    def test_keep_none_removes_all_responses_only(self, tmp_path):
        tsv = _write_events_tsv(tmp_path / "events.tsv", self._base_rows())
        n_removed = drop_response_rows_from_events_tsv(tsv, keep_onsets=[])
        assert n_removed == 3

        df = pd.read_csv(tsv, sep="\t")
        assert not df["trial_type"].str.startswith("response").any()
        assert df["trial_type"].str.startswith("trial").sum() == 2

    def test_no_response_rows_is_noop(self, tmp_path):
        rows = [
            {"onset": 1.0, "duration": 0, "trial_type": "trial/a", "value": 9, "sample": 300},
            {"onset": 2.0, "duration": 0, "trial_type": "ITI", "value": 2, "sample": 600},
        ]
        tsv = _write_events_tsv(tmp_path / "events.tsv", rows)
        before = (tmp_path / "events.tsv").read_text()
        n_removed = drop_response_rows_from_events_tsv(tsv, keep_onsets=[1.5])
        assert n_removed == 0
        assert (tmp_path / "events.tsv").read_text() == before

    def test_missing_file_returns_zero(self, tmp_path):
        assert drop_response_rows_from_events_tsv(
            tmp_path / "does_not_exist.tsv", keep_onsets=[1.0]
        ) == 0

    def test_tolerance_matching(self, tmp_path):
        tsv = _write_events_tsv(tmp_path / "events.tsv", self._base_rows())
        # Kept onsets within tol of the TSV onsets still claim those rows.
        n_removed = drop_response_rows_from_events_tsv(
            tsv, keep_onsets=[1.51, 3.41], tol_sec=0.02
        )
        assert n_removed == 1
        df = pd.read_csv(tsv, sep="\t")
        resp = df[df["trial_type"].str.startswith("response")]
        np.testing.assert_allclose(sorted(resp["onset"]), [1.5, 3.4])


# ---------------------------------------------------------------------------
# End-to-end: SelectTrialResponseAnalysis through the real write guards
# ---------------------------------------------------------------------------

class TestSelectTrialResponseStep:
    """Full round-trip: seed a proc-init derivative, run the step, and verify
    the FIF and events.tsv are reduced consistently (so write_raw_bids_custom_step's
    response-count trim/verify guards pass)."""

    SFREQ = 300.0

    def _make_raw(self):
        """5 trials; trial 2 double-presses, trial 3 has no response."""
        info = _make_meg_info(n_meg=10, sfreq=self.SFREQ)
        n_samples = int(self.SFREQ * 11.0)
        raw = mne.io.RawArray(
            np.random.RandomState(0).randn(10, n_samples) * 1e-13, info
        )
        events = [
            (1.0, "trial/read_read"), (1.4, "response/left"),
            (3.0, "trial/listen_read"), (3.4, "response/right"), (3.6, "response/right"),
            (5.0, "trial/read_listen"),                          # no response
            (7.0, "trial/listen_listen"), (7.2, "response/left"),
            (9.0, "trial/read_read"), (9.3, "response/right"),
        ]
        raw.set_annotations(mne.Annotations(
            onset=[o for o, _ in events],
            duration=[0.0] * len(events),
            description=[d for _, d in events],
        ))
        return raw

    def _cfg(self, tmp_path):
        return SimpleNamespace(
            bids_root=str(tmp_path / "bids"),
            deriv_root=str(tmp_path / "deriv"),
            subjects=["001"],
            sessions=["01"],
            task="test",
            custom_proc="init",
            conditions=["response/left", "response/right"],
            _select_trial_response=True,
            process_empty_room=False,
        )

    def _seed_proc_init(self, raw, cfg):
        """Write source BIDS, then seed a proc-init derivative (as the first
        custom step would), returning nothing — leaves derivative on disk."""
        source_bp = mne_bids.BIDSPath(
            root=cfg.bids_root, subject="001", session="01", task="test",
            datatype="meg", suffix="meg", extension=".fif",
        )
        mne_bids.write_raw_bids(
            raw=raw, bids_path=source_bp, allow_preload=True,
            overwrite=True, format="FIF",
        )
        # Seeds proc-init FIF + events.tsv (all responses) and passes guards.
        write_raw_bids_custom_step(raw, cfg, source_bp)

    def test_end_to_end_reduces_to_first_response_per_trial(self, tmp_path):
        raw = self._make_raw()
        cfg = self._cfg(tmp_path)
        self._seed_proc_init(raw, cfg)

        # Sanity: the seeded derivative carries all 5 responses.
        deriv_fif = list(
            (tmp_path / "deriv").glob("sub-001/**/*proc-init_raw.fif")
        )
        assert deriv_fif, "proc-init derivative not seeded"
        raw_seed = mne.io.read_raw_fif(deriv_fif[0], verbose="ERROR")
        n_resp_before, _ = count_condition_events_in_raw(
            raw_seed, cfg.conditions
        )
        assert n_resp_before == 5

        # Run the step (no exception => write guards passed).
        select_run(cfg)

        # FIF now has one response per responded trial (4), trials unchanged (5).
        raw_after = mne.io.read_raw_fif(deriv_fif[0], verbose="ERROR")
        n_resp_after, _ = count_condition_events_in_raw(raw_after, cfg.conditions)
        n_trial_after, _ = count_condition_events_in_raw(raw_after, ("trial",))
        assert n_resp_after == 4
        assert n_trial_after == 5

        # The kept responses are the first within each responded trial window.
        resp_onsets = sorted(
            float(o) for o, d in zip(raw_after.annotations.onset,
                                     raw_after.annotations.description)
            if d.startswith("response")
        )
        np.testing.assert_allclose(resp_onsets, [1.4, 3.4, 7.2, 9.3])

        # events.tsv stays consistent with the FIF (guards depend on this).
        events_tsv = deriv_fif[0].parent / (
            deriv_fif[0].name.split("_proc-")[0]
            + "_proc-init_events.tsv"
        )
        n_resp_tsv, _ = count_condition_events_in_tsv(events_tsv, cfg.conditions)
        assert n_resp_tsv == 4

        # Config-side mask (re-derived from the reduced FIF) matches the epochs.
        mask, _, _, _ = first_response_per_trial(raw_after)
        assert int(mask.sum()) == n_resp_after == 4

    def test_disabled_is_noop(self, tmp_path):
        raw = self._make_raw()
        cfg = self._cfg(tmp_path)
        cfg._select_trial_response = False
        self._seed_proc_init(raw, cfg)

        deriv_fif = list(
            (tmp_path / "deriv").glob("sub-001/**/*proc-init_raw.fif")
        )[0]
        before, _ = count_condition_events_in_raw(
            mne.io.read_raw_fif(deriv_fif, verbose="ERROR"), cfg.conditions
        )

        select_run(cfg)  # gated off → no change

        after, _ = count_condition_events_in_raw(
            mne.io.read_raw_fif(deriv_fif, verbose="ERROR"), cfg.conditions
        )
        assert before == after == 5
