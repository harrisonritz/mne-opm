"""Tests for :func:`custom.preprocessing._io.trial_following_event`.

The helper pairs each locking event (e.g. the TSX switch cue, ``CSI``) with the
first trial annotation after it, so an event-locked config can select the
per-trial metadata row of the trial each event precedes.  The positional join
is only safe when every event has its own following trial, so the helper must
refuse orphaned and doubled events rather than pair them silently.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from custom.preprocessing._io import (
    count_condition_events_in_raw,
    trial_following_event,
)


def _raw_from_events(events, sfreq: float = 100.0, pad: float = 1.0):
    """Tiny RawArray carrying ``(onset, description)`` annotations."""
    onsets = [float(o) for o, _ in events]
    n_samples = int(sfreq * (max(onsets) + pad)) + 1
    info = mne.create_info(["MEG001"], sfreq, ["mag"])
    raw = mne.io.RawArray(np.zeros((1, n_samples)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onsets, [0.0] * len(onsets), [d for _, d in events])
    )
    return raw


# Two mini-blocks: pre trials, cue, post trial.  Responses interleaved.
_EVENTS = [
    (1.0, "trial/read_listen"),
    (1.6, "response/left"),
    (3.5, "trial/read_listen"),
    (4.0, "response/right"),
    (6.0, "CSI"),
    (6.8, "trial/read_listen"),
    (7.4, "response/right"),
    (10.0, "trial/listen_listen"),
    (12.5, "CSI"),
    (13.3, "trial/listen_listen"),
    (14.0, "BAD_break"),
]


def test_pairs_each_event_with_next_trial():
    raw = _raw_from_events(_EVENTS)
    trial_idx, lags = trial_following_event(
        raw, event_conditions=("CSI",), trial_conditions=("trial",)
    )
    np.testing.assert_array_equal(trial_idx, [2, 4])
    np.testing.assert_allclose(lags, [0.8, 0.8])


def test_length_matches_pipeline_event_count():
    raw = _raw_from_events(_EVENTS)
    trial_idx, _ = trial_following_event(raw, event_conditions=("CSI",))
    n_events, _ = count_condition_events_in_raw(raw, ("CSI",))
    assert len(trial_idx) == n_events == 2


def test_annotation_order_does_not_matter():
    raw = _raw_from_events(list(reversed(_EVENTS)))
    trial_idx, lags = trial_following_event(raw, event_conditions=("CSI",))
    np.testing.assert_array_equal(trial_idx, [2, 4])
    np.testing.assert_allclose(lags, [0.8, 0.8])


def test_event_without_later_trial_raises():
    raw = _raw_from_events(_EVENTS + [(20.0, "CSI")])
    with pytest.raises(RuntimeError, match="no later trial"):
        trial_following_event(raw, event_conditions=("CSI",))


def test_two_events_sharing_a_trial_raise():
    raw = _raw_from_events(_EVENTS + [(12.9, "CSI")])
    with pytest.raises(RuntimeError, match="no trial between them"):
        trial_following_event(raw, event_conditions=("CSI",))


def test_event_matching_trial_conditions_raises():
    raw = _raw_from_events(_EVENTS)
    with pytest.raises(ValueError, match="match both"):
        trial_following_event(
            raw, event_conditions=("trial",), trial_conditions=("trial",)
        )


def test_no_events_returns_empty():
    raw = _raw_from_events([e for e in _EVENTS if e[1] != "CSI"])
    trial_idx, lags = trial_following_event(raw, event_conditions=("CSI",))
    assert trial_idx.shape == (0,) and lags.shape == (0,)
