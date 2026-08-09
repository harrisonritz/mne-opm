"""Tests for the custom osl-ephys preprocessing wrappers."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from custom.osl.extra_funcs import (
    DEFAULT_EXCLUDE_PREFIXES,
    PREPROC_EXTRA_FUNCS,
    events_from_annotations,
)


TRIGGER_CODES = {
    "trial/read_read": 9,
    "trial/listen_listen": 17,
    "response/left": 201,
    "response/right": 202,
}


@pytest.fixture
def annotated_raw(raw_meg):
    """Raw data annotated like a BIDS file written by format_bids."""
    raw = raw_meg.copy()
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 1.0, 1.5, 2.0, 2.5],
            duration=[0.0] * 5,
            description=[
                "trial/read_read",
                "response/left",
                "trial/listen_listen",
                "response/right",
                "trial/read_read",
            ],
        )
    )
    return raw


def make_dataset(raw, event_id=None):
    """Build the dataset dict osl-ephys hands to a preprocessing step."""
    return {
        "raw": raw,
        "events": None,
        "epochs": None,
        "event_id": event_id,
        "ica": None,
        "fig": {},
    }


class TestEventsFromAnnotations:
    def test_uses_the_configured_event_codes(self, annotated_raw):
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        result = events_from_annotations(dataset, {})

        assert result["events"].shape == (5, 3)
        assert result["event_id"] == TRIGGER_CODES
        # Codes come from meta.event_codes, not from the annotation order.
        assert sorted(np.unique(result["events"][:, 2])) == [9, 17, 201, 202]

    def test_events_are_ordered_by_sample(self, annotated_raw):
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        events = events_from_annotations(dataset, {})["events"]
        assert np.all(np.diff(events[:, 0]) > 0)

    def test_derives_codes_when_none_are_configured(self, annotated_raw):
        dataset = make_dataset(annotated_raw, None)
        result = events_from_annotations(dataset, {})

        assert set(result["event_id"]) == {
            "trial/read_read",
            "trial/listen_listen",
            "response/left",
            "response/right",
        }
        assert len(result["events"]) == 5

    def test_userargs_event_id_overrides_the_dataset(self, annotated_raw):
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        result = events_from_annotations(
            dataset, {"event_id": {"response/left": 7}}
        )
        assert result["event_id"] == {"response/left": 7}
        assert len(result["events"]) == 1

    def test_drops_configured_codes_with_no_annotations(self, annotated_raw):
        # A subject missing a condition must not break epoching: mne.Epochs
        # raises on an event_id entry that matches no event.
        codes = {**TRIGGER_CODES, "trial/never_happened": 55}
        dataset = make_dataset(annotated_raw, codes)
        result = events_from_annotations(dataset, {})

        assert "trial/never_happened" not in result["event_id"]
        assert result["event_id"] == TRIGGER_CODES

    def test_strict_raises_on_a_missing_code(self, annotated_raw):
        codes = {**TRIGGER_CODES, "trial/never_happened": 55}
        dataset = make_dataset(annotated_raw, codes)
        with pytest.raises(ValueError, match="never_happened"):
            events_from_annotations(dataset, {"strict": True})

    def test_regexp_selects_a_subset(self, annotated_raw):
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        result = events_from_annotations(dataset, {"regexp": "response"})

        assert set(result["event_id"]) == {"response/left", "response/right"}
        assert len(result["events"]) == 2

    def test_excludes_artefact_annotations_when_deriving_codes(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations(
                onset=[0.5, 1.0, 1.5],
                duration=[0.0, 0.2, 0.0],
                description=["trial/read_read", "BAD_segment", "EDGE boundary"],
            )
        )
        dataset = make_dataset(raw, None)
        result = events_from_annotations(dataset, {})

        assert set(result["event_id"]) == {"trial/read_read"}

    def test_exclude_prefixes_are_configurable(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations(
                onset=[0.5, 1.0],
                duration=[0.0, 0.0],
                description=["trial/read_read", "ignoreme"],
            )
        )
        dataset = make_dataset(raw, None)
        result = events_from_annotations(dataset, {"exclude": ["ignore"]})

        assert set(result["event_id"]) == {"trial/read_read"}

    def test_explicit_codes_ignore_the_exclude_prefixes(self, raw_meg):
        # An explicit mapping is taken at face value: if someone deliberately
        # maps a BAD_ annotation, honour it.
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations(
                onset=[0.5], duration=[0.0], description=["BAD_deliberate"]
            )
        )
        dataset = make_dataset(raw, {"BAD_deliberate": 3})
        result = events_from_annotations(dataset, {})

        assert result["event_id"] == {"BAD_deliberate": 3}

    def test_raises_without_annotations(self, raw_meg):
        dataset = make_dataset(raw_meg.copy(), TRIGGER_CODES)
        with pytest.raises(ValueError, match="no annotations"):
            events_from_annotations(dataset, {})

    def test_raises_when_no_configured_code_matches(self, annotated_raw):
        dataset = make_dataset(annotated_raw, {"nothing/here": 1})
        with pytest.raises(ValueError, match="none of the configured event codes"):
            events_from_annotations(dataset, {})

    def test_raises_when_every_description_is_excluded(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations(onset=[0.5], duration=[0.0], description=["BAD_only"])
        )
        dataset = make_dataset(raw, None)
        with pytest.raises(ValueError, match="excluded by prefixes"):
            events_from_annotations(dataset, {})

    def test_raises_when_the_regexp_matches_nothing(self, annotated_raw):
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        with pytest.raises(ValueError):
            events_from_annotations(dataset, {"regexp": "nomatch"})

    def test_output_epochs_correctly(self, annotated_raw):
        # The point of the step: the events it produces must build epochs.
        dataset = make_dataset(annotated_raw, TRIGGER_CODES)
        result = events_from_annotations(dataset, {})

        epochs = mne.Epochs(
            result["raw"],
            result["events"],
            result["event_id"],
            tmin=-0.1,
            tmax=0.1,
            baseline=None,
            preload=True,
        )
        assert len(epochs) == 5
        assert set(epochs.event_id) == set(TRIGGER_CODES)


class TestRegistry:
    def test_the_step_is_registered_under_its_config_name(self):
        # osl-ephys matches extra_funcs by __name__, so the function name is
        # the config key.
        assert "events_from_annotations" in [
            f.__name__ for f in PREPROC_EXTRA_FUNCS
        ]

    def test_default_exclude_prefixes(self):
        assert DEFAULT_EXCLUDE_PREFIXES == ("BAD", "EDGE")
