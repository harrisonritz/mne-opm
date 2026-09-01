"""Tests for the custom osl-ephys preprocessing wrappers."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from custom.osl.extra_funcs import (
    DEFAULT_EXCLUDE_PREFIXES,
    PREPROC_EXTRA_FUNCS,
    _channel_sds,
    bad_channels_clean,
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


@pytest.fixture
def raw_with_a_noisy_channel(raw_meg):
    """Raw where one channel is steadily noisy and another has one transient.

    The transient is annotated, the way the ``bad_segments`` steps ahead of
    this one annotate theirs.  osl-ephys' detector measures the transient and
    misses the steadily-noisy channel; measuring the un-annotated data only
    reverses that.
    """
    raw = raw_meg.copy()
    picks = mne.pick_types(raw.info, meg=True)
    data = raw.get_data()

    steady = picks[0]
    data[steady] *= 12.0

    spiky = picks[1]
    onset, duration = 2.0, 0.5
    start = int(onset * raw.info["sfreq"])
    stop = int((onset + duration) * raw.info["sfreq"])
    data[spiky, start:stop] *= 400.0

    out = mne.io.RawArray(data, raw.info.copy(), verbose="error")
    out.set_annotations(
        mne.Annotations(onset=[onset], duration=[duration], description=["bad_segment_mag"])
    )
    return out, out.ch_names[steady], out.ch_names[spiky]


class TestBadChannelsClean:
    """The GESD metric has to ignore what bad_segments already annotated."""

    def test_the_steadily_noisy_channel_is_found(self, raw_with_a_noisy_channel):
        raw, steady, _ = raw_with_a_noisy_channel
        bad_channels_clean({"raw": raw}, {"picks": "mag"})
        assert steady in raw.info["bads"]

    def test_keeping_the_annotated_segment_hides_it(self, raw_with_a_noisy_channel):
        # This is osl-ephys' behaviour, and the bug being worked around: the
        # annotated transient dominates the SDs and masks the noisy channel.
        raw, steady, spiky = raw_with_a_noisy_channel
        bad_channels_clean(
            {"raw": raw}, {"picks": "mag", "reject_by_annotation": None}
        )
        assert spiky in raw.info["bads"]
        assert steady not in raw.info["bads"]

    def test_channels_already_marked_bad_are_not_retested(
        self, raw_with_a_noisy_channel
    ):
        raw, steady, _ = raw_with_a_noisy_channel
        raw.info["bads"] = [steady]
        bad_channels_clean({"raw": raw}, {"picks": "mag"})
        assert raw.info["bads"].count(steady) == 1

    def test_the_dataset_is_returned_for_the_next_step(
        self, raw_with_a_noisy_channel
    ):
        raw, _, _ = raw_with_a_noisy_channel
        dataset = {"raw": raw}
        assert bad_channels_clean(dataset, {"picks": "mag"}) is dataset

    def test_an_unknown_channel_type_is_rejected(self, raw_meg):
        with pytest.raises(ValueError, match="picks"):
            bad_channels_clean({"raw": raw_meg}, {"picks": "seeg"})

    def test_a_missing_picks_is_rejected(self, raw_meg):
        with pytest.raises(ValueError, match="picks"):
            bad_channels_clean({"raw": raw_meg}, {})

    def test_unknown_options_are_rejected(self, raw_meg):
        # A typo in the YAML should fail here, not be silently ignored.
        with pytest.raises(ValueError, match="significance_leve"):
            bad_channels_clean(
                {"raw": raw_meg}, {"picks": "mag", "significance_leve": 0.05}
            )

    def test_raises_when_everything_is_annotated_bad(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations(
                onset=[0.0],
                duration=[raw.times[-1] + 1.0],
                description=["bad_segment_mag"],
            )
        )
        with pytest.raises(ValueError, match="every sample"):
            bad_channels_clean({"raw": raw}, {"picks": "mag"})


class TestChannelSds:
    """The streamed metric has to equal the one-shot one it replaces."""

    def test_it_matches_np_std_whatever_the_block_size(self, raw_meg):
        picks = mne.pick_types(raw_meg.info, meg=True)
        expected = raw_meg.get_data(picks=picks).std(axis=1)
        for block in (raw_meg.n_times, 1000, 137):
            sd, n_kept = _channel_sds(raw_meg, picks, None, block=block)
            assert n_kept == raw_meg.n_times
            assert np.allclose(sd, expected, rtol=1e-9)

    def test_omitting_annotations_matches_get_data(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations([2.0], [1.5], ["bad_segment_mag"])
        )
        picks = mne.pick_types(raw.info, meg=True)
        expected = raw.get_data(picks=picks, reject_by_annotation="omit")

        sd, n_kept = _channel_sds(raw, picks, "omit", block=500)

        assert n_kept == expected.shape[1] < raw.n_times
        assert np.allclose(sd, expected.std(axis=1), rtol=1e-6)

    def test_a_fully_annotated_recording_keeps_nothing(self, raw_meg):
        raw = raw_meg.copy()
        raw.set_annotations(
            mne.Annotations([0.0], [raw.times[-1] + 1.0], ["bad_segment_mag"])
        )
        picks = mne.pick_types(raw.info, meg=True)
        sd, n_kept = _channel_sds(raw, picks, "omit")
        assert n_kept == 0
        assert not sd.any()


class TestRegistry:
    def test_the_step_is_registered_under_its_config_name(self):
        # osl-ephys matches extra_funcs by __name__, so the function name is
        # the config key.
        assert "events_from_annotations" in [
            f.__name__ for f in PREPROC_EXTRA_FUNCS
        ]

    def test_bad_channels_clean_is_registered(self):
        assert "bad_channels_clean" in [f.__name__ for f in PREPROC_EXTRA_FUNCS]

    def test_default_exclude_prefixes(self):
        assert DEFAULT_EXCLUDE_PREFIXES == ("BAD", "EDGE")
