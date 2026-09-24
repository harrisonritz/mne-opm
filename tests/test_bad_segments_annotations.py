"""Regression tests for transferring OSL annotations to unfiltered raw data."""

from types import SimpleNamespace

import mne
import numpy as np

from custom.preprocessing import bad_segments


def _raw():
    info = mne.create_info(["MAG001", "MAG002", "MAG003"], 100, "mag")
    raw = mne.io.RawArray(np.zeros((3, 700)), info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations([1.0, 3.0, 5.0], [0.0] * 3, ["trial"] * 3)
    )
    return raw


def _analysis(*, find_breaks=False):
    cfg = SimpleNamespace(
        _bad_segments_stage="1",
        l_freq=1.0,
        h_freq=30.0,
        ch_types=["mag"],
        find_breaks=False,
        _bad_segments_find_breaks=find_breaks,
        min_break_duration=1.0,
        t_break_annot_start_after_previous_event=0.1,
        t_break_annot_stop_before_next_event=0.1,
    )
    return bad_segments.BadSegmentsAnalysis(cfg)


def test_interleaved_bad_segment_is_transferred(monkeypatch):
    def fake_osl(filt, **kwargs):
        filt.annotations.append(2.0, 0.5, "bad_segment_mag")
        return filt

    monkeypatch.setattr(bad_segments, "osl_bad_segments", fake_osl)
    result = _analysis()._detect_bad_segments(_raw())

    assert list(result.annotations.description).count("bad_segment_mag") == 1
    assert list(result.annotations.description).count("trial") == 3
    bad_idx = list(result.annotations.description).index("bad_segment_mag")
    assert result.annotations.onset[bad_idx] == 2.0


def test_break_annotations_are_retained_before_detection(monkeypatch):
    def fake_break(raw, **kwargs):
        return mne.Annotations([2.0], [0.5], ["BAD_break"])

    def fake_osl(filt, **kwargs):
        assert "BAD_break" in filt.annotations.description
        return filt

    monkeypatch.setattr(mne.preprocessing, "annotate_break", fake_break)
    monkeypatch.setattr(bad_segments, "osl_bad_segments", fake_osl)
    result = _analysis(find_breaks=True)._detect_bad_segments(_raw())

    assert list(result.annotations.description).count("BAD_break") == 1
