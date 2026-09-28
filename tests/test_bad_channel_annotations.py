"""All configured channel scores must omit BAD-annotated samples."""

from types import SimpleNamespace

import mne
import numpy as np

from custom.preprocessing.bad_channels import (
    BadChannelsAnalysis,
    _mask_short_filter_segments,
)


def test_lof_receives_only_unannotated_samples(monkeypatch):
    rng = np.random.default_rng(4)
    data = rng.normal(size=(4, 3600))
    data[:, 1200:1800] = 1000.0
    raw = mne.io.RawArray(
        data,
        mne.create_info(["M1", "M2", "M3", "M4"], 600, "mag"),
        verbose="ERROR",
    )
    raw.set_annotations(mne.Annotations([2.0], [1.0], ["bad_segment_mag"]))
    cfg = SimpleNamespace(
        ch_types=["mag"],
        _channel_metrics=["lof"],
        _bad_channel_lfreq=1.0,
        _bad_channel_hfreq=90.0,
    )
    analysis = BadChannelsAnalysis(cfg)
    seen = {}

    def capture_lof(clean_data):
        seen["data"] = clean_data.copy()
        return np.arange(1, 5, dtype=float)

    monkeypatch.setattr(analysis, "_lof_scores", capture_lof)
    analysis._compute_channel_metrics(raw)

    assert seen["data"].shape == (4, 3000)
    assert np.max(np.abs(seen["data"])) < 100.0


def test_short_good_gaps_are_excluded_before_spectral_notch():
    sfreq = 600
    raw = mne.io.RawArray(
        np.random.default_rng(5).normal(size=(4, 3000)),
        mne.create_info(["M1", "M2", "M3", "M4"], sfreq, "mag"),
        verbose="ERROR",
    )
    # Four good samples between BAD spans, plus one at the recording end.
    raw.set_annotations(
        mne.Annotations(
            [2400 / sfreq, 2504 / sfreq],
            [100 / sfreq, 495 / sfreq],
            ["BAD_segment", "BAD_segment"],
        )
    )
    metric_copy = raw.copy()
    assert _mask_short_filter_segments(metric_copy) == 2
    assert len(raw.annotations) == 2

    cfg = SimpleNamespace(
        ch_types=["mag"],
        _channel_metrics=["log_std"],
        _bad_channel_lfreq=1.0,
        _bad_channel_hfreq=90.0,
    )
    names, specs = BadChannelsAnalysis(cfg)._compute_channel_metrics(raw)
    assert names == raw.ch_names
    assert len(specs) == 1
    assert np.isfinite(specs[0].values).all()
    assert len(raw.annotations) == 2
