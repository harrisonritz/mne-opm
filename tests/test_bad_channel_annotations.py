"""All configured channel scores must omit BAD-annotated samples."""

from types import SimpleNamespace

import mne
import numpy as np

from custom.preprocessing.bad_channels import BadChannelsAnalysis


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
