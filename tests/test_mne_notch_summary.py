"""Preserve notch output while avoiding quadratic frequency aggregation."""

from collections import Counter
import inspect

import mne.filter
import numpy as np

from custom.mne_compat import _notch_frequency_counts, install_notch_summary_fix


def test_counts_preserve_per_window_rounding_and_duplicates():
    windows = [[np.array([59.8, 60.1, 120.2]), np.array([])],
               [np.array([60.2, 119.8]), np.array([60.0])]]
    assert _notch_frequency_counts(windows) == Counter({60.0: 3, 120.0: 2})


def test_notch_fix_preserves_filtered_samples_and_log(monkeypatch):
    original = mne.filter._mt_spectrum_proc
    # A config imported earlier in the suite may already have installed the fix.
    namespace = {}
    lines, first_line = inspect.getsourcelines(original)
    exec(compile("\n" * (first_line - 1) + "".join(lines),
                 original.__code__.co_filename, "exec"),
         original.__globals__, namespace)
    monkeypatch.setattr(mne.filter, "_mt_spectrum_proc", namespace[original.__name__])
    messages = []
    monkeypatch.setattr(mne.filter.logger, "info", messages.append)
    sfreq = 300.0
    t = np.arange(3000) / sfreq
    data = np.random.default_rng(7).normal(size=(2, len(t)))
    data += 8 * np.sin(2 * np.pi * 60.1 * t)
    kwargs = dict(Fs=sfreq, freqs=None, method="spectrum_fit",
                  filter_length="2s", n_jobs=1)
    before = mne.filter.notch_filter(data, **kwargs)
    original_messages = [m for m in messages if m.startswith("Detected notch frequencies")]
    messages.clear()
    assert install_notch_summary_fix()
    assert not install_notch_summary_fix()
    after = mne.filter.notch_filter(data, **kwargs)
    np.testing.assert_array_equal(after, before)
    assert [m for m in messages if m.startswith("Detected notch frequencies")] == original_messages


def test_summary_accepts_many_windows_without_materializing_flat_list():
    # Comparable to a full recording's channels x overlapping windows.
    windows = ((np.array([60.0, 60.2, 120.0]) for _ in range(600))
               for _ in range(150))
    assert _notch_frequency_counts(windows) == Counter({60.0: 90000, 120.0: 90000})
