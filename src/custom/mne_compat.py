"""Narrow compatibility fixes for the installed MNE fork."""

from collections import Counter
import inspect

import numpy as np


def _notch_frequency_counts(freq_list):
    """Count rounded frequencies once per window without copying growing lists."""
    return Counter(
        frequency
        for channel in freq_list
        for window in channel
        for frequency in np.unique(np.round(window)).tolist()
    )


def install_notch_summary_fix():
    """Replace the known quadratic summary, leaving spectrum fitting unchanged.

    Configs are loaded in both the driver and joblib workers, so install there.
    Match only the affected source expression; other MNE versions are left
    alone. No installed dependency files are changed, so uv sync is safe.
    """
    import mne.filter

    original = mne.filter._mt_spectrum_proc
    if getattr(original, "_linear_notch_summary", False):
        return False
    lines, first_line = inspect.getsourcelines(original)
    source = "".join(lines)
    old = (
        "Counter(\n"
        "        sum((np.unique(np.round(ff)).tolist() for f in freq_list for ff in f), list())\n"
        "    )"
    )
    if old not in source:
        return False
    source = source.replace(old, "_opm_notch_frequency_counts(freq_list)", 1)
    namespace = {}
    original.__globals__["_opm_notch_frequency_counts"] = _notch_frequency_counts
    exec(
        compile("\n" * (first_line - 1) + source, original.__code__.co_filename, "exec"),
        original.__globals__,
        namespace,
    )
    replacement = namespace[original.__name__]
    replacement._linear_notch_summary = True
    mne.filter._mt_spectrum_proc = replacement
    return True
