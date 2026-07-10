"""Shared fixtures for mne-opm test suite.

Provides synthetic MNE objects (Raw, Epochs, ICA, Info) and config
namespaces so that tests can run without real MEG data files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_info(
    n_meg: int = 10,
    n_ref: int = 3,
    sfreq: float = 300.0,
    *,
    include_stim: bool = False,
) -> mne.Info:
    """Create a minimal MNE Info with MEG + optional ref_meg + stim channels."""
    ch_names = [f"MEG{i:03d}" for i in range(n_meg)]
    ch_types = ["mag"] * n_meg

    if n_ref > 0:
        ch_names += [f"REF{i:03d}" for i in range(n_ref)]
        ch_types += ["ref_meg"] * n_ref

    if include_stim:
        ch_names.append("STI001")
        ch_types.append("stim")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Assign unique sensor locations so HFC and forward-model helpers don't complain.
    rng = np.random.RandomState(42)
    for idx, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        # Position on a sphere of radius ~0.1 m
        theta = 2 * np.pi * idx / len(info["chs"])
        phi = np.pi / 4  # 45-degree elevation
        r = 0.1
        loc[0] = r * np.sin(phi) * np.cos(theta)
        loc[1] = r * np.sin(phi) * np.sin(theta)
        loc[2] = r * np.cos(phi)
        # Unit-normal pointing inward
        loc[3:6] = -loc[:3] / np.linalg.norm(loc[:3])
        # Two tangential directions (arbitrary but orthonormal)
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

    info.set_meas_date(None)
    return info


# ---------------------------------------------------------------------------
# Fixtures — small synthetic objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Deterministic NumPy random state."""
    return np.random.RandomState(0)


@pytest.fixture()
def meg_info() -> mne.Info:
    """MNE Info with 10 MEG + 3 ref_meg channels at 300 Hz."""
    return _make_info(n_meg=10, n_ref=3, sfreq=300.0)


@pytest.fixture()
def raw_meg(meg_info, rng) -> mne.io.RawArray:
    """Synthetic Raw with 10 MEG + 3 ref_meg channels, 10 s of data."""
    n_ch = len(meg_info["ch_names"])
    n_samples = int(meg_info["sfreq"] * 10)  # 10 seconds
    data = rng.randn(n_ch, n_samples) * 1e-13  # ~fT scale
    raw = mne.io.RawArray(data, meg_info)
    return raw


@pytest.fixture()
def raw_with_stim(rng) -> mne.io.RawArray:
    """Synthetic Raw with 10 MEG + 3 ref_meg + 1 stim channel, 10 s."""
    info = _make_info(n_meg=10, n_ref=3, sfreq=300.0, include_stim=True)
    n_ch = len(info["ch_names"])
    n_samples = int(info["sfreq"] * 10)
    data = rng.randn(n_ch, n_samples) * 1e-13
    # Put some events on the stim channel
    stim_idx = info["ch_names"].index("STI001")
    data[stim_idx, :] = 0
    data[stim_idx, 300] = 1  # event at 1 s
    data[stim_idx, 900] = 2  # event at 3 s
    return mne.io.RawArray(data, info)


@pytest.fixture()
def epochs_meg(raw_meg) -> mne.Epochs:
    """Create 5 synthetic epochs from raw_meg (2 s each, no overlap)."""
    sfreq = raw_meg.info["sfreq"]
    # Create events every 2 seconds starting at t=0
    events = np.array([[int(i * 2 * sfreq), 0, 1] for i in range(5)])
    return mne.Epochs(
        raw_meg,
        events,
        event_id={"stim": 1},
        tmin=0,
        tmax=1.0 - 1 / sfreq,
        baseline=None,
        preload=True,
    )


@pytest.fixture()
def tmp_dir(tmp_path) -> Path:
    """Convenience alias for pytest's tmp_path."""
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures — config SimpleNamespace
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_cfg(tmp_dir) -> SimpleNamespace:
    """Minimal configuration namespace for testing."""
    return SimpleNamespace(
        bids_root=str(tmp_dir / "bids"),
        deriv_root=str(tmp_dir / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        process_empty_room=False,
        find_breaks=False,
        n_jobs=1,
        spatial_filter=None,
    )


# ---------------------------------------------------------------------------
# Fixtures — synthetic data with injected artifacts
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_with_bad_channel(meg_info, rng) -> mne.io.RawArray:
    """Raw data where MEG001 has 100x higher variance (clearly bad)."""
    n_ch = len(meg_info["ch_names"])
    n_samples = int(meg_info["sfreq"] * 10)
    data = rng.randn(n_ch, n_samples) * 1e-13
    # Make MEG001 (index 1) a bad channel — 100x larger variance
    data[1, :] *= 100.0
    return mne.io.RawArray(data, meg_info)


@pytest.fixture()
def raw_with_artifact_segment(rng) -> mne.io.RawArray:
    """Raw data with a large-amplitude artifact segment at 4–5 s.

    Uses 30 MEG channels so OSL thresholds (5% * n_ch >= 1) are met.
    """
    info = _make_info(n_meg=30, n_ref=3, sfreq=300.0)
    n_ch = len(info["ch_names"])
    n_samples = int(info["sfreq"] * 10)
    data = rng.randn(n_ch, n_samples) * 1e-13
    # Large artifact from 4–5 s on MEG channels only
    n_meg = 30
    start = int(4 * info["sfreq"])
    stop = int(5 * info["sfreq"])
    data[:n_meg, start:stop] *= 100.0
    return mne.io.RawArray(data, info)


@pytest.fixture()
def raw_with_ref_contamination(rng) -> tuple[mne.io.RawArray, np.ndarray]:
    """Raw where MEG channels are contaminated by ref_meg signals.

    Returns (raw, brain_signal) where brain_signal is the original
    uncontaminated MEG data for verification.
    """
    sfreq = 300.0
    n_samples = int(sfreq * 10)
    n_meg, n_ref = 10, 3

    ref_signals = rng.randn(n_ref, n_samples) * 1e-12
    mixing = rng.randn(n_meg, n_ref)
    brain = rng.randn(n_meg, n_samples) * 1e-14
    meg_data = brain + mixing @ ref_signals

    info = _make_info(n_meg=n_meg, n_ref=n_ref, sfreq=sfreq)
    data = np.vstack([meg_data, ref_signals])
    return mne.io.RawArray(data, info), brain
