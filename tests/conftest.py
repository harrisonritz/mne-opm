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

    info["meas_date"] = None
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
    events = np.array(
        [[int(i * 2 * sfreq), 0, 1] for i in range(5)]
    )
    return mne.Epochs(
        raw_meg, events, event_id={"stim": 1},
        tmin=0, tmax=1.0 - 1 / sfreq,
        baseline=None, preload=True,
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
