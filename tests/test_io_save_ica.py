"""Tests for preprocessing._io.save_ica_bids.

Covers the main utility function in _io.py that combines ICA saving
with TSV component status updates. Uses real file I/O in a tmp directory.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from custom.preprocessing._io import save_ica_bids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ica_setup(tmp_path, meg_info, rng):
    """Create a fitted ICA, matching TSV, and config pointing to tmp dirs."""
    n_ch = len(meg_info["ch_names"])
    n_samples = int(meg_info["sfreq"] * 10)
    raw = mne.io.RawArray(rng.randn(n_ch, n_samples) * 1e-13, meg_info)

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=5, random_state=42, max_iter=100)
    ica.fit(raw, picks="mag")
    ica.exclude = [0, 2]

    # Build BIDS directory structure
    deriv = tmp_path / "derivatives" / "meg" / "sub-001" / "ses-01" / "meg"
    deriv.mkdir(parents=True)

    cfg = SimpleNamespace(
        deriv_root=str(tmp_path / "derivatives" / "meg"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
    )

    # Create components TSV
    tsv_path = deriv / "sub-001_ses-01_task-restingstate_proc-ica_components.tsv"
    df = pd.DataFrame(
        {
            "component": list(range(5)),
            "status": ["good"] * 5,
            "status_description": [""] * 5,
        }
    )
    df.to_csv(tsv_path, sep="\t", index=False)

    return ica, cfg, tsv_path, deriv


# ---------------------------------------------------------------------------
# save_ica_bids
# ---------------------------------------------------------------------------


class TestSaveIcaBids:
    """Test that save_ica_bids updates TSV and saves ICA object."""

    def test_updates_tsv_status(self, ica_setup):
        """Excluded components should be marked 'bad' in TSV."""
        ica, cfg, tsv_path, _ = ica_setup
        save_ica_bids(ica, cfg)

        df = pd.read_csv(tsv_path, sep="\t")
        # Components 0 and 2 were excluded
        assert df.loc[df["component"] == 0, "status"].iloc[0] == "bad"
        assert df.loc[df["component"] == 2, "status"].iloc[0] == "bad"
        # Component 1 should remain good
        assert df.loc[df["component"] == 1, "status"].iloc[0] == "good"

    def test_updates_status_description(self, ica_setup):
        """Excluded components should have status_description='manual'
        when the column has string dtype (not NaN-only)."""
        ica, cfg, tsv_path, _ = ica_setup

        # Re-create TSV with explicitly-typed string column
        df = pd.DataFrame(
            {
                "component": list(range(5)),
                "status": ["good"] * 5,
                "status_description": ["none"] * 5,  # String values, not empty
            }
        )
        df.to_csv(tsv_path, sep="\t", index=False)

        save_ica_bids(ica, cfg)

        df = pd.read_csv(tsv_path, sep="\t")
        assert df.loc[df["component"] == 0, "status_description"].iloc[0] == "manual"
        assert df.loc[df["component"] == 2, "status_description"].iloc[0] == "manual"

    def test_saves_ica_fif(self, ica_setup):
        """ICA should be saved as a .fif file."""
        ica, cfg, _, deriv = ica_setup
        save_ica_bids(ica, cfg)

        ica_path = deriv / "sub-001_ses-01_task-restingstate_proc-ica_ica.fif"
        assert ica_path.exists()

    def test_saved_ica_preserves_excludes(self, ica_setup):
        """The saved ICA should preserve the exclude list."""
        ica, cfg, _, deriv = ica_setup
        save_ica_bids(ica, cfg)

        ica_path = deriv / "sub-001_ses-01_task-restingstate_proc-ica_ica.fif"
        loaded = mne.preprocessing.read_ica(ica_path)
        assert loaded.exclude == [0, 2]

    def test_empty_exclude_no_changes(self, ica_setup):
        """ICA with empty exclude list should leave TSV unchanged."""
        ica, cfg, tsv_path, _ = ica_setup
        ica.exclude = []
        save_ica_bids(ica, cfg)

        df = pd.read_csv(tsv_path, sep="\t")
        assert all(df["status"] == "good")

    def test_string_subjects(self, ica_setup):
        """Should work when subjects/sessions are strings not lists."""
        ica, cfg, tsv_path, _ = ica_setup
        cfg.subjects = "001"
        cfg.sessions = "01"
        save_ica_bids(ica, cfg)

        df = pd.read_csv(tsv_path, sep="\t")
        assert df.loc[df["component"] == 0, "status"].iloc[0] == "bad"
