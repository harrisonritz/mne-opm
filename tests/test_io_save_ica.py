"""Tests for preprocessing._io.save_ica_bids and read_ica_bids.

The components TSV is the hand-off to mne-bids-pipeline's apply_ica step:
custom steps update it and never rewrite the pipeline's ICA .fif. Uses real
file I/O in a tmp directory.
"""

from __future__ import annotations

from types import SimpleNamespace

import mne
import pandas as pd
import pytest

from custom.preprocessing._io import get_ica_bids_path, read_ica_bids, save_ica_bids


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
    """Test that save_ica_bids updates the TSV only."""

    def test_shared_ica_updates_pipeline_artifacts(self, ica_setup):
        """Save exclusions where the downstream pipeline reads them."""
        ica, cfg, legacy_tsv, deriv = ica_setup
        shared_tsv = deriv / "sub-001_ses-01_proc-ica_components.tsv"
        legacy_tsv.rename(shared_tsv)
        save_ica_bids(ica, cfg)

        path = get_ica_bids_path(cfg)
        assert path.task is None
        df = pd.read_csv(shared_tsv, sep="\t")
        assert df.loc[df.status == "bad", "component"].tolist() == [0, 2]
        assert not legacy_tsv.exists()

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

    def test_does_not_rewrite_ica_fif(self, ica_setup):
        """The pipeline's ICA .fif is left alone, so its cache stays valid."""
        ica, cfg, _, deriv = ica_setup
        ica_path = deriv / "sub-001_ses-01_task-restingstate_proc-ica_ica.fif"
        ica.save(ica_path)
        before = ica_path.read_bytes()
        ica.exclude = [1, 3]
        save_ica_bids(ica, cfg)
        assert ica_path.read_bytes() == before

    def test_unmarked_component_becomes_good(self, ica_setup):
        """Removing a component from ica.exclude marks it good again."""
        ica, cfg, tsv_path, _ = ica_setup
        save_ica_bids(ica, cfg)
        ica.exclude = [2]
        save_ica_bids(ica, cfg)
        df = pd.read_csv(tsv_path, sep="\t")
        assert df.loc[df.status == "bad", "component"].tolist() == [2]

    def test_read_ica_bids_uses_tsv_exclusions(self, ica_setup):
        """read_ica_bids takes ica.exclude from the TSV, not the .fif."""
        ica, cfg, tsv_path, deriv = ica_setup
        ica.save(deriv / "sub-001_ses-01_task-restingstate_proc-ica_ica.fif")
        ica.exclude = [1, 4]
        save_ica_bids(ica, cfg)
        assert read_ica_bids(cfg).exclude == [1, 4]

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
