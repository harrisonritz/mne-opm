"""Tests for format_bids.py — BIDS conversion, triggers, file tree, validation."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest

from custom.format_bids import (
    _DEFAULT_SCREEN_DISTANCE,
    _DEFAULT_SCREEN_RESOLUTION,
    _DEFAULT_SCREEN_SIZE,
    _HEAD_POS_CHANNELS,
    _build_file_tree,
    _get_head_pos_channels,
    _interpolate_nans,
    _annotation_to_timeseries,
    _match_lengths,
    _reset_first_samp,
    convert_triggers,
    set_bids_params,
    validate_raw_folder,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_screen_resolution(self):
        assert _DEFAULT_SCREEN_RESOLUTION == (1920, 1080)

    def test_screen_size(self):
        assert len(_DEFAULT_SCREEN_SIZE) == 2
        assert all(isinstance(v, float) for v in _DEFAULT_SCREEN_SIZE)

    def test_screen_distance(self):
        assert isinstance(_DEFAULT_SCREEN_DISTANCE, float)
        assert _DEFAULT_SCREEN_DISTANCE > 0


# ---------------------------------------------------------------------------
# _build_file_tree
# ---------------------------------------------------------------------------


class TestBuildFileTree:
    def test_nonexistent_dir(self, tmp_path):
        result = _build_file_tree(str(tmp_path / "nonexistent"))
        assert "NOT FOUND" in result

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = _build_file_tree(str(empty))
        assert result == ""

    def test_flat_dir(self, tmp_path):
        d = tmp_path / "flat"
        d.mkdir()
        (d / "a.txt").touch()
        (d / "b.txt").touch()
        result = _build_file_tree(str(d))
        assert "a.txt" in result
        assert "b.txt" in result

    def test_nested_dir(self, tmp_path):
        d = tmp_path / "root"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        (sub / "inner.txt").touch()
        result = _build_file_tree(str(d))
        assert "sub/" in result
        assert "inner.txt" in result

    def test_max_depth_respected(self, tmp_path):
        """Depth-limited tree should not show deeply nested files."""
        d = tmp_path / "deep"
        d.mkdir()
        level1 = d / "l1"
        level1.mkdir()
        level2 = level1 / "l2"
        level2.mkdir()
        level3 = level2 / "l3"
        level3.mkdir()
        (level3 / "deep_file.txt").touch()

        result_shallow = _build_file_tree(str(d), max_depth=1)
        assert "l1/" in result_shallow
        assert "deep_file.txt" not in result_shallow


# ---------------------------------------------------------------------------
# set_bids_params
# ---------------------------------------------------------------------------


class TestSetBidsParams:
    def test_defaults_without_config(self):
        cfg = set_bids_params("")
        assert cfg.line_freq == 60.0
        assert cfg.bads == []
        assert cfg.crop == 0
        assert cfg.screen_resolution == _DEFAULT_SCREEN_RESOLUTION

    def test_env_vars_used(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAW_DIR", str(tmp_path / "raw"))
        monkeypatch.setenv("BIDS_DIR", str(tmp_path / "bids"))
        cfg = set_bids_params("")
        assert cfg.raw_dir == str(tmp_path / "raw")
        assert cfg.bids_dir == str(tmp_path / "bids")


# ---------------------------------------------------------------------------
# validate_raw_folder
# ---------------------------------------------------------------------------


class TestValidateRawFolder:
    def test_missing_subject_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No subject folder"):
            validate_raw_folder(str(tmp_path), 1)

    def test_missing_task_files_raises(self, tmp_path):
        """Subject folder exists but no task FIFs inside."""
        subj_dir = tmp_path / "experiment_001"
        subj_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No task files"):
            validate_raw_folder(str(tmp_path), 1)

    def test_valid_folder_returns_dict(self, tmp_path):
        """Minimal valid folder structure with a task file."""
        subj_dir = tmp_path / "experiment_001"
        task_dir = subj_dir / "run_task"
        task_dir.mkdir(parents=True)
        (task_dir / "data_meg.fif").touch()

        paths = validate_raw_folder(str(tmp_path), 1)
        assert "task" in paths
        assert len(paths["task"]) == 1
        assert "emptyroom" in paths
        assert "t1w" in paths

    def test_noise_detected(self, tmp_path):
        subj_dir = tmp_path / "experiment_001"
        task_dir = subj_dir / "run_task"
        task_dir.mkdir(parents=True)
        (task_dir / "data_meg.fif").touch()
        noise_dir = subj_dir / "er_noise"
        noise_dir.mkdir()
        (noise_dir / "empty_meg.fif").touch()

        paths = validate_raw_folder(str(tmp_path), 1)
        assert paths["emptyroom"] is not None

    def test_anatomical_detected(self, tmp_path):
        subj_dir = tmp_path / "experiment_001"
        task_dir = subj_dir / "run_task"
        task_dir.mkdir(parents=True)
        (task_dir / "data_meg.fif").touch()
        anat_dir = subj_dir / "anat"
        anat_dir.mkdir()
        (anat_dir / "sub_t1w.nii.gz").touch()

        paths = validate_raw_folder(str(tmp_path), 1)
        assert paths["t1w"] is not None


# ---------------------------------------------------------------------------
# convert_triggers
# ---------------------------------------------------------------------------


class TestConvertTriggers:
    @pytest.fixture()
    def raw_with_triggers(self):
        """Create a Raw with 8 trigger channels + 2 MEG channels."""
        import datetime

        sfreq = 1000.0
        n_samples = 5000  # 5 seconds

        ch_names = [f"Trigger {i}" for i in range(1, 9)] + ["MEG001", "MEG002"]
        ch_types = ["stim"] * 8 + ["mag"] * 2
        info = mne.create_info(ch_names, sfreq, ch_types)
        info.set_meas_date(datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc))

        data = np.zeros((10, n_samples))
        # Set MEG channels to some noise
        rng = np.random.RandomState(42)
        data[8:, :] = rng.randn(2, n_samples) * 1e-13

        # Set trigger 1 high at sample 1000 (event code 1)
        data[0, 1000:1010] = 5.0

        # Set triggers 1+2 high at sample 2000 (event code 3 = 1+2)
        data[0, 2000:2010] = 5.0
        data[1, 2000:2010] = 5.0

        raw = mne.io.RawArray(data, info)
        # Add a dummy annotation so the old_ann filtering works
        raw.annotations.append(onset=0.5, duration=0.1, description="dummy")
        return raw

    def test_drops_stim_channels(self, raw_with_triggers):
        """After convert_triggers, stim channels should be dropped."""
        cfg = SimpleNamespace(
            trigger_desc={1: "stim_a", 3: "stim_b"},
            response_desc={},
        )
        raw_out = convert_triggers(raw_with_triggers, cfg)
        # Trigger Combined and individual trigger channels should be gone
        assert "Trigger Combined" not in raw_out.ch_names
        stim_chs = [ch for ch in raw_out.ch_names if ch.startswith("Trigger")]
        assert len(stim_chs) == 0, f"Stim channels should be dropped, found: {stim_chs}"

    def test_creates_annotations(self, raw_with_triggers):
        cfg = SimpleNamespace(
            trigger_desc={1: "stim_a", 3: "stim_b"},
            response_desc={},
        )
        raw_out = convert_triggers(raw_with_triggers, cfg)
        descriptions = list(raw_out.annotations.description)
        assert "stim_a" in descriptions or "stim_b" in descriptions

    def test_response_desc_applied(self, raw_with_triggers):
        """If response_desc maps annotations, they should be renamed."""
        cfg = SimpleNamespace(
            trigger_desc={1: "original"},
            response_desc={"original": "renamed"},
        )
        raw_out = convert_triggers(raw_with_triggers, cfg)
        descriptions = list(raw_out.annotations.description)
        assert "renamed" in descriptions


# ---------------------------------------------------------------------------
# _interpolate_nans
# ---------------------------------------------------------------------------


class TestInterpolateNans:
    def test_no_nans(self):
        """Data without NaN should be unchanged."""
        info = mne.create_info(["ch1", "ch2"], 100.0, ["eog", "eog"])
        data = np.ones((2, 500))
        raw = mne.io.RawArray(data, info)
        mask = _interpolate_nans(raw, buffer_sec=0.1)
        np.testing.assert_array_equal(mask, False)
        np.testing.assert_allclose(raw.get_data(), 1.0)

    def test_nans_interpolated(self):
        """NaN values should be filled by interpolation."""
        info = mne.create_info(["ch1"], 100.0, ["eog"])
        data = np.arange(100, dtype=float).reshape(1, -1)
        data[0, 40:60] = np.nan
        raw = mne.io.RawArray(data, info)
        mask = _interpolate_nans(raw, buffer_sec=0.05)
        assert mask.any()
        assert not np.isnan(raw.get_data()).any()

    def test_exclude_from_mask_isolates_channels(self):
        """Head position NaNs should not appear in the returned mask."""
        info = mne.create_info(["ch1", "head_x"], 100.0, ["eog", "misc"])
        data = np.ones((2, 500))
        data[1, 200:300] = np.nan  # NaN only in head_x
        raw = mne.io.RawArray(data, info)
        mask = _interpolate_nans(raw, buffer_sec=0.1, exclude_from_mask=["head_x"])
        # mask should be all-False since ch1 has no NaN
        np.testing.assert_array_equal(mask, False)
        # head_x should still be interpolated
        assert not np.isnan(raw.get_data()).any()

    def test_exclude_from_mask_none_is_default(self):
        """Default behavior (no exclusion) should be unchanged."""
        info = mne.create_info(["ch1", "ch2"], 100.0, ["eog", "eog"])
        data = np.ones((2, 500))
        data[1, 200:300] = np.nan
        raw = mne.io.RawArray(data, info)
        mask = _interpolate_nans(raw, buffer_sec=0.1)
        # mask should reflect NaN in ch2
        assert mask[200:300].all()


# ---------------------------------------------------------------------------
# _get_head_pos_channels
# ---------------------------------------------------------------------------


class TestGetHeadPosChannels:
    def test_all_present(self):
        info = mne.create_info(
            ["x_head", "y_head", "distance", "ch1"], 100.0, 4 * ["misc"]
        )
        raw = mne.io.RawArray(np.zeros((4, 100)), info)
        assert _get_head_pos_channels(raw) == ["x_head", "y_head", "distance"]

    def test_none_present(self):
        info = mne.create_info(["ch1", "ch2"], 100.0, ["eog", "eog"])
        raw = mne.io.RawArray(np.zeros((2, 100)), info)
        assert _get_head_pos_channels(raw) == []

    def test_partial(self):
        info = mne.create_info(["x_head", "ch1"], 100.0, ["misc", "eog"])
        raw = mne.io.RawArray(np.zeros((2, 100)), info)
        assert _get_head_pos_channels(raw) == ["x_head"]


# ---------------------------------------------------------------------------
# _annotation_to_timeseries
# ---------------------------------------------------------------------------


class TestAnnotationToTimeseries:
    def test_basic_conversion(self):
        info = mne.create_info(["ch1"], 100.0, ["eog"])
        data = np.zeros((1, 1000))
        raw = mne.io.RawArray(data, info)
        raw.annotations.append(onset=1.0, duration=0.5, description="blink")

        ts = _annotation_to_timeseries(raw, "blink")
        assert ts.shape == (1, 1000)
        assert ts.max() > 0  # Should have nonzero values where blink is

    def test_no_matching_annotation(self):
        info = mne.create_info(["ch1"], 100.0, ["eog"])
        data = np.zeros((1, 1000))
        raw = mne.io.RawArray(data, info)
        ts = _annotation_to_timeseries(raw, "nonexistent")
        np.testing.assert_allclose(ts, 0.0)


# ---------------------------------------------------------------------------
# _match_lengths
# ---------------------------------------------------------------------------


class TestMatchLengths:
    def test_eye_shorter_gets_padded(self):
        info_raw = mne.create_info(["m1"], 100.0, ["mag"])
        info_eye = mne.create_info(["e1"], 100.0, ["eog"])
        raw = mne.io.RawArray(np.zeros((1, 1000)), info_raw)
        eye = mne.io.RawArray(np.zeros((1, 800)), info_eye)
        eye_out = _match_lengths(raw, eye)
        assert eye_out._data.shape[1] == 1000

    def test_eye_longer_gets_trimmed(self):
        info_raw = mne.create_info(["m1"], 100.0, ["mag"])
        info_eye = mne.create_info(["e1"], 100.0, ["eog"])
        raw = mne.io.RawArray(np.zeros((1, 800)), info_raw)
        eye = mne.io.RawArray(np.zeros((1, 1000)), info_eye)
        eye_out = _match_lengths(raw, eye)
        assert eye_out._data.shape[1] == 800

    def test_equal_lengths_unchanged(self):
        info_raw = mne.create_info(["m1"], 100.0, ["mag"])
        info_eye = mne.create_info(["e1"], 100.0, ["eog"])
        raw = mne.io.RawArray(np.zeros((1, 500)), info_raw)
        eye = mne.io.RawArray(np.zeros((1, 500)), info_eye)
        eye_out = _match_lengths(raw, eye)
        assert eye_out._data.shape[1] == 500


# ---------------------------------------------------------------------------
# _reset_first_samp
# ---------------------------------------------------------------------------


class TestResetFirstSamp:
    def test_first_samp_is_zero(self):
        info = mne.create_info(["ch1"], 100.0, ["mag"])
        data = np.arange(100, dtype=float).reshape(1, -1)
        raw = mne.io.RawArray(data, info, first_samp=500)
        raw_out = _reset_first_samp(raw)
        assert raw_out.first_samp == 0

    def test_data_preserved(self):
        info = mne.create_info(["ch1"], 100.0, ["mag"])
        data = np.arange(100, dtype=float).reshape(1, -1) * 1e-13
        raw = mne.io.RawArray(data.copy(), info, first_samp=500)
        raw_out = _reset_first_samp(raw)
        np.testing.assert_allclose(raw_out.get_data(), data)
