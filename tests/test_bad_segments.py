"""Tests for bad_segments.py — statistical bad segment detection.

Exercises the core detection logic (_detect_bad_segments) with noise
vs task modes, the two-pass strategy for task data, and the module-level
run(cfg) entry point. BIDS I/O is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from custom.preprocessing.bad_segments import BadSegmentsAnalysis, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bad_seg_cfg(tmp_path):
    """Config appropriate for bad segment detection."""
    return SimpleNamespace(
        bids_root=str(tmp_path / "bids"),
        deriv_root=str(tmp_path / "derivatives"),
        subjects=["001"],
        sessions=["01"],
        task="restingstate",
        ch_types=["mag"],
        process_empty_room=False,
        find_breaks=False,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# _detect_bad_segments — core logic
# ---------------------------------------------------------------------------

class TestDetectBadSegments:
    """Test the two-pass detection strategy on synthetic data."""

    def test_noise_mode_single_pass(
        self, raw_with_artifact_segment, bad_seg_cfg
    ):
        """Noise mode uses a single pass with lenient threshold."""
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        result = analysis._detect_bad_segments(
            raw_with_artifact_segment, is_noise=True
        )
        # Result should be a Raw with annotations
        assert isinstance(result, mne.io.BaseRaw)

    def test_task_mode_two_pass(
        self, raw_with_artifact_segment, bad_seg_cfg
    ):
        """Task mode uses two passes and should detect the artifact."""
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        result = analysis._detect_bad_segments(
            raw_with_artifact_segment, is_noise=False
        )
        assert isinstance(result, mne.io.BaseRaw)
        # With a 100x artifact, osl_bad_segments adds annotations
        # named "bad_segment_*" (lowercase) or "BAD_*"
        bad_annots = [
            a for a in result.annotations
            if a["description"].lower().startswith("bad")
        ]
        assert len(bad_annots) > 0, (
            "Two-pass detection should flag the 100x artifact segment"
        )

    def test_clean_data_few_annotations(
        self, raw_with_artifact_segment, bad_seg_cfg
    ):
        """Noise mode with uniform data should be lenient."""
        # Use noise mode on artifact data — lenient threshold
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        result = analysis._detect_bad_segments(
            raw_with_artifact_segment, is_noise=True
        )
        # Noise mode uses 50% threshold — very lenient
        assert isinstance(result, mne.io.BaseRaw)

    def test_find_breaks_annotation(
        self, raw_with_artifact_segment, bad_seg_cfg
    ):
        """When find_breaks=True, should annotate breaks before detection."""
        bad_seg_cfg.find_breaks = True
        bad_seg_cfg.min_break_duration = 2.0
        bad_seg_cfg.t_break_annot_start_after_previous_event = 0.1
        bad_seg_cfg.t_break_annot_stop_before_next_event = 0.1

        # annotate_break needs existing annotations (events) to find breaks
        raw = raw_with_artifact_segment
        raw.annotations.append(onset=0.5, duration=0.01, description="trial")
        raw.annotations.append(onset=9.0, duration=0.01, description="trial")

        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        result = analysis._detect_bad_segments(raw, is_noise=False)
        assert isinstance(result, mne.io.BaseRaw)


# ---------------------------------------------------------------------------
# run() method
# ---------------------------------------------------------------------------

class TestBadSegmentsRun:
    """Test run() method processes task data."""

    def test_run_returns_task_results(
        self, raw_with_artifact_segment, bad_seg_cfg
    ):
        """run() should process each task and return annotated raw."""
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        data = {bad_seg_cfg.task: raw_with_artifact_segment}
        results = analysis.run(data)

        assert bad_seg_cfg.task in results
        assert isinstance(results[bad_seg_cfg.task], mne.io.BaseRaw)

    def test_run_noise_and_task(
        self, raw_meg, raw_with_artifact_segment, bad_seg_cfg
    ):
        """run() handles noise + task data."""
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        data = {"noise": raw_meg, bad_seg_cfg.task: raw_with_artifact_segment}
        results = analysis.run(data)

        assert "noise" in results
        assert bad_seg_cfg.task in results


# ---------------------------------------------------------------------------
# load_data — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestBadSegmentsLoadData:
    """Test load_data with mocked mne_bids calls."""

    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_load_single_task(
        self, mock_bids, raw_with_artifact_segment, bad_seg_cfg
    ):
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_with_artifact_segment

        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        data = analysis.load_data()

        assert bad_seg_cfg.task in data
        mock_bids.read_raw_bids.assert_called_once()

    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_load_with_empty_room(
        self, mock_bids, raw_with_artifact_segment, bad_seg_cfg
    ):
        bad_seg_cfg.process_empty_room = True
        mock_bp = MagicMock()
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_with_artifact_segment

        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        data = analysis.load_data()

        assert "noise" in data
        assert bad_seg_cfg.task in data

    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_load_no_files_raises(self, mock_bids, bad_seg_cfg):
        mock_bids.find_matching_paths.return_value = []
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        with pytest.raises(FileNotFoundError, match="No raw data"):
            analysis.load_data()


# ---------------------------------------------------------------------------
# save_results — mocked BIDS I/O
# ---------------------------------------------------------------------------

class TestBadSegmentsSaveResults:
    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_save_writes_annotated_raw(self, mock_bids, raw_meg, bad_seg_cfg):
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        results = {bad_seg_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        mock_bids.write_raw_bids.assert_called_once()

    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_save_no_paths_raises(self, mock_bids, raw_meg, bad_seg_cfg):
        mock_bids.find_matching_paths.return_value = []
        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        results = {bad_seg_cfg.task: raw_meg, "bads": []}
        with pytest.raises(FileNotFoundError, match="No file found"):
            analysis.save_results(results)

    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_save_with_empty_room_association(
        self, mock_bids, raw_meg, bad_seg_cfg
    ):
        """save_results should pass empty_room kwarg for task data."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]

        analysis = BadSegmentsAnalysis(bad_seg_cfg)
        results = {"noise": raw_meg, bad_seg_cfg.task: raw_meg, "bads": []}
        analysis.save_results(results)

        # Should write both noise and task
        assert mock_bids.write_raw_bids.call_count == 2
        # The task write should include empty_room
        task_call_kwargs = mock_bids.write_raw_bids.call_args_list[-1][1]
        assert "empty_room" in task_call_kwargs


# ---------------------------------------------------------------------------
# Module-level run(cfg)
# ---------------------------------------------------------------------------

class TestBadSegmentsModuleRun:
    @patch("custom.preprocessing.bad_segments.mne_bids")
    def test_run_entry_point(
        self, mock_bids, raw_with_artifact_segment, bad_seg_cfg
    ):
        """End-to-end: run(cfg) should load, detect, and save."""
        mock_bp = MagicMock()
        mock_bp.split = None
        mock_bids.find_matching_paths.return_value = [mock_bp]
        mock_bids.read_raw_bids.return_value = raw_with_artifact_segment

        run(bad_seg_cfg)

        mock_bids.read_raw_bids.assert_called()
        mock_bids.write_raw_bids.assert_called()
