"""Bad segment detection for OPM-MEG data.

This module detects and annotates bad segments in raw MEG data using
statistical methods from the OSL-ephys library. Bad segments are time
periods with abnormally high or low signal variance, often caused by
movement artifacts, sensor noise, or other transient issues.

Detection Strategy
------------------
For task data:
    1. First pass: Coarse detection with 1-second segments and 5% threshold
    2. Second pass: Finer detection with 0.66-second segments and 5% threshold

For noise data:
    Single pass with 50% threshold (more lenient for empty room recordings)

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=bad_segments --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.bad_segments import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.

Optional:
    process_empty_room : bool
        Also process empty room noise recording. Default: False.
    find_breaks : bool
        Annotate recording breaks before detection. Default: False.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

from osl_ephys.preprocessing.osl_wrappers import bad_segments as osl_bad_segments

import mne_bids

from ._base import BaseAnalysis, SEGMENT_LEN_SEC


class BadSegmentsAnalysis(BaseAnalysis):
    """Detect and annotate bad segments in raw MEG data.

    Uses OSL-ephys statistical detection to identify time segments with
    abnormal signal characteristics. Detected segments are annotated
    as 'BAD_' in the raw data annotations.

    The two-pass approach for task data allows for both coarse artifact
    rejection (large movements, dropouts) and finer detection of smaller
    transient artifacts.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badsegments'
    ANALYSIS_NAME : str
        'bad_segments'

    See Also
    --------
    osl_ephys.preprocessing.osl_wrappers.bad_segments : OSL detection function.
    """

    ANALYSIS_KEY = "badsegments"
    ANALYSIS_NAME = "bad_segments"

    def is_enabled(self) -> bool:
        """Check if bad segment detection is enabled.

        Returns
        -------
        enabled : bool
            Always True (no config flag required for this analysis).
        """
        # Bad segment detection is always enabled when called
        return True

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for bad segment detection.

        Returns
        -------
        data : dict
            Dictionary with raw data per task.
        """
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        # Determine which tasks to load
        tasks = [self.cfg.task]
        if getattr(self.cfg, "process_empty_room", False):
            tasks.insert(0, "noise")

        for task in tasks:
            # Search for raw files (handles runs, splits, etc.)
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks=task,
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if not paths:
                raise FileNotFoundError(f"No raw data found for task={task}")
            
            raw = mne_bids.read_raw_bids(paths[0], extra_params={"preload": True})
            data[task] = raw
            self.log(f"Loaded raw data for task={task}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bad segment detection on all loaded data.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with annotated raw data for each task.
        """
        results: Dict[str, Any] = {"bads": []}

        for task, raw in data.items():
            is_noise = task == "noise"
            self.log(f"Processing task={task} (is_noise={is_noise})")

            cleaned = self._detect_bad_segments(raw, is_noise=is_noise)
            results[task] = cleaned

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save annotated data back to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary with annotated raw data per task.
        """
        self.log("Saving results...")

        # Separate task data from metadata
        tasks = {k: v for k, v in results.items() if k not in {"bads"}}

        # Find empty room path if needed
        er_bids_path = None
        if "noise" in tasks:
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks="noise",
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if paths:
                er_bids_path = paths[0]

        for task, raw in tasks.items():
            # Find existing file to get correct run/split info
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks=task,
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if not paths:
                raise FileNotFoundError(f"No file found for task={task} to save to")
            
            bp = paths[0]
            bp.split = None  # Clear split to write to base file
            
            # Associate with empty room for non-noise tasks
            write_kwargs = dict(
                raw=raw,
                bids_path=bp,
                allow_preload=True,
                overwrite=True,
                format="FIF",
            )
            if er_bids_path and task != "noise":
                write_kwargs["empty_room"] = er_bids_path
            mne_bids.write_raw_bids(**write_kwargs)
            self.log(f"Saved task={task}")

    def _detect_bad_segments(
        self, raw: mne.io.BaseRaw, is_noise: bool = False
    ) -> mne.io.BaseRaw:
        """Detect bad segments in raw data.

        For noise recordings, uses a single pass with lenient thresholds.
        For task recordings, uses two passes with stricter thresholds.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process.
        is_noise : bool
            If True, this is empty room noise data.

        Returns
        -------
        raw_annotated : mne.io.BaseRaw
            Raw data with bad segments annotated.
        """
        sfreq = raw.info["sfreq"]

        if is_noise:
            # Single pass with lenient threshold for noise recordings
            self.log("Detecting bad segments (noise: single pass, 50% threshold)")
            return osl_bad_segments(
                raw,
                picks=self.cfg.ch_types[0],
                ref_meg=False,
                metric="std",
                detect_zeros=False,
                channel_wise=True,
                segment_len=round(sfreq * SEGMENT_LEN_SEC),
                channel_threshold=0.50,
            )

        # Task recording: two-pass detection
        # Annotate breaks first if configured
        if getattr(self.cfg, "find_breaks", False):
            mne.preprocessing.annotate_break(
                raw,
                min_break_duration=self.cfg.min_break_duration,
                t_start_after_previous=self.cfg.t_break_annot_start_after_previous_event,
                t_stop_before_next=self.cfg.t_break_annot_stop_before_next_event,
            )

        # First pass: 1-second segments, 5% threshold
        self.log("Detecting bad segments (pass 1: 1.0s segments, 5% threshold)")
        first_pass = osl_bad_segments(
            raw,
            picks=self.cfg.ch_types[0],
            ref_meg=False,
            metric="std",
            detect_zeros=False,
            channel_wise=True,
            segment_len=round(sfreq * SEGMENT_LEN_SEC),
            channel_threshold=0.05,
        )

        # Second pass: finer segments (0.66s), 5% threshold
        self.log("Detecting bad segments (pass 2: 0.66s segments, 5% threshold)")
        second_pass = osl_bad_segments(
            first_pass,
            picks=self.cfg.ch_types[0],
            ref_meg=False,
            metric="std",
            detect_zeros=False,
            channel_wise=True,
            segment_len=round(first_pass.info["sfreq"] * SEGMENT_LEN_SEC * 0.66),
            channel_threshold=0.05,
        )

        # Count annotated segments
        bad_annots = [a for a in second_pass.annotations if a["description"].startswith("BAD")]
        self.log(f"Detected {len(bad_annots)} bad segments")

        return second_pass


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = BadSegmentsAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
