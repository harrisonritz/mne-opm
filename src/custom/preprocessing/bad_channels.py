"""Bad channel detection for OPM-MEG data.

This module detects bad channels in raw MEG data using the Generalized
Extreme Studentized Deviate (GESD) test from the OSL-ephys library.
Bad channels are those with statistical properties significantly
different from other channels, indicating sensor malfunction or
poor contact.

Detection is performed on bandpass-filtered data to focus on the
frequency range of interest and avoid contamination from low-frequency
drifts or high-frequency noise.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=bad_channels --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.bad_channels import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    l_freq : float
        High-pass filter frequency for detection.
    h_freq : float
        Low-pass filter frequency for detection.
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

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

from osl_ephys.preprocessing.osl_wrappers import bad_channels as osl_bad_channels

import mne_bids

from ._base import BaseAnalysis
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


class BadChannelsAnalysis(BaseAnalysis):
    """Detect bad channels using statistical methods.

    Uses GESD (Generalized Extreme Studentized Deviate) test to identify
    channels that are statistical outliers compared to other channels.
    This is robust to multiple outliers and doesn't assume a specific
    distribution of the data.

    When processing multiple tasks (e.g., main task + noise), bad channels
    are detected separately but the union of all detected bad channels is
    marked in all recordings.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badchannels'
    ANALYSIS_NAME : str
        'bad_channels'

    See Also
    --------
    osl_ephys.preprocessing.osl_wrappers.bad_channels : OSL detection function.
    """

    ANALYSIS_KEY = "badchannels"
    ANALYSIS_NAME = "bad_channels"

    def is_enabled(self) -> bool:
        """Check if bad channel detection is enabled.

        Returns
        -------
        enabled : bool
            Always True (no config flag required for this analysis).
        """
        return True

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for bad channel detection.

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
            # Search for raw files (handles runs, splits, etc.); honours
            # cfg.custom_proc so subsequent custom steps read from deriv.
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No raw data found for task={task}")

            raw = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
            data[task] = raw
            self.log(f"Loaded raw data for task={task}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bad channel detection on all loaded data.

        Detects bad channels in each task separately, then creates a
        union of all detected bad channels.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with raw data per task and 'bads' key containing
            the union of all detected bad channels.
        """
        results: Dict[str, Any] = {}
        all_bads: set[str] = set()

        for task, raw in data.items():
            self.log(f"Processing task={task}")
            bads = self._detect_bad_channels(raw)
            results[task] = raw
            all_bads.update(bads)

        results["bads"] = sorted(all_bads)
        self.log(f"Total unique bad channels: {len(results['bads'])}")

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results with bad channel markings.

        Updates raw data info['bads'] with newly detected channels
        (merged with existing bad channels) and writes to BIDS.

        Parameters
        ----------
        results : dict
            Dictionary with raw data per task and 'bads' list.
        """
        self.log("Saving results...")

        # Get detected bad channels
        unique_bads = sorted(set(results.get("bads", []))) if results.get("bads") else []

        # Separate task data from metadata
        tasks = {k: v for k, v in results.items() if k not in {"bads"}}

        # Process noise FIRST (when present) so the task save can use the
        # already-written noise as its empty-room association.
        ordered_tasks = sorted(tasks.items(), key=lambda kv: kv[0] != "noise")

        er_output_bp = None
        for task, raw in ordered_tasks:
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No file found for task={task}")

            source_bp = paths[0]

            # Merge existing and newly detected bad channels into raw.info,
            # which write_raw_bids will reflect in the output channels.tsv.
            if unique_bads:
                existing_bads = raw.info.get("bads", [])
                merged_bads = sorted(set(existing_bads) | set(unique_bads))
                raw.info["bads"] = merged_bads
                self.log(f"task={task}: {len(merged_bads)} total bad channels")

            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            # Tag the new bad channels with description="osl" in the
            # channels.tsv that write_raw_bids just produced.
            if unique_bads:
                mne_bids.mark_channels(
                    bids_path=output_bp,
                    ch_names=unique_bads,
                    status="bad",
                    descriptions="osl",
                )

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task}")

    def _detect_bad_channels(self, raw: mne.io.BaseRaw) -> list[str]:
        """Detect bad channels in raw data.

        Applies bandpass filtering before detection to focus on the
        frequency range of interest.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to analyze.

        Returns
        -------
        bads : list of str
            List of bad channel names.
        """
        self.log(
            f"Detecting bad channels (filter: {self.cfg.l_freq}-{self.cfg.h_freq} Hz)"
        )

        # Filter before detection
        filt = raw.copy().filter(
            l_freq=self.cfg.l_freq, h_freq=self.cfg.h_freq, method="iir"
        )

        # Run GESD detection
        detected = osl_bad_channels(
            filt,
            picks=self.cfg.ch_types[0],
            significance_level=0.05,
        )

        bads = list(detected.info["bads"])
        self.log(f"Detected {len(bads)} bad channels: {bads}")

        del filt
        return bads


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = BadChannelsAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
