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
                raise FileNotFoundError(f"No file found for task={task}")
            
            bp = paths[0]
            
            # Merge existing and newly detected bad channels
            if unique_bads:
                existing_bads = raw.info.get("bads", [])
                merged_bads = sorted(set(existing_bads) | set(unique_bads))
                raw.info["bads"] = merged_bads
                self.log(f"task={task}: {len(merged_bads)} total bad channels")

                # Update BIDS sidecar
                mne_bids.mark_channels(
                    bids_path=bp,
                    ch_names=unique_bads,
                    status="bad",
                    descriptions="osl",
                )

            # Save raw data
            bp.split = None  # Clear split to write to base file
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
