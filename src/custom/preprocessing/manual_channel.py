"""Interactive manual channel selection for OPM-MEG data.

This module provides an interactive interface for visual inspection
of MEG data and manual marking of bad channels. It uses MNE's Qt
browser backend to display the data and allow users to click on
channels to mark them as bad.

This step is typically performed after automatic bad channel detection
to catch any channels that the statistical methods missed, or to
verify/correct the automatic detections.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=manual_channel --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.manual_channel import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    l_freq : float
        High-pass filter frequency for display.
    h_freq : float
        Low-pass filter frequency for display.
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.

Optional:
    _manual_channels : bool
        Enable/disable this analysis. Default: False.
    process_empty_room : bool
        Apply same bad channel markings to noise recording. Default: False.

Notes
-----
If the Qt browser is not available, this step will be skipped with a
warning message. Set the environment variable SKIP_MANUAL=1 to suppress
this warning.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

import mne_bids

from ._base import BaseAnalysis, have_qt_browser
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


class ManualChannelAnalysis(BaseAnalysis):
    """Interactive manual selection of bad channels.

    Opens an interactive plot for visual inspection of the data,
    allowing the user to mark bad channels by clicking on them.
    The same bad channels are also marked in the noise data if
    process_empty_room is enabled.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'manualchannel'
    ANALYSIS_NAME : str
        'manual_channel'

    See Also
    --------
    mne.io.BaseRaw.plot : MNE's interactive data browser.
    """

    ANALYSIS_KEY = "manualchannel"
    ANALYSIS_NAME = "manual_channel"

    def is_enabled(self) -> bool:
        """Check if manual channel selection is enabled.

        Returns
        -------
        enabled : bool
            True if cfg._manual_channels is True.
        """
        return getattr(self.cfg, "_manual_channels", False)

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for manual channel inspection.

        Returns
        -------
        data : dict
            Dictionary with raw data per task.
        """
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        # Load main task (search for files with runs/splits)
        paths = find_custom_input_paths(self.cfg, task=self.cfg.task)
        if not paths:
            raise FileNotFoundError(f"No raw data found for task={self.cfg.task}")
        data[self.cfg.task] = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
        self.log(f"Loaded raw data for task={self.cfg.task} at {paths[0].fpath}")

        # Optionally load noise
        if getattr(self.cfg, "process_empty_room", False):
            paths_noise = find_custom_input_paths(self.cfg, task="noise")
            if paths_noise:
                data["noise"] = read_raw_bids_with_retry(paths_noise[0], extra_params={"preload": True})
                self.log(f"Loaded raw data for task=noise at {paths_noise[0].fpath}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Open interactive browser for manual channel selection.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with:
            - task : Raw data with bad channels marked
            - 'bads' : List of bad channel names
            - 'noise' : Noise data with same bad channels (if applicable)
        """
        results: Dict[str, Any] = {}

        # Get the main task raw data
        raw = data[self.cfg.task]
        noise = data.get("noise")

        # Run interactive selection
        raw, bads, noise = self._manual_channel_selection(raw, noise)

        results[self.cfg.task] = raw
        results["bads"] = bads
        if noise is not None:
            results["noise"] = noise

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results with bad channel markings.

        Parameters
        ----------
        results : dict
            Dictionary with raw data per task and 'bads' list.
        """
        self.log("Saving results...")

        unique_bads = list(results.get("bads", []))

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

            if unique_bads:
                raw.info["bads"] = unique_bads

            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            # Tag the user-selected bad channels with description="osl"
            # in the channels.tsv that write_raw_bids just produced.
            if unique_bads:
                mne_bids.mark_channels(
                    bids_path=output_bp,
                    ch_names=unique_bads,
                    status="bad",
                    descriptions="osl",
                )

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task} → {output_bp.fpath}")

    def _manual_channel_selection(
        self,
        raw: mne.io.BaseRaw,
        noise: mne.io.BaseRaw | None = None,
    ) -> tuple[mne.io.BaseRaw, list[str], mne.io.BaseRaw | None]:
        """Interactive manual selection of bad channels.

        Opens an interactive plot for visual inspection. The user can
        click on channels to mark them as bad. The same markings are
        applied to the noise data if provided.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data for channel inspection.
        noise : mne.io.BaseRaw or None
            Optional noise data to apply same markings.

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw data with bad channels marked.
        bads : list of str
            List of bad channel names selected by user.
        noise : mne.io.BaseRaw or None
            Noise data with same bad channels (if provided).
        """
        if not have_qt_browser():
            self.log(
                "Qt browser not available; skipping interactive plot "
                "(set SKIP_MANUAL=1 to suppress this message)"
            )
        else:
            self.log("Opening interactive plot for channel inspection")
            self.log("Instructions: Click on channels to mark as bad, then close window")

            raw.plot(
                precompute=True,
                n_channels=64,
                show_options=True,
                show=True,
                block=True,
                highpass=self.cfg.l_freq,
                lowpass=self.cfg.h_freq,
                decim=4,
                scalings=dict(mag=1e-11, eyegaze=0.01, pupil=0.01),
            )

        # Process bad channel markings
        # Ensure all entries are strings (handle numpy string types)
        bads: list[str] = []
        for ch in raw.info["bads"]:
            bads.append(ch if isinstance(ch, str) else ch.item())
        raw.info["bads"] = bads

        # Apply same markings to noise data
        if noise is not None:
            self.log(f"Marking {len(bads)} bad channels in noise data")
            noise.info["bads"] = bads.copy()

        self.log(f"Marked {len(bads)} bad channels: {bads}")

        return raw, bads, noise


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ManualChannelAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
