"""Interactive manual ICA component review for OPM-MEG data.

This module provides an interactive interface for reviewing and
selecting ICA components to exclude. It displays:

1. **Component topographies**: Spatial maps showing the scalp
   distribution of each component.

2. **Component time courses**: Time series of component activations
   overlaid on the original data.

Users can click on components to mark them for exclusion. This step
is typically performed after automatic ICA labeling to verify the
automatic selections and catch any missed artifacts.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=manual_ica --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.manual_ica import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    deriv_root : str
        Root directory of derivatives.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.
    spatial_filter : str
        Must be 'ica' to enable ICA analyses.

Optional:
    _manual_ica : bool
        Enable/disable manual ICA review. Default: False.

Notes
-----
This step requires a graphical display. If running on a remote server,
ensure X11 forwarding is enabled or use a virtual display.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne
import mne_bids
from mne_bids import BIDSPath

from ._base import BaseAnalysis
from ._io import save_ica_bids


class ManualICAAnalysis(BaseAnalysis):
    """Interactive manual ICA component review.

    Opens interactive plots for visual inspection of ICA components,
    allowing users to review and select components for exclusion.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'manualica'
    ANALYSIS_NAME : str
        'manual_ica'

    See Also
    --------
    mne.preprocessing.ICA.plot_components : Component topography plots.
    mne.preprocessing.ICA.plot_sources : Component time course plots.
    """

    ANALYSIS_KEY = "manualica"
    ANALYSIS_NAME = "manual_ica"

    def is_enabled(self) -> bool:
        """Check if manual ICA review is enabled.

        Requires both _manual_ica=True and spatial_filter='ica'.

        Returns
        -------
        enabled : bool
            True if both conditions are met.
        """
        manual_enabled = getattr(self.cfg, "_manual_ica", False)
        ica_enabled = getattr(self.cfg, "spatial_filter", None) == "ica"
        return manual_enabled and ica_enabled

    def load_data(self) -> Dict[str, Any]:
        """Load raw data and ICA solution.

        Returns
        -------
        data : dict
            Dictionary with:
            - cfg.task : Raw data
            - 'ica' : ICA solution
        """
        self.log("Loading data...")

        # Construct BIDSPath for cleaned raw data
        subject = self.cfg.subjects[0] if isinstance(self.cfg.subjects, list) else self.cfg.subjects
        session = self.cfg.sessions[0] if isinstance(self.cfg.sessions, list) else self.cfg.sessions

        bp_raw = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="raw",
            processing="clean",
            extension=".fif",
        )
        raw = mne_bids.read_raw_bids(bp_raw, extra_params={"preload": True})
        self.log("Loaded cleaned raw data")

        # Load ICA solution
        bp_ica = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="ica",
            processing="ica",
            extension=".fif",
        )
        ica = mne.preprocessing.read_ica(bp_ica.fpath)
        self.log(f"Loaded ICA solution with {ica.n_components_} components")
        self.log(f"Currently excluded: {ica.exclude}")

        return {self.cfg.task: raw, "ica": ica}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Open interactive ICA review interface.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data and ICA.

        Returns
        -------
        results : dict
            Dictionary with reviewed ICA and raw data.
        """
        raw = data[self.cfg.task]
        ica = data["ica"]

        self.log("Opening interactive ICA review...")

        ica = self._manual_ica_review(ica, raw)

        return {self.cfg.task: raw, "ica": ica}

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save ICA solution with updated exclusions.

        Parameters
        ----------
        results : dict
            Dictionary with reviewed ICA.
        """
        self.log("Saving ICA results...")

        ica = results["ica"]
        save_ica_bids(ica, self.cfg)

        self.log(f"Saved ICA with {len(ica.exclude)} excluded components")

    def _manual_ica_review(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Open interactive plots for ICA component review.

        Displays:
        1. Component topographies (spatial maps)
        2. Component sources (time courses)

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution to review.
        raw : mne.io.BaseRaw
            Raw data for visualizing sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with updated exclude list.
        """
        self.log("Displaying component topographies...")
        self.log("Instructions: Click on components to toggle exclusion")

        # Plot component topographies (non-blocking to show both)
        ica.plot_components(inst=raw, nrows=5)

        # Plot component sources (blocking - waits for window close)
        self.log("Displaying component time courses...")
        self.log("Instructions: Close this window when done reviewing")

        ica.plot_sources(inst=raw, show_scrollbars=True, block=True)

        # Report final exclusions
        self.log(f"Final excluded components: {ica.exclude}")

        return ica


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ManualICAAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
