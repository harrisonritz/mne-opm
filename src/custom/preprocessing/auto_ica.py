"""Automatic ICA component labeling for OPM-MEG data.

This module automatically identifies and labels bad ICA components using
multiple detection strategies:

1. **Reference sensor correlation**: Identifies components that correlate
   with reference sensor ICA (environmental noise).

2. **GESD outlier detection**: Uses statistical tests on component
   properties (kurtosis, variance) to identify artifact components.

These automatic methods reduce the burden of manual ICA component
selection while catching common artifact patterns.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=auto_ica --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.auto_ica import run
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
    _auto_ica : bool
        Enable/disable automatic ICA labeling. Default: False.
    ref_bads : bool
        Use reference sensor ICA correlation. Default: True.
    gesd_bads : bool
        Use GESD outlier detection. Default: True.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
from scipy import stats

import mne
import mne_bids
from mne_bids import BIDSPath
from osl_ephys.preprocessing.osl_wrappers import gesd as osl_gesd

from ._base import BaseAnalysis
from ._io import save_ica_bids


class AutoICAAnalysis(BaseAnalysis):
    """Automatic ICA component labeling.

    Uses multiple strategies to identify artifact components:
    - Reference sensor correlation (if ref_bads=True)
    - GESD outlier detection on kurtosis/variance (if gesd_bads=True)

    Components identified by any method are added to ica.exclude
    and will be removed when ICA is applied to the data.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'autoica'
    ANALYSIS_NAME : str
        'auto_ica'

    See Also
    --------
    mne.preprocessing.ICA : MNE's ICA class.
    osl_ephys.preprocessing.osl_wrappers.gesd : GESD outlier detection.
    """

    ANALYSIS_KEY = "autoica"
    ANALYSIS_NAME = "auto_ica"

    def is_enabled(self) -> bool:
        """Check if automatic ICA is enabled.

        Requires both _auto_ica=True and spatial_filter='ica'.

        Returns
        -------
        enabled : bool
            True if both conditions are met.
        """
        auto_enabled = getattr(self.cfg, "_auto_ica", False)
        ica_enabled = getattr(self.cfg, "spatial_filter", None) == "ica"
        return auto_enabled and ica_enabled

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

        return {self.cfg.task: raw, "ica": ica}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run automatic ICA component labeling.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data and ICA.

        Returns
        -------
        results : dict
            Dictionary with labeled ICA and raw data.
        """
        raw = data[self.cfg.task]
        ica = data["ica"]

        self.log("Running automatic ICA component labeling...")

        # Apply labeling methods
        ica = self._auto_ica(ica, raw)

        return {self.cfg.task: raw, "ica": ica}

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save ICA solution with updated exclusions.

        Parameters
        ----------
        results : dict
            Dictionary with labeled ICA.
        """
        self.log("Saving ICA results...")

        ica = results["ica"]
        save_ica_bids(ica, self.cfg)

        self.log(f"Saved ICA with {len(ica.exclude)} excluded components")

    def _auto_ica(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Apply automatic ICA component labeling.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution to label.
        raw : mne.io.BaseRaw
            Raw data for computing component sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with updated exclude list.
        """
        # Reference sensor correlation method
        if getattr(self.cfg, "ref_bads", True):
            ica = self._label_by_reference(ica, raw)

        # GESD outlier detection method
        if getattr(self.cfg, "gesd_bads", True):
            ica = self._label_by_gesd(ica, raw)

        # Remove duplicates from exclude list
        ica.exclude = sorted(set(ica.exclude))

        self.log(f"Total excluded components: {len(ica.exclude)}")
        self.log(f"Excluded: {ica.exclude}")

        return ica

    def _label_by_reference(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Identify bad components by correlation with reference sensor ICA.

        Fits a separate ICA on reference sensors and identifies main ICA
        components that correlate with reference ICA sources.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Main ICA solution.
        raw : mne.io.BaseRaw
            Raw data containing reference channels.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with reference-correlated components added to exclude.
        """
        self.log("Identifying bad components from reference sensors...")

        # Fit ICA on reference sensors
        ref_raw = raw.copy().pick("ref_meg").filter(l_freq=1, h_freq=None)

        ref_ica = mne.preprocessing.ICA(
            n_components=0.99,
            method="picard",
            max_iter=256,
            allow_ref_meg=True,
        )
        ref_ica.fit(ref_raw, decim=2, reject_by_annotation=True)

        # Get reference ICA sources and add to raw
        ref_src = ref_ica.get_sources(ref_raw)
        ref_src.rename_channels(lambda x: f"REF_{x}")
        raw.add_channels([ref_src], force_update_info=True)

        # Find main ICA components correlated with reference
        ref_idx, _ = ica.find_bads_ref(inst=raw, method="separate")

        self.log(f"Found {len(ref_idx)} reference-correlated components: {ref_idx}")

        ica.exclude.extend(ref_idx)

        # Cleanup
        del ref_raw, ref_ica, ref_src

        return ica

    def _label_by_gesd(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Identify bad components using GESD outlier detection.

        Uses multiple metrics (kurtosis, variance) to identify components
        with unusual statistical properties.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Raw data for computing sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with outlier components added to exclude.
        """
        self.log("Identifying bad components using GESD...")

        # Get ICA sources
        sources = ica.get_sources(raw).get_data()
        n_comps = sources.shape[0]

        # Check if enough components remain for GESD
        n_remaining = n_comps - len(ica.exclude)
        if n_remaining < 5:
            self.log(
                f"Too few components remaining ({n_remaining}) for GESD; skipping"
            )
            return ica

        # Compute statistics for each component
        kurtosis_scores = stats.kurtosis(sources, axis=1)
        std_scores = np.std(sources, axis=1, ddof=1)
        std_diff_scores = np.linalg.norm(np.diff(sources, axis=1), axis=1)

        # Apply GESD to each metric
        self.log(f"Before GESD: {len(ica.exclude)} excluded components")

        metrics = [
            (kurtosis_scores, "kurtosis"),
            (std_scores, "std"),
            (std_diff_scores, "std_diff"),
        ]

        for scores, name in metrics:
            gesd_mask, _ = osl_gesd(scores, p_out=1.0)

            if gesd_mask.sum() == 0:
                self.log(f"{name}: no outliers found")
            else:
                outlier_idx = np.where(gesd_mask)[0].tolist()
                ica.exclude.extend(outlier_idx)
                self.log(f"{name}: found {len(outlier_idx)} outliers: {outlier_idx}")

        self.log(f"After GESD: {len(ica.exclude)} excluded components")

        return ica


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = AutoICAAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
