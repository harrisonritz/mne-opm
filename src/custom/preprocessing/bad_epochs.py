"""Bad epoch detection for OPM-MEG data.

This module detects and drops bad epochs from epoched MEG data using
the Generalized Extreme Studentized Deviate (GESD) test from the
OSL-ephys library. Bad epochs are trials with abnormal signal
characteristics compared to other trials, typically caused by
movement artifacts or transient noise.

This step is performed after epoching (via mne-bids-pipeline) and
is the final quality control step before sensor-level analysis.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=bad_epochs --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.bad_epochs import run
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

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

from osl_ephys.preprocessing.osl_wrappers import drop_bad_epochs as osl_drop_bad_epochs

from mne_bids import BIDSPath

from ._base import BaseAnalysis


class BadEpochsAnalysis(BaseAnalysis):
    """Detect and drop bad epochs using statistical methods.

    Uses GESD (Generalized Extreme Studentized Deviate) test to identify
    epochs that are statistical outliers compared to other epochs. This
    is robust to multiple outliers and removes trials that would
    negatively impact subsequent analyses.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badepochs'
    ANALYSIS_NAME : str
        'bad_epochs'

    See Also
    --------
    osl_ephys.preprocessing.osl_wrappers.drop_bad_epochs : OSL detection function.
    """

    ANALYSIS_KEY = "badepochs"
    ANALYSIS_NAME = "bad_epochs"

    def is_enabled(self) -> bool:
        """Check if bad epoch detection is enabled.

        Returns
        -------
        enabled : bool
            Always True (no config flag required for this analysis).
        """
        return True

    def load_data(self) -> Dict[str, Any]:
        """Load epoched data for bad epoch detection.

        Returns
        -------
        data : dict
            Dictionary with epochs under the task key.
        """
        self.log("Loading epochs...")

        # Construct BIDSPath for epochs
        subject = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        bids_path = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="epo",
            processing="clean",
            extension=".fif",
            check=False,  # Allow non-standard suffix 'epo'
        )

        epochs = mne.read_epochs(bids_path.fpath, preload=True)
        self.log(f"Loaded {len(epochs)} epochs")

        return {self.cfg.task: epochs}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bad epoch detection.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with epochs.

        Returns
        -------
        results : dict
            Dictionary with cleaned epochs.
        """
        epochs = data[self.cfg.task]

        self.log(f"Detecting bad epochs from {len(epochs)} total epochs")

        clean_epochs = self._drop_bad_epochs(epochs)

        return {self.cfg.task: clean_epochs}

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save cleaned epochs back to derivatives.

        Parameters
        ----------
        results : dict
            Dictionary with cleaned epochs.
        """
        self.log("Saving cleaned epochs...")

        # Construct BIDSPath for epochs
        subject = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        bids_path = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="epo",
            processing="clean",
            extension=".fif",
            check=False,  # Allow non-standard suffix 'epo'
        )

        epochs = results[self.cfg.task]
        epochs.save(bids_path.fpath, split_naming="bids", overwrite=True)

        self.log("Saved cleaned epochs")

    def _drop_bad_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        """Detect and drop bad epochs.

        Uses GESD with standard deviation metric to identify outlier
        epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to analyze.

        Returns
        -------
        clean_epochs : mne.Epochs
            Epochs with bad trials removed.
        """
        n_before = len(epochs)

        clean_epochs = osl_drop_bad_epochs(
            epochs,
            picks=self.cfg.ch_types[0],
            ref_meg=None,
            metric="std",
        )

        n_dropped = n_before - len(clean_epochs)
        pct_dropped = 100 * n_dropped / n_before if n_before > 0 else 0

        self.log(f"Dropped {n_dropped}/{n_before} epochs ({pct_dropped:.1f}%)")
        self.log(f"Remaining epochs: {len(clean_epochs)}")

        return clean_epochs


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = BadEpochsAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
