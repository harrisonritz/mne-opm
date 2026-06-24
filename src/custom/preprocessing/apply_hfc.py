"""Homogeneous Field Correction (HFC) for OPM-MEG data.

This module applies Homogeneous Field Correction (HFC) projections to
MEG data. HFC removes spatial field gradients that are uniform across
the sensor array, typically caused by distant interference sources or
movements of the subject relative to the sensors.

HFC is particularly useful for OPM-MEG data because OPM sensors are
more susceptible to environmental magnetic fields due to their
on-scalp positioning and lack of shielding compared to cryogenic
MEG systems.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=apply_hfc --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.apply_hfc import run
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
    _do_HFC : bool
        Enable/disable HFC. Default: False.
    _hfc_order : int
        Order of HFC projections (1-3). Higher orders remove more
        complex spatial patterns. Default: 1.
    process_empty_room : bool
        Apply same projections to noise recording. Default: False.

Notes
-----
HFC projections are computed based on the sensor positions and applied
as SSP (Signal Space Projection) projectors. The projections remove
components that vary smoothly across the sensor array according to
low-order spherical harmonics.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne
import mne_bids

from ._base import BaseAnalysis
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


class ApplyHFCAnalysis(BaseAnalysis):
    """Apply Homogeneous Field Correction (HFC) projections.

    HFC projections remove spatial gradients in the magnetic field that
    are uniform across the sensor array. This is implemented using
    MNE's compute_proj_hfc function.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'applyhfc'
    ANALYSIS_NAME : str
        'apply_hfc'

    See Also
    --------
    mne.preprocessing.compute_proj_hfc : MNE's HFC projection function.
    """

    ANALYSIS_KEY = "applyhfc"
    ANALYSIS_NAME = "apply_hfc"

    def is_enabled(self) -> bool:
        """Check if HFC is enabled.

        Returns
        -------
        enabled : bool
            True if cfg._do_HFC is True.
        """
        return getattr(self.cfg, "_do_HFC", False)

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for HFC application.

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
        """Compute and apply HFC projections.

        Projections are computed from the main task data and applied
        to both task and noise recordings.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with HFC-corrected raw data per task.
        """
        results: Dict[str, Any] = {"bads": []}

        # Get data
        raw = data[self.cfg.task]
        noise = data.get("noise")

        # Apply HFC
        raw, noise = self._apply_hfc(raw, noise)

        results[self.cfg.task] = raw
        if noise is not None:
            results["noise"] = noise

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save HFC-corrected data back to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary with HFC-corrected raw data per task.
        """
        self.log("Saving results...")

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
            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task} → {output_bp.fpath}")

    def _apply_hfc(
        self,
        raw: mne.io.BaseRaw,
        noise: mne.io.BaseRaw | None = None,
    ) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw | None]:
        """Compute and apply HFC projections.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw MEG data to apply HFC to.
        noise : mne.io.BaseRaw or None
            Optional noise data to apply same projections.

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw data with HFC projections applied.
        noise : mne.io.BaseRaw or None
            Noise data with HFC projections applied (if provided).
        """
        hfc_order = getattr(self.cfg, "_hfc_order", 1)

        self.log(f"Computing HFC projections (order={hfc_order})")

        # Compute HFC projections
        projs = mne.preprocessing.compute_proj_hfc(
            raw.info,
            order=hfc_order,
            picks=self.cfg.ch_types[0],
        )

        self.log(f"Computed {len(projs)} HFC projection(s)")

        # Apply to main task data
        raw.add_proj(projs=projs)#.apply_proj()
        self.log("Applied HFC projections to task data")

        # Apply same projections to noise data
        if noise is not None:
            noise.add_proj(projs=projs)#.apply_proj()
            self.log("Applied HFC projections to noise data")

        self.log("HFC application complete!")

        return raw, noise


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ApplyHFCAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
