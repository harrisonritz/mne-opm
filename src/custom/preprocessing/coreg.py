"""Automatic coregistration for OPM-MEG data.

This module performs MRI-MEG coregistration using iterative closest point (ICP)
fitting, and writes anatomical landmarks to the BIDS structure. It requires
FreeSurfer fiducials to be defined (either pre-existing or via GUI).

Coregistration Strategy
-----------------------
1. Check for existing fiducials in FreeSurfer subjects directory
2. If none exist, launch GUI for manual fiducial placement
3. If GUI fails or fiducials cannot be set, raise an error
4. Run iterative ICP fitting with configurable parameters
5. Write anatomical landmarks to BIDS structure

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=coreg --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.coreg import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process (uses first subject).
    sessions : list
        Session IDs to process (uses first session).
    subjects_dir : str
        FreeSurfer subjects directory (typically set via SUBJECTS_DIR env var).
    task : str
        Task name for MEG data.

Optional:
    _coreg_hair_grow : float
        Hair growth parameter in mm for ICP fitting. Default: 5.0.
    _coreg_omit_distance : float
        Distance threshold for omitting HSP points (meters). Default: 0.0025.
    _coreg_n_rounds : int
        Number of ICP fitting rounds. Default: 2.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import os
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict

import mne
import mne_bids
from mne.io import read_info
from mne_bids import BIDSPath, get_anat_landmarks, write_anat
import numpy as np

from ._base import BaseAnalysis


class CoregAnalysis(BaseAnalysis):
    """Automatic MRI-MEG coregistration using ICP fitting.

    This analysis performs automated coregistration between MRI and MEG
    coordinate systems using iterative closest point (ICP) fitting.
    Fiducials must be defined before running; if not found, the GUI
    is launched for manual placement. The analysis will fail if
    fiducials cannot be established.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'coreg'
    ANALYSIS_NAME : str
        'coreg'

    See Also
    --------
    mne.coreg.Coregistration : MNE coregistration class.
    mne.gui.coregistration : Interactive coregistration GUI.
    """

    ANALYSIS_KEY = "coreg"
    ANALYSIS_NAME = "coreg"

    # Default coregistration parameters
    DEFAULT_HAIR_GROW: float = 5.0
    """Default hair growth offset in mm."""

    DEFAULT_OMIT_DISTANCE: float = 2.5 / 1e3  # 2.5mm in meters
    """Default distance threshold for omitting head shape points."""

    DEFAULT_N_ROUNDS: int = 2
    """Default number of ICP fitting rounds."""

    def is_enabled(self) -> bool:
        """Check if coregistration is enabled.

        Coregistration is fundamental and always runs when called.

        Returns
        -------
        enabled : bool
            Always True.
        """
        return True

    def load_data(self) -> Dict[str, Any]:
        """Load MEG info for coregistration.

        Finds the MEG data file for the configured subject/session and
        loads the info structure needed for coregistration.

        The subject in config may be in FreeSurfer format (sub-XX_ses-YY)
        or MNE format (XX). This method handles both by parsing accordingly.

        Returns
        -------
        data : dict
            Dictionary containing:
            - info: mne.Info object
            - fname_raw: path to raw MEG file
            - subject: subject ID (without 'sub-' prefix, e.g., '009')
            - session: session ID (without 'ses-' prefix, e.g., '01')
            - fs_subject: FreeSurfer subject name (sub-XX_ses-YY format)

        Raises
        ------
        FileNotFoundError
            If no MEG data is found for the subject/session.
        """
        self.log("Loading MEG info...")

        # Get subject from config (handle list format)
        subject_raw = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session_raw = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        # Parse subject - may be in FreeSurfer format (sub-XX_ses-YY) or MNE format (XX)
        if "_ses-" in str(subject_raw):
            # FreeSurfer format: sub-XX_ses-YY
            # Extract subject number and session from the combined string
            fs_subject = str(subject_raw)
            # Parse: "sub-009_ses-01" -> subject="009", session="01"
            parts = fs_subject.split("_ses-")
            subject = parts[0].replace("sub-", "")
            session = parts[1] if len(parts) > 1 else session_raw
            self.log(f"Parsed FreeSurfer format: {fs_subject} -> subject={subject}, session={session}")
        else:
            # MNE format: just the subject number
            subject = str(subject_raw)
            session = str(session_raw)
            fs_subject = f"sub-{subject}_ses-{session}"

        # Find MEG file using BIDS subject format (just the number)
        paths = mne_bids.find_matching_paths(
            root=self.cfg.bids_root,
            subjects=subject,
            sessions=session,
            tasks=self.cfg.task,
            datatypes="meg",
            extensions=".fif",
            ignore_nosub=True,
        )

        if not paths:
            raise FileNotFoundError(
                f"No MEG data found for subject={subject}, session={session}, "
                f"task={self.cfg.task}"
            )

        fname_raw = paths[0]
        info = read_info(fname_raw)

        self.log(f"Loaded info from {fname_raw}")
        self.log(f"FreeSurfer subject: {fs_subject}")

        return {
            "info": info,
            "fname_raw": fname_raw,
            "subject": subject,
            "session": session,
            "fs_subject": fs_subject,
        }

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run coregistration.

        Checks for existing fiducials, launches GUI if needed, then
        performs iterative ICP fitting to align MEG and MRI coordinates.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() containing MEG info and metadata.

        Returns
        -------
        results : dict
            Dictionary containing:
            - coreg: mne.coreg.Coregistration object with fitted transform
            - All keys from input data

        Raises
        ------
        RuntimeError
            If fiducials are not found and cannot be created via GUI.
        """
        info = data["info"]
        fs_subject = data["fs_subject"]
        subjects_dir = self.cfg.subjects_dir

        # Get parameters from config (with defaults)
        hair_grow = getattr(self.cfg, "_coreg_hair_grow", self.DEFAULT_HAIR_GROW)
        omit_distance = getattr(
            self.cfg, "_coreg_omit_distance", self.DEFAULT_OMIT_DISTANCE
        )
        n_rounds = getattr(self.cfg, "_coreg_n_rounds", self.DEFAULT_N_ROUNDS)

        self.log("Coregistration parameters:")
        self.log(f"  Hair grow: {hair_grow} mm")
        self.log(f"  Omit distance: {omit_distance * 1e3:.2f} mm")
        self.log(f"  N rounds: {n_rounds}")

        # Check for existing fiducials
        fid_pattern = os.path.join(subjects_dir, fs_subject, "bem", "*fiducials.fif")
        existing_fids = glob(fid_pattern)

        if not existing_fids:
            self.log("No fiducials found - launching GUI for manual placement...")
            try:
                mne.gui.coregistration(
                    inst=data["fname_raw"],
                    subject=fs_subject,
                    subjects_dir=subjects_dir,
                    block=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"GUI coregistration failed: {e}. "
                    f"Fiducials are required for coregistration. "
                    f"Please ensure a display is available or pre-define fiducials."
                ) from e

            # Check if fiducials were created
            existing_fids = glob(fid_pattern)
            if not existing_fids:
                raise RuntimeError(
                    f"Fiducials were not saved after GUI session. "
                    f"Coregistration requires fiducials to be defined. "
                    f"Expected location: {fid_pattern}"
                )

        self.log(f"Using fiducials: {existing_fids[0]}")

        # Create coregistration object
        self.log("Running automatic coregistration...")
        coreg = mne.coreg.Coregistration(
            info, fs_subject, subjects_dir, fiducials="estimated"
        )

        # Fit fiducials
        coreg.set_scale_mode("Uniform")
        coreg.fit_fiducials(verbose=True)

        # Fit head shape points iteratively
        coreg.set_scale_mode("3-axis")
        coreg.set_grow_hair(hair_grow)

        for rr in range(n_rounds):
            coreg.omit_head_shape_points(distance=omit_distance)
            coreg.fit_icp(n_iterations=100, verbose=True)

            dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
            self.log(
                f"Round {rr + 1}/{n_rounds}: HSP-MRI distance "
                f"(mean/median/max): {np.mean(dists):.2f} / "
                f"{np.median(dists):.2f} / {np.max(dists):.2f} mm"
            )

        # Final distance report
        final_dists = coreg.compute_dig_mri_distances() * 1e3
        self.log(
            f"Final HSP-MRI distance: mean={np.mean(final_dists):.2f} mm, "
            f"median={np.median(final_dists):.2f} mm"
        )

        return {
            "coreg": coreg,
            "info": info,
            **data,
        }

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save coregistration results to BIDS structure.

        Writes the T1w MRI with anatomical landmarks derived from
        the coregistration transform.

        Parameters
        ----------
        results : dict
            Dictionary from run() containing coregistration results.
        """
        self.log("Saving anatomical landmarks to BIDS...")

        coreg = results["coreg"]
        info = results["info"]
        subject = results["subject"]
        session = results["session"]
        fs_subject = results["fs_subject"]
        subjects_dir = self.cfg.subjects_dir

        # FreeSurfer T1 path (use T1.mgz to avoid datatype issues)
        t1w_fs_path = os.path.join(subjects_dir, fs_subject, "mri", "T1.mgz")

        if not os.path.exists(t1w_fs_path):
            raise FileNotFoundError(
                f"FreeSurfer T1 not found: {t1w_fs_path}. "
                f"Ensure FreeSurfer recon-all has been run."
            )

        # BIDS anat path
        anat_bids_path = BIDSPath(
            subject=subject,
            session=session,
            root=self.cfg.bids_root,
            suffix="T1w",
            datatype="anat",
        )

        # Get landmarks in voxel space using the coregistration transform
        landmarks = get_anat_landmarks(
            t1w_fs_path,
            info=info,
            trans=coreg.trans,
            fs_subject=fs_subject,
            fs_subjects_dir=subjects_dir,
        )

        # Write anatomical data with landmarks
        t1w_bids_path = write_anat(
            image=t1w_fs_path,
            bids_path=anat_bids_path,
            landmarks=landmarks,
            verbose=True,
            overwrite=True,
        )

        self.log(f"Saved T1w with landmarks to {t1w_bids_path}")


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = CoregAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
