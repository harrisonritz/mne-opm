"""BIDS path construction utilities for OPM-MEG preprocessing.

This module provides a lightweight helper for constructing BIDSPath objects
from configuration settings. It uses mne_bids.BIDSPath directly rather than
wrapping mne_bids functions unnecessarily.

Note: Prefer using mne_bids.BIDSPath() and mne_bids.find_matching_paths()
directly in your code. This module exists only for commonly repeated patterns.

Functions
---------
get_bids_path
    Construct a BIDSPath from config (convenience wrapper).

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

from mne_bids import BIDSPath


def get_bids_path(
    cfg: SimpleNamespace,
    task: str,
    from_derivatives: bool = False,
    processing: Optional[str] = None,
    suffix: str = "meg",
    extension: str = ".fif",
) -> BIDSPath:
    """Construct a BIDSPath from configuration settings.

    This is a convenience function for creating BIDSPath objects with common
    parameters from the configuration. For more control, use BIDSPath()
    directly.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.
    task : str
        Task name (e.g., 'restingstate', 'noise').
    from_derivatives : bool
        If True, use deriv_root; otherwise use bids_root.
    processing : str, optional
        Processing label (e.g., 'clean').
    suffix : str
        BIDS suffix (default 'meg' for raw source, 'raw' for derivatives).
    extension : str
        File extension (default '.fif').

    Returns
    -------
    bids_path : BIDSPath
        Constructed BIDSPath object.

    Examples
    --------
    >>> bp = get_bids_path(cfg, task='restingstate')
    >>> raw = mne_bids.read_raw_bids(bp)
    """
    root = cfg.deriv_root if from_derivatives else cfg.bids_root

    # Get subject/session (handle lists)
    subject = cfg.subjects[0] if isinstance(cfg.subjects, list) else cfg.subjects
    session = cfg.sessions[0] if isinstance(cfg.sessions, list) else cfg.sessions

    # For derivatives, suffix is 'raw'; for source data, suffix is 'meg'
    if from_derivatives and suffix == "meg":
        suffix = "raw"

    return BIDSPath(
        root=root,
        subject=subject,
        session=session,
        task=task,
        datatype="meg",
        suffix=suffix,
        processing=processing,
        extension=extension,
    )
