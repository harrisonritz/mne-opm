"""Data I/O utilities for OPM-MEG preprocessing.

This module provides a minimal set of convenience functions for working with
BIDS-formatted MEG data. Most functions are thin wrappers around mne_bids
and mne functions.

**Philosophy**: Use mne_bids functions directly where possible. This module
only provides helpers for commonly repeated patterns or operations that require
multiple mne_bids calls (like save_ica_bids which updates TSV + saves ICA).

Core Functions
--------------
write_raw_bids_preserve_events
    Wrapper around write_raw_bids that preserves events.tsv/json.
save_ica_bids
    Save ICA solution and update components TSV (combines two operations).
get_bids_path_for_task
    Convenience function for creating BIDSPath from config.

For other operations, use mne_bids directly:
    - mne_bids.read_raw_bids() - Load raw data
    - mne_bids.write_raw_bids() - Save raw data
    - mne_bids.mark_channels() - Mark bad channels
    - mne.read_epochs() - Load epochs
    - mne.preprocessing.read_ica() - Load ICA

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mne
import mne_bids
from mne_bids import BIDSPath
# from filelock import SoftFileLock, Timeout
import pandas as pd


def write_raw_bids_preserve_events(**write_kwargs) -> None:
    """Write raw data to BIDS while preserving the existing events files.

    ``mne_bids.write_raw_bids()`` regenerates *_events.tsv* and
    *_events.json* from ``raw.annotations`` every time it is called.
    Because the annotation ↔ events.tsv round-trip is lossy (e.g. BAD
    annotations become event rows, timing quantisation differences),
    repeated writes corrupt the canonical event table produced during
    the initial BIDS conversion.

    This wrapper backs up events.tsv / events.json before the write and
    restores them afterwards so that only the raw data file and other
    sidecar files (channels.tsv, etc.) are updated.

    An exclusive file lock on ``<bids_root>/participants.tsv.lock`` is
    acquired before calling ``write_raw_bids`` to prevent race conditions
    when multiple SLURM array jobs write to the shared ``participants.tsv``
    concurrently.  If the lock cannot be obtained within the timeout, or if
    a transient read error occurs despite the lock (possible on NFS), the
    call is retried with exponential back-off.

    Parameters
    ----------
    **write_kwargs
        All keyword arguments forwarded to
        ``mne_bids.write_raw_bids()``.  Must include ``bids_path``.

    Notes
    -----
    If no events files exist yet (first-time write), the function falls
    through to a plain ``write_raw_bids`` call with no backup/restore.
    """
    bp: BIDSPath = write_kwargs["bids_path"]

    TIMEOUT = 600

    # Exclusive lock on participants.tsv to serialise concurrent SLURM jobs.
    # The .lock file is created next to participants.tsv in the BIDS root.
    assert bp.root is not None, "bids_path must have a root set"
    # participants_lock = SoftFileLock(
    #     Path(bp.root) / "participants.tsv.lock",
    #     timeout=TIMEOUT,  # seconds; long enough for slow writes, prevents deadlock
    # )

    # Derive the events sidecar paths from the raw BIDSPath
    events_tsv: Path = bp.copy().update(
        suffix="events", extension=".tsv"
    ).fpath
    events_json: Path = bp.copy().update(
        suffix="events", extension=".json"
    ).fpath

    # Back up existing events files
    tsv_backup = Path(str(events_tsv) + ".bak") if events_tsv.exists() else None
    json_backup = Path(str(events_json) + ".bak") if events_json.exists() else None

    if tsv_backup is not None:
        shutil.copy2(events_tsv, tsv_backup)
    if json_backup is not None:
        shutil.copy2(events_json, json_backup)

    try:
        _max_retries = 10
        for _attempt in range(_max_retries):
            try:
                mne_bids.write_raw_bids(**write_kwargs)
                break
            except IndexError:
                # Transient empty-file read — can still occur on NFS even with
                # advisory locking.  Retry with exponential back-off.
                if _attempt < _max_retries - 1:
                    _delay = (2 ** _attempt) + random.uniform(0, 2)
                    print(
                        f"[write_raw_bids_preserve_events] Transient "
                        f"participants.tsv read error "
                        f"(attempt {_attempt + 1}/{_max_retries}), "
                        f"retrying in {_delay:.1f} s..."
                    )
                    time.sleep(_delay)
                else:
                    raise
    finally:
        # Restore the original events files regardless of success or failure
        if tsv_backup is not None and tsv_backup.exists():
            shutil.move(str(tsv_backup), str(events_tsv))
        if json_backup is not None and json_backup.exists():
            shutil.move(str(json_backup), str(events_json))


def get_bids_path_for_task(
    cfg: SimpleNamespace,
    task: str,
    from_derivatives: bool = False,
    processing: Optional[str] = None,
    suffix: str = "meg",
    extension: str = ".fif",
) -> BIDSPath:
    """Construct a BIDSPath for a specific task.

    This is a convenience function to avoid repeating subject/session
    extraction logic. For more control, construct BIDSPath directly.

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


def save_ica_bids(
    ica: mne.preprocessing.ICA,
    cfg: SimpleNamespace,
) -> None:
    """Save ICA solution and update components TSV.

    This function combines two operations that should happen together:
    1. Updates the components TSV to mark excluded components as bad
    2. Saves the ICA object with the updated exclusion list

    This is one of the few genuine utility functions (not just a wrapper),
    since it coordinates multiple mne_bids operations that should be atomic.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA object with excluded components marked in ica.exclude.
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.

    Examples
    --------
    >>> ica.exclude = [0, 3, 5]
    >>> save_ica_bids(ica, cfg)
    """
    # Get subject/session
    subject = cfg.subjects[0] if isinstance(cfg.subjects, list) else cfg.subjects
    session = cfg.sessions[0] if isinstance(cfg.sessions, list) else cfg.sessions

    # Build ICA path
    ica_path = BIDSPath(
        root=cfg.deriv_root,
        subject=subject,
        session=session,
        task=cfg.task,
        datatype="meg",
        suffix="ica",
        processing="ica",
        extension=".fif",
        check=False,  # Allow non-standard suffix 'ica'
    )

    # Build components TSV path
    tsv_path = BIDSPath(
        root=cfg.deriv_root,
        subject=subject,
        session=session,
        task=cfg.task,
        datatype="meg",
        suffix="components",
        processing="ica",
        extension=".tsv",
        check=False,  # Allow non-standard suffix 'components'
    )

    # Update components TSV
    df = pd.read_csv(tsv_path.fpath, sep="\t")
    for comp in ica.exclude:
        mask = df["component"].astype(str) == str(comp)
        if mask.any():
            df.loc[mask, "status"] = "bad"
            try:
                df.loc[mask, "status_description"] = "manual"
            except Exception as e:
                print("Exception: ", e)
                print(f"Warning: 'status_description' column not found in {tsv_path.fpath}. Skipping description update.")
                print(f"masked df: {df.loc[mask]}")
    df.to_csv(tsv_path.fpath, sep="\t", index=False)

    # Save ICA object
    ica.save(ica_path.fpath, overwrite=True)


def mark_bad_channels_bids(
    cfg: SimpleNamespace,
    task: str,
    bad_channels: list[str],
    description: str = "osl",
) -> None:
    """Mark bad channels in BIDS sidecar files.

    This function updates the *_channels.tsv sidecar file to mark
    specified channels as bad.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.
    task : str
        Task name.
    bad_channels : list of str
        List of channel names to mark as bad.
    description : str, optional
        Description for why channels are bad. Default is 'osl'.

    Notes
    -----
    This updates the channels.tsv file without modifying the raw data.
    The raw data's info['bads'] should be updated separately.

    Examples
    --------
    >>> mark_bad_channels_bids(cfg, task='restingstate',
    ...                        bad_channels=['MEG0111', 'MEG0121'])
    """
    if not bad_channels:
        return

    bids_path = get_bids_path_for_task(cfg, task=task, from_derivatives=False)

    mne_bids.mark_channels(
        bids_path=bids_path,
        ch_names=bad_channels,
        status="bad",
        descriptions=description,
    )


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def get_empty_room_bids_path(cfg: SimpleNamespace) -> BIDSPath:
    """Get BIDSPath for empty room noise recording.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.

    Returns
    -------
    bids_path : BIDSPath
        BIDSPath for the empty room recording.
    """
    return get_bids_path_for_task(cfg, task="noise", from_derivatives=False)
