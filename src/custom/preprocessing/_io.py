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

custom_proc helpers
-------------------
``cfg.custom_proc`` lets custom preprocessing steps write their results to
``deriv_root`` under a ``proc-<custom_proc>`` BIDS suffix (e.g. ``proc-init``)
rather than overwriting the raw files in ``bids_root``.  When set, the matching
mne-bids-pipeline run will read its inputs from the same proc-tagged
derivatives.  When ``custom_proc`` is unset or ``None``, the legacy behaviour
is preserved: custom steps read from and overwrite the BIDS data files.

The following helpers implement that routing:
    get_custom_proc(cfg)
    find_custom_input_paths(cfg, task, ...)
    get_custom_output_path(cfg, source_bp)
    write_raw_bids_custom_step(raw, cfg, source_bp, ...)

For other operations, use mne_bids directly:
    - mne_bids.read_raw_bids() - Load raw data
    - mne_bids.write_raw_bids() - Save raw data
    - mne_bids.mark_channels() - Mark bad channels
    - mne.read_epochs() - Load epochs
    - mne.preprocessing.read_ica() - Load ICA

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import json
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


def read_raw_bids_with_retry(bids_path, extra_params=None, max_retries=10):
    """Read raw BIDS data with retry logic for NFS race conditions.

    When multiple SLURM jobs run in parallel, ``mne_bids.read_raw_bids``
    can fail with an ``IndexError`` because a shared TSV file
    (``participants.tsv``, ``scans.tsv``, etc.) is caught mid-write by
    another job.  This wrapper retries with exponential back-off.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path to the BIDS dataset.
    extra_params : dict, optional
        Extra parameters forwarded to the raw reader (e.g.
        ``{"preload": True}``).
    max_retries : int
        Maximum number of attempts (default 10).

    Returns
    -------
    raw : mne.io.Raw
        The loaded raw object.
    """
    if extra_params is None:
        extra_params = {}

    for attempt in range(max_retries):
        try:
            return mne_bids.read_raw_bids(
                bids_path, extra_params=extra_params
            )
        except (IndexError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 2)
                print(
                    f"[read_raw_bids_with_retry] Transient read error "
                    f"(attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                raise


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
            except (IndexError, json.JSONDecodeError):
                # Transient empty-file read — can still occur on NFS even with
                # advisory locking.  Retry with exponential back-off.
                if _attempt < _max_retries - 1:
                    _delay = (2 ** _attempt) + random.uniform(0, 2)
                    print(
                        f"[write_raw_bids_preserve_events] Transient "
                        f"BIDS sidecar read error "
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


# -----------------------------------------------------------------------------
# custom_proc routing helpers
# -----------------------------------------------------------------------------


def get_custom_proc(cfg: SimpleNamespace) -> Optional[str]:
    """Return ``cfg.custom_proc`` (or ``None`` if not set / empty).

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    proc : str or None
        The value of ``cfg.custom_proc`` when truthy, otherwise ``None``.

    Notes
    -----
    When this returns a string (e.g. ``"init"``), custom preprocessing
    steps should read from / write to ``deriv_root`` using
    ``processing="init"``.  When it returns ``None``, custom steps fall
    back to the legacy behaviour of reading from / overwriting
    ``bids_root``.
    """
    val = getattr(cfg, "custom_proc", None)
    return val if val else None


def find_custom_input_paths(
    cfg: SimpleNamespace,
    task: str,
    **find_kwargs,
) -> list[BIDSPath]:
    """Locate input raw files for a custom preprocessing step.

    When ``cfg.custom_proc`` is set, prefer files in ``deriv_root`` with
    that processing label (e.g. ``proc-init``).  These will only exist
    after a previous custom step has written there.  If no such files
    exist yet (first custom step), or ``custom_proc`` is unset, fall
    back to the raw files in ``bids_root``.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.
    task : str
        Task name (e.g. ``"restingstate"``, ``"noise"``).
    **find_kwargs
        Extra keyword arguments forwarded to
        :func:`mne_bids.find_matching_paths` (e.g.
        ``runs="01"``).  ``subjects``, ``sessions``, ``tasks``,
        ``datatypes``, ``extensions`` and ``ignore_nosub`` have sensible
        defaults but may be overridden.

    Returns
    -------
    paths : list of BIDSPath
        Matching paths.  Empty list if nothing was found.
    """
    common = dict(
        subjects=cfg.subjects,
        sessions=cfg.sessions,
        tasks=task,
        datatypes="meg",
        extensions=".fif",
        ignore_nosub=True,
    )
    common.update(find_kwargs)

    proc = get_custom_proc(cfg)
    if proc is not None:
        deriv_root = getattr(cfg, "deriv_root", None)
        if deriv_root is not None and Path(deriv_root).exists():
            deriv_paths = mne_bids.find_matching_paths(
                root=deriv_root,
                processings=proc,
                check=False,
                **common,
            )
            if deriv_paths:
                return deriv_paths

    return mne_bids.find_matching_paths(
        root=cfg.bids_root,
        **common,
    )


def get_custom_output_path(
    cfg: SimpleNamespace,
    source_bp: BIDSPath,
) -> BIDSPath:
    """Compute the output BIDSPath for a custom preprocessing step.

    When ``cfg.custom_proc`` is set, redirects to ``deriv_root`` with
    ``processing=cfg.custom_proc``.  Otherwise returns a copy of
    ``source_bp`` unchanged so the legacy "write back to source"
    behaviour is preserved.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    source_bp : BIDSPath
        Path the data was read from.

    Returns
    -------
    output_bp : BIDSPath
        Path the data should be written to.

    Notes
    -----
    The returned path always has ``check=False`` because the
    ``proc-<custom_proc>`` label is non-canonical for raw data.
    """
    if source_bp is None:
        raise ValueError("source_bp must not be None")

    proc = get_custom_proc(cfg)
    if proc is None:
        return source_bp.copy()

    return source_bp.copy().update(
        root=cfg.deriv_root,
        processing=proc,
        check=False,
    )


def _seed_events_files(source_bp: BIDSPath, output_bp: BIDSPath) -> None:
    """Copy events.tsv / events.json from source to destination if needed.

    ``write_raw_bids_preserve_events`` backs up and restores the events
    sidecars at the destination so that the lossy
    annotation ↔ events.tsv round-trip does not corrupt them.  When the
    destination is a fresh ``proc-<custom_proc>`` directory there is
    nothing to back up yet, which means the first write would clobber
    whatever round-trip ``write_raw_bids`` produces from
    ``raw.annotations``.  Seeding the destination with the source's
    events files restores the preserve-and-restore guarantee.
    """
    if output_bp.fpath == source_bp.fpath:
        return

    for ext in (".tsv", ".json"):
        src = source_bp.copy().update(
            suffix="events", extension=ext, check=False
        ).fpath
        dst = output_bp.copy().update(
            suffix="events", extension=ext, check=False
        ).fpath
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_raw_bids_custom_step(
    raw,
    cfg: SimpleNamespace,
    source_bp: BIDSPath,
    *,
    empty_room: Optional[BIDSPath] = None,
    **extra_write_kwargs,
) -> BIDSPath:
    """Write raw data, redirecting to deriv_root when ``custom_proc`` is set.

    Encapsulates the common save pattern used by every custom
    preprocessing step:

    1. Resolve the output path using :func:`get_custom_output_path`.
    2. Seed the destination's events files from the source on the first
       redirected write so :func:`write_raw_bids_preserve_events` has
       canonical events to preserve.
    3. Clear ``output_bp.split`` so the write goes to the base file.
    4. Forward the ``empty_room`` association (if provided) and any
       extra keyword arguments to
       :func:`write_raw_bids_preserve_events`.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data to write.
    cfg : SimpleNamespace
        Configuration object.
    source_bp : BIDSPath
        Path the data was read from (used as the base for output path
        construction and event-file seeding).
    empty_room : BIDSPath or None
        Optional empty-room association.  Should refer to the noise
        recording at the *output* location so the BIDS association is
        consistent with where the data is being written.
    **extra_write_kwargs
        Additional keyword arguments forwarded to
        :func:`write_raw_bids_preserve_events` (e.g. ``format="FIF"``).
        ``raw``, ``bids_path``, ``allow_preload`` and ``overwrite`` have
        defaults but may be overridden.

    Returns
    -------
    output_bp : BIDSPath
        The BIDSPath the data was written to.
    """
    output_bp = get_custom_output_path(cfg, source_bp)
    _seed_events_files(source_bp, output_bp)
    output_bp.split = None

    write_kwargs = dict(
        raw=raw,
        bids_path=output_bp,
        allow_preload=True,
        overwrite=True,
        format="FIF",
    )
    if empty_room is not None:
        write_kwargs["empty_room"] = empty_room
    write_kwargs.update(extra_write_kwargs)
    write_raw_bids_preserve_events(**write_kwargs)
    return output_bp


def save_ica_bids(
    ica: mne.preprocessing.ICA,
    cfg: SimpleNamespace,
    components_df: "pd.DataFrame | None" = None,
) -> None:
    """Save ICA solution and update components TSV.

    This function combines two operations that should happen together:
    1. Updates the components TSV to mark excluded components as bad
    2. Saves the ICA object with the updated exclusion list

    When ``components_df`` is provided (e.g. from
    ``AutoICAAnalysis._build_components_tsv``), it is written directly,
    giving full control over per-method attribution columns.  Otherwise
    the existing TSV is read and excluded components are marked as
    ``status='bad'`` with ``status_description='manual'``.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA object with excluded components marked in ica.exclude.
    cfg : SimpleNamespace
        Configuration object containing BIDS settings.
    components_df : pd.DataFrame | None
        If provided, written as the components TSV verbatim.

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
    if components_df is not None:
        components_df.to_csv(tsv_path.fpath, sep="\t", index=False)
    else:
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
    bids_path: Optional[BIDSPath] = None,
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
    bids_path : BIDSPath, optional
        Explicit BIDSPath to mark.  When omitted, the path is derived
        from ``cfg``: if ``cfg.custom_proc`` is set, the
        ``deriv_root`` / ``proc-<custom_proc>`` location is used;
        otherwise ``bids_root`` is used.

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

    if bids_path is None:
        proc = get_custom_proc(cfg)
        if proc is None:
            bids_path = get_bids_path_for_task(
                cfg, task=task, from_derivatives=False
            )
        else:
            bids_path = get_bids_path_for_task(
                cfg, task=task, from_derivatives=False
            ).copy().update(
                root=cfg.deriv_root,
                processing=proc,
                check=False,
            )

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
