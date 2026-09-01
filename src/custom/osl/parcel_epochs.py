"""Parcel Epochs that keep their condition names and epoch time axis.

osl-ephys turns beamformed parcel data back into an MNE object with
:func:`osl_ephys.source_recon.parcellation.convert2mne_epochs`, which builds::

    mne.EpochsArray(np.swapaxes(parc_data.T, 1, 2), parc_info, parc_events)

It passes the *events* but neither ``event_id`` nor ``tmin``, so MNE
synthesises a mapping of stringified integer codes (``{'201': 201}``) and
starts the time axis at zero.  Two things are lost:

* the condition names, so ``epochs['response/left']`` raises :exc:`KeyError`
  and the group stage's condition averaging fills every subject with NaN;
* the epoch time axis, so a response-locked ``-0.5 .. 0.5`` window is
  relabelled ``0 .. 1.0`` and every ``t = 0`` marker -- and any baseline
  window -- lands half a second late.

Both are silent: the arrays keep their shape and the pipeline reports success.

Everything needed to keep them is already in the source :class:`mne.Epochs`
object osl-ephys is handed, so this module supplies a drop-in replacement that
uses it, a context manager that installs the replacement over osl-ephys' own
for the backends whose parcellation step this repository does not own, and a
repair path for parcel files that were written before the fix.

Functions
---------
build_parcel_epochs
    Assemble parcel data into an Epochs object with the metadata supplied.
convert2mne_epochs
    Drop-in replacement for osl-ephys' function of the same name.
restore_epoch_metadata
    Rebuild an already-written parcel Epochs with names and a time axis.
preserving_epoch_metadata
    Patch osl-ephys' converter for the enclosing block.
is_placeholder_event_id
    Whether a mapping is the stringified-code one MNE synthesises.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional, Sequence

import mne
import numpy as np


logger = logging.getLogger(__name__)


def is_placeholder_event_id(event_id: dict) -> bool:
    """Whether ``event_id`` is the stringified-code mapping MNE synthesises.

    :class:`mne.EpochsArray` builds ``{'201': 201, '202': 202, ...}`` when it
    is given events but no ``event_id``.  That is what a parcel file written
    before this module existed contains, and it is the marker the repair path
    reports on.

    Parameters
    ----------
    event_id : dict
        Mapping of condition name to integer code.

    Returns
    -------
    placeholder : bool
        True when every name is just its own code written out, and False for
        an empty mapping (there is nothing to have lost).

    Examples
    --------
    >>> is_placeholder_event_id({"201": 201, "202": 202})
    True
    >>> is_placeholder_event_id({"response/left": 201})
    False
    >>> is_placeholder_event_id({})
    False
    """
    if not event_id:
        return False
    return all(str(name) == str(code) for name, code in event_id.items())


def build_parcel_epochs(
    parc_data: np.ndarray,
    *,
    sfreq: float,
    events: np.ndarray,
    event_id: Optional[dict] = None,
    tmin: float = 0.0,
    metadata=None,
    description: Optional[str] = None,
    parcel_names: Optional[Sequence[str]] = None,
) -> mne.EpochsArray:
    """Assemble parcel data into an Epochs object with the metadata supplied.

    The single place parcel :class:`mne.Epochs` objects are constructed, so
    that a file written by the pipeline and one rewritten by the repair path
    are built the same way.

    Parameters
    ----------
    parc_data : numpy.ndarray
        ``(parcels, times, epochs)`` parcel data, the layout osl-ephys'
        beamforming produces.
    sfreq : float
        Sampling rate of ``parc_data``, which is the rate *after* any
        decimation the source stage applied -- not necessarily the sensor rate.
    events : numpy.ndarray
        ``(epochs, 3)`` events array, one row per epoch in ``parc_data``.
    event_id : dict, optional
        Condition name to code.  Omitted, MNE synthesises the stringified-code
        mapping this module exists to avoid, so callers should pass it.
        Entries whose code appears in no event are dropped, because
        :class:`mne.EpochsArray` refuses them; see the Notes.
    tmin : float, optional
        Time of the first sample, in seconds.  Default 0.0, matching
        :class:`mne.EpochsArray`.
    metadata : pandas.DataFrame, optional
        Per-epoch metadata, carried over unchanged.
    description : str, optional
        Written to ``info['description']``, as osl-ephys does.
    parcel_names : sequence of str, optional
        Channel names.  Defaults to ``parcel_0 .. parcel_{n-1}``, the
        convention :func:`custom.osl.sign_flip.parcel_channels` looks for.

    Returns
    -------
    parc_epo : mne.EpochsArray
        Parcel data as ``misc`` channels.

    Raises
    ------
    ValueError
        If ``parc_data`` is not three-dimensional, or has a different number
        of epochs from ``events``.

    Notes
    -----
    An ``event_id`` read back from a FIF file can name conditions that no
    longer have any epochs: :func:`mne.read_epochs` restores the mapping the
    file was written with, while epoch rejection may since have dropped every
    epoch of a condition.  :class:`mne.Epochs` tolerates that on read but
    :class:`mne.EpochsArray` raises "No matching events found", so those
    entries are dropped here -- the same guard
    :func:`custom.osl.extra_funcs.events_from_annotations` applies when it
    builds the mapping in the first place.
    """
    parc_data = np.asarray(parc_data)
    if parc_data.ndim != 3:
        raise ValueError(
            f"parc_data must be (parcels, times, epochs), got shape "
            f"{parc_data.shape}."
        )

    events = np.asarray(events)
    n_epochs = parc_data.shape[2]
    if len(events) != n_epochs:
        raise ValueError(
            f"got {len(events)} events for {n_epochs} epoch(s) of parcel data."
        )

    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(parc_data.shape[0])]
    parcel_names = list(parcel_names)
    if len(parcel_names) != parc_data.shape[0]:
        raise ValueError(
            f"got {len(parcel_names)} parcel name(s) for "
            f"{parc_data.shape[0]} parcel(s)."
        )

    event_id = _drop_unmatched(event_id, events)

    info = mne.create_info(
        ch_names=parcel_names, ch_types="misc", sfreq=float(sfreq)
    )

    # (parcels, times, epochs) -> (epochs, parcels, times), as MNE stores it.
    data = np.swapaxes(parc_data.T, 1, 2)

    parc_epo = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=event_id,
        tmin=float(tmin),
        # The parcel data inherits whatever baseline correction the sensor
        # epochs carried, but has been band-passed since, so re-applying the
        # window here would change the data rather than describe it.  Baseline
        # correction of parcel data belongs to the group stage's
        # ``group.baseline`` option.
        baseline=None,
        metadata=metadata,
        verbose=False,
    )
    if description is not None:
        parc_epo.info["description"] = description

    return parc_epo


def _drop_unmatched(event_id: Optional[dict], events: np.ndarray) -> Optional[dict]:
    """Drop ``event_id`` entries whose code appears in no event.

    See the Notes on :func:`build_parcel_epochs` for why this is needed.
    """
    if not event_id:
        return event_id

    present = set(np.unique(events[:, 2]).tolist())
    kept = {name: code for name, code in event_id.items() if code in present}

    dropped = sorted(set(event_id) - set(kept))
    if dropped:
        logger.warning(
            "dropping %d condition(s) with no surviving epochs from the parcel "
            "event_id: %s",
            len(dropped),
            dropped,
        )

    return kept


def convert2mne_epochs(
    parc_data: np.ndarray, epochs, parcel_names: Optional[Sequence[str]] = None
) -> mne.EpochsArray:
    """Drop-in replacement for osl-ephys' function of the same name.

    Same signature and same output, except that the condition names, the epoch
    time axis and any per-epoch metadata are carried over from ``epochs``
    instead of being dropped.

    Parameters
    ----------
    parc_data : numpy.ndarray
        ``(parcels, times, epochs)`` parcel data.
    epochs : mne.Epochs
        The epochs the parcel data was reconstructed from.  Supplies the
        sampling rate, events, ``event_id``, ``tmin``, metadata and
        description.
    parcel_names : sequence of str, optional
        Channel names; see :func:`build_parcel_epochs`.

    Returns
    -------
    parc_epo : mne.EpochsArray
        Parcellated data.

    Notes
    -----
    ``epochs`` must be the object the beamformer was actually applied to, so
    that its sampling rate matches ``parc_data``: the source stage decimates
    before beamforming, and taking the rate from the undecimated sensor epochs
    would stretch the time axis by the decimation factor.
    """
    return build_parcel_epochs(
        parc_data,
        sfreq=epochs.info["sfreq"],
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin,
        metadata=epochs.metadata,
        description=epochs.info["description"],
        parcel_names=parcel_names,
    )


def restore_epoch_metadata(parc_epochs, source_epochs) -> mne.EpochsArray:
    """Rebuild an already-written parcel Epochs with names and a time axis.

    For parcel files written before :func:`convert2mne_epochs` replaced
    osl-ephys' converter.  The parcel *data* in those files is correct -- only
    the header lost the condition names and ``tmin`` -- so this reads both back
    and rebuilds the object through :func:`build_parcel_epochs`, giving a file
    identical to what the fixed pipeline would have written.

    Sampling rate, channel names and events come from ``parc_epochs``, because
    the source stage may have decimated and dropped epochs before beamforming.
    Condition names, ``tmin`` and metadata come from ``source_epochs``.

    Parameters
    ----------
    parc_epochs : mne.Epochs
        Parcellated epochs read back from disk, preloaded.
    source_epochs : mne.Epochs
        The subject's sensor-level epochs, which still carry the names and the
        time axis.  Need not be preloaded; only the header is read.

    Returns
    -------
    repaired : mne.EpochsArray
        The same parcel data, with the metadata restored.

    Raises
    ------
    ValueError
        If the two objects do not describe the same epochs -- a different
        number of them, or a different sequence of event codes.  Either means
        the files are not a matching pair and the names cannot be trusted.

    Warns
    -----
    The restored window is checked against the source epochs' own: a parcel
    file whose duration does not match ``source_epochs`` to within one sample
    is logged as a warning, since ``tmin`` is then being taken from epochs that
    were cut differently.
    """
    events = np.asarray(parc_epochs.events)
    source_events = np.asarray(source_epochs.events)

    if len(events) != len(source_events):
        raise ValueError(
            f"the parcel file has {len(events)} epoch(s) but the sensor file "
            f"has {len(source_events)}; these are not a matching pair."
        )
    if not np.array_equal(events[:, 2], source_events[:, 2]):
        raise ValueError(
            "the parcel and sensor files disagree on the event codes, so the "
            "condition names cannot be matched to the parcel epochs."
        )

    sfreq = parc_epochs.info["sfreq"]
    n_times = len(parc_epochs.times)
    duration = (n_times - 1) / sfreq
    expected = source_epochs.tmax - source_epochs.tmin
    if abs(duration - expected) > 1.0 / sfreq:
        logger.warning(
            "the parcel epochs span %.3f s but the sensor epochs span %.3f s; "
            "restoring tmin=%.3f from the sensor file anyway, but check that "
            "these files came from the same run",
            duration,
            expected,
            source_epochs.tmin,
        )

    # (epochs, parcels, times) -> (parcels, times, epochs), the layout
    # build_parcel_epochs takes, so that a repaired file is assembled by
    # exactly the code path that writes a new one.
    parc_data = np.swapaxes(parc_epochs.get_data(copy=False), 1, 2).T

    return build_parcel_epochs(
        parc_data,
        sfreq=sfreq,
        events=events,
        event_id=source_epochs.event_id,
        tmin=source_epochs.tmin,
        metadata=source_epochs.metadata,
        description=parc_epochs.info["description"],
        parcel_names=parc_epochs.ch_names,
    )


@contextmanager
def preserving_epoch_metadata():
    """Patch osl-ephys' parcel converter for the enclosing block.

    :func:`osl_ephys.source_recon.wrappers.beamform_and_parcellate` -- the
    step the ``rhino`` backend runs, which this repository does not own --
    reaches ``parcellation.convert2mne_epochs`` through the module, so
    replacing the attribute for the duration of the chain is enough to make
    that backend write parcel epochs that keep their condition names and time
    axis.  The original is restored afterwards.

    The ``freesurfer`` backend calls :func:`convert2mne_epochs` here directly
    and does not need the patch, but the context manager is applied to both so
    that any other osl-ephys code path that parcellates epochs during the
    chain is covered too.

    Yields
    ------
    None
    """
    from osl_ephys.source_recon import parcellation

    original = getattr(parcellation, "convert2mne_epochs", None)
    if original is None:  # pragma: no cover - osl-ephys renamed it
        logger.warning(
            "osl_ephys.source_recon.parcellation has no convert2mne_epochs to "
            "patch; parcel epochs may lose their condition names."
        )
        yield
        return

    parcellation.convert2mne_epochs = convert2mne_epochs
    try:
        yield
    finally:
        parcellation.convert2mne_epochs = original
