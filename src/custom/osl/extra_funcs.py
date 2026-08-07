"""Custom osl-ephys preprocessing wrappers for OPM-MEG data.

osl-ephys builds epochs from ``dataset['events']`` and ``dataset['event_id']``,
and its only built-in way to populate them is ``find_events``, which reads a
stim channel.  ``custom.format_bids`` deliberately converts the Cerca trigger
channels to *annotations* and then drops the stim channels, because leaving
them in makes :func:`mne_bids.write_raw_bids` re-extract events with different
``find_events`` parameters every time a derivative is re-saved.

:func:`events_from_annotations` bridges that gap: it builds the events array
from ``raw.annotations`` instead, so the osl-ephys pipeline reads exactly the
same BIDS data as the mne-bids-pipeline route with no change to
``format_bids``.

Use it in a pipeline YAML like any other step::

    preproc:
      - events_from_annotations: {exclude: [BAD, EDGE]}
      - epochs: {tmin: -0.5, tmax: 0.5}

Functions
---------
events_from_annotations
    Populate ``dataset['events']`` / ``dataset['event_id']`` from annotations.

Constants
---------
PREPROC_EXTRA_FUNCS
    Functions passed to osl-ephys as ``extra_funcs`` for the preproc stage.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import mne
import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ("BAD", "EDGE")
"""Annotation description prefixes ignored when deriving events."""


def events_from_annotations(dataset: Dict[str, Any], userargs: Dict[str, Any]) -> Dict:
    """Populate ``dataset['events']`` and ``dataset['event_id']`` from annotations.

    A drop-in replacement for osl-ephys' ``find_events`` step for data whose
    triggers live in ``raw.annotations`` rather than a stim channel.

    Parameters
    ----------
    dataset : dict
        osl-ephys dataset dict.  Must contain ``'raw'``.  ``'event_id'``, if
        already set (osl-ephys initialises it from ``config['meta']
        ['event_codes']``), is used as the description-to-code mapping.
    userargs : dict
        Step options:

        ``event_id`` : dict, optional
            Explicit ``{description: code}`` mapping.  Overrides
            ``dataset['event_id']``.
        ``regexp`` : str, optional
            Regular expression matched against annotation descriptions, passed
            through to :func:`mne.events_from_annotations`.
        ``exclude`` : list of str, optional
            Description *prefixes* to ignore.  Defaults to
            :data:`DEFAULT_EXCLUDE_PREFIXES`.  Only applies when the mapping is
            derived automatically; an explicit mapping is taken at face value.
        ``strict`` : bool, optional
            If True, raise when a mapped description has no matching
            annotation.  Default False (warn and drop it).
        ``chunk_duration`` : float, optional
            Passed through to :func:`mne.events_from_annotations`.

    Returns
    -------
    dataset : dict
        The input dict, with ``'events'`` and ``'event_id'`` set.

    Raises
    ------
    ValueError
        If no events could be derived, or if ``strict`` is True and a mapped
        description is missing from the annotations.

    Notes
    -----
    Codes come from the config's ``meta.event_codes`` whenever it is given, so
    that event numbering is identical across subjects even when a subject is
    missing a condition.  Descriptions with no annotations are dropped from the
    mapping, because :class:`mne.Epochs` raises on an ``event_id`` entry that
    matches no event.
    """
    logger.info("OSL Stage - %s", "events_from_annotations")
    logger.info("userargs: %s", str(userargs))

    raw = dataset["raw"]

    regexp = userargs.get("regexp", None)
    strict = userargs.get("strict", False)
    exclude = tuple(userargs.get("exclude", DEFAULT_EXCLUDE_PREFIXES))
    chunk_duration = userargs.get("chunk_duration", None)

    mapping = userargs.get("event_id", None) or dataset.get("event_id", None)

    present = set(raw.annotations.description)
    if not present:
        raise ValueError(
            "events_from_annotations: raw has no annotations. Check that "
            "format_bids ran with rename_annot=True."
        )

    if mapping:
        mapping = {str(desc): int(code) for desc, code in mapping.items()}
        missing = sorted(set(mapping) - present)
        if missing:
            message = (
                f"events_from_annotations: no annotations for {missing}. "
                f"Present descriptions: {sorted(present)}."
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)
        mapping = {desc: code for desc, code in mapping.items() if desc in present}
        if not mapping:
            raise ValueError(
                "events_from_annotations: none of the configured event codes "
                f"match the annotations. Configured: "
                f"{sorted(userargs.get('event_id', None) or dataset.get('event_id') or [])}; "
                f"present: {sorted(present)}."
            )
    else:
        # Derive codes from the annotations themselves, skipping the artefact
        # and discontinuity annotations MNE adds.
        descriptions = sorted(
            desc for desc in present if not str(desc).startswith(exclude)
        )
        if not descriptions:
            raise ValueError(
                "events_from_annotations: every annotation description was "
                f"excluded by prefixes {list(exclude)}. Present: {sorted(present)}."
            )
        mapping = {desc: idx + 1 for idx, desc in enumerate(descriptions)}

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=mapping,
        regexp=regexp,
        chunk_duration=chunk_duration,
    )

    if len(events) == 0:
        raise ValueError(
            f"events_from_annotations: no events found for mapping {mapping} "
            f"(regexp={regexp!r})."
        )

    # MNE returns the mapping it actually used; fall back to ours if it does
    # not, so that downstream steps always see a non-empty event_id.
    event_id = event_id or mapping

    # Guard against an event_id entry with no events, which mne.Epochs rejects.
    found_codes = set(np.unique(events[:, 2]).tolist())
    event_id = {desc: code for desc, code in event_id.items() if code in found_codes}

    logger.info(
        "events_from_annotations: %d events across %d condition(s)",
        len(events),
        len(event_id),
    )

    dataset["events"] = events
    dataset["event_id"] = event_id
    return dataset


PREPROC_EXTRA_FUNCS = [events_from_annotations]
"""Custom functions passed to osl-ephys as ``extra_funcs`` for the preproc stage."""
