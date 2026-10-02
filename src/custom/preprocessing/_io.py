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
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Optional

import mne
import mne_bids
import numpy as np
from mne_bids import BIDSPath

# from filelock import SoftFileLock, Timeout
import pandas as pd


# -----------------------------------------------------------------------------
# Event-count verification (used to catch metadata ↔ epoch mismatches early)
# -----------------------------------------------------------------------------


def count_condition_events_in_raw(raw, conditions):
    """Count annotations that will produce epochs for the given conditions.

    Mirrors the exact selection logic used by ``mne-bids-pipeline``'s
    ``make_epochs``:

      1. ``mne.events_from_annotations`` with the default regexp
         (which excludes ``BAD_*`` and ``EDGE_*`` annotations).
      2. ``mne.event.match_event_names`` for hierarchical
         (slash-separated) matching against ``conditions``.

    Using the same logic as the pipeline ensures the count returned here
    is what the pipeline will actually epoch.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``annotations`` are inspected.
    conditions : iterable of str
        Condition names (e.g. ``['trial']``) matched against annotation
        descriptions hierarchically (i.e. ``'trial'`` matches ``'trial/X'``).

    Returns
    -------
    n_events : int
        Number of annotations that would become epochs.
    matched_names : list of str
        Annotation description names that matched ``conditions``.
    """
    events, matched_names = _condition_events_in_raw(raw, conditions)
    return len(events), matched_names


def _condition_events_in_raw(raw, conditions):
    """Events array and matched names for :func:`count_condition_events_in_raw`.

    Shared by the count and onset helpers so both select exactly the events
    ``mne-bids-pipeline`` will epoch.  Returns an empty ``(0, 3)`` events array
    when nothing matches.
    """
    empty = np.zeros((0, 3), dtype=int)
    if len(raw.annotations) == 0:
        return empty, []

    try:
        _, event_id = mne.events_from_annotations(raw=raw, verbose="ERROR")
    except ValueError:
        # All annotations filtered out (e.g. only BAD_*).
        return empty, []

    try:
        matched_names = mne.event.match_event_names(
            event_names=event_id,
            keys=list(conditions),
            on_missing="ignore",
        )
    except KeyError:
        return empty, []

    if not matched_names:
        return empty, []

    event_id_filtered = {
        name: event_id[name] for name in matched_names if name in event_id
    }
    events, _ = mne.events_from_annotations(
        raw, event_id=event_id_filtered, verbose="ERROR"
    )
    return events, matched_names


def trial_onsets_in_raw(raw, conditions):
    """Onsets (seconds, ascending) of the events matched by ``conditions``.

    Selects the same events as :func:`count_condition_events_in_raw`, so
    ``len(trial_onsets_in_raw(raw, c)) == count_condition_events_in_raw(raw, c)[0]``.
    Onsets are relative to the first sample; only their differences are used
    (see :func:`reconcile_trial_metadata`).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``annotations`` are inspected.
    conditions : iterable of str
        Condition names matched hierarchically against annotation descriptions.

    Returns
    -------
    onsets : numpy.ndarray of float
        Sorted event onsets in seconds.
    """
    events, _ = _condition_events_in_raw(raw, conditions)
    samples = np.sort(events[:, 0] - raw.first_samp)
    return samples.astype(float) / float(raw.info["sfreq"])


def count_condition_events_in_tsv(events_tsv_path, conditions):
    """Count rows in ``events.tsv`` that would produce epochs for ``conditions``.

    Applies the same hierarchical matching as
    :func:`count_condition_events_in_raw`, but on a BIDS events.tsv file.
    Rows with ``trial_type == 'n/a'`` are excluded (matching
    ``mne_bids.read_raw_bids`` behaviour).

    Parameters
    ----------
    events_tsv_path : str or Path
        Path to a BIDS ``*_events.tsv`` file.
    conditions : iterable of str
        Condition names to match.

    Returns
    -------
    n_events : int
        Number of rows that would become epochs.
    matched_names : list of str
        Description names that matched ``conditions``.
    """
    rows, matched_names = _condition_rows_in_tsv(events_tsv_path, conditions)
    return len(rows), matched_names


def _condition_rows_in_tsv(events_tsv_path, conditions):
    """Matched rows and names for :func:`count_condition_events_in_tsv`.

    Shared by the count and onset helpers.  Returns an empty DataFrame when the
    file is missing, lacks ``trial_type``, or nothing matches.
    """
    events_path = Path(events_tsv_path)
    if not events_path.exists():
        return pd.DataFrame(), []

    events_df = pd.read_csv(events_path, sep="\t")
    if "trial_type" not in events_df.columns:
        return pd.DataFrame(), []

    # Mirror mne_bids drop of n/a trial_type rows
    descriptions = events_df["trial_type"].astype(str)
    events_df = events_df[descriptions != "n/a"]
    descriptions = descriptions[descriptions != "n/a"]
    unique_names = sorted(set(descriptions))

    if not unique_names:
        return pd.DataFrame(), []

    try:
        matched_names = mne.event.match_event_names(
            event_names=unique_names,
            keys=list(conditions),
            on_missing="ignore",
        )
    except KeyError:
        return pd.DataFrame(), []

    if not matched_names:
        return pd.DataFrame(), []

    return events_df[descriptions.isin(matched_names)], matched_names


def trial_onsets_in_tsv(events_tsv_path, conditions):
    """Onsets (seconds, ascending) of the ``events.tsv`` rows matched by ``conditions``.

    The events.tsv counterpart of :func:`trial_onsets_in_raw`, selecting the
    same rows as :func:`count_condition_events_in_tsv`.

    Parameters
    ----------
    events_tsv_path : str or Path
        Path to a BIDS ``*_events.tsv`` file.
    conditions : iterable of str
        Condition names to match.

    Returns
    -------
    onsets : numpy.ndarray of float
        Sorted event onsets in seconds (empty if nothing matches).
    """
    rows, _ = _condition_rows_in_tsv(events_tsv_path, conditions)
    if rows.empty:
        return np.zeros(0, dtype=float)
    return np.sort(rows["onset"].to_numpy(dtype=float))


class TrialAlignment(NamedTuple):
    """Outcome of :func:`reconcile_trial_metadata`.

    Attributes
    ----------
    dropped_rows : tuple of int
        Indices (into the concatenated behavioral rows) of the rows removed
        because they have no trial trigger; empty when the counts matched.
    n_isi_agree : int
        Number of compared inter-trial intervals that agree within tolerance.
    n_isi : int
        Number of inter-trial intervals compared (0 if fewer than two trials).
    """

    dropped_rows: tuple
    n_isi_agree: int
    n_isi: int

    @property
    def isi_agreement(self):
        """Fraction of compared ISIs that agree (``nan`` when none compared)."""
        return self.n_isi_agree / self.n_isi if self.n_isi else float("nan")


def reconcile_trial_metadata(
    meta_dfs,
    trial_onsets,
    *,
    policy="error",
    onset_column="stim_start_time",
    isi_tol=0.05,
    min_isi_agreement=0.95,
    max_excess=20,
):
    """Put behavioral rows in 1:1 correspondence with the trial triggers.

    ``mne-bids-pipeline`` joins metadata to epochs by position, so every
    behavioral row must belong to the trial trigger at the same position.  The
    trigger onsets and the behavioral stimulus-onset times are recorded on
    different clocks, but their *differences* — the inter-trial intervals
    (ISIs) — must agree trial by trial.  That makes the ISIs a
    timing-based alignment check that is independent of the responses.

    When the behavioral log has ``excess`` more rows than there are triggers
    (e.g. the MEG recording started after the task, or a task restart), the
    ``align`` policy drops the contiguous block of ``excess`` rows whose removal
    makes the most ISIs agree.  Every candidate block position is scored in
    O(n): triggers before the block pair with rows at offset 0, triggers after
    it with rows at offset ``excess``, and the one ISI spanning the block is not
    compared.  Ties are resolved in favour of the start or end of the log
    (which leave no spanning ISI); any other tie is ambiguous and raises.

    The ISI agreement is verified in every case, including when the counts
    already match, so a missing *and* an extra trigger cannot slip through.

    Parameters
    ----------
    meta_dfs : list of pandas.DataFrame
        Behavioral rows, one DataFrame per log file, in recording order.
    trial_onsets : array-like of float
        Onsets (seconds) of the trial events the pipeline will epoch, as from
        :func:`trial_onsets_in_raw` / :func:`trial_onsets_in_tsv`.
    policy : {"error", "align"}
        ``"error"`` refuses any count mismatch; ``"align"`` drops the
        best-aligned block of surplus rows.
    onset_column : str
        Behavioral column holding each trial's stimulus-onset time (seconds).
    isi_tol : float
        Maximum |trigger ISI - behavioral ISI| (seconds) for an ISI to agree.
    min_isi_agreement : float
        Minimum fraction of compared ISIs that must agree.
    max_excess : int
        Largest number of surplus rows the ``align`` policy will drop.

    Returns
    -------
    metadata : pandas.DataFrame
        The reconciled rows (index reset), one per trial onset.
    alignment : TrialAlignment
        Which rows were dropped and the ISI agreement.

    Raises
    ------
    ValueError
        On an unknown policy, no metadata, or a missing ``onset_column``.
    RuntimeError
        If there are fewer rows than triggers, the surplus is not allowed by
        ``policy`` / ``max_excess``, the dropped block is ambiguous, or the ISI
        agreement is below ``min_isi_agreement``.
    """
    if policy not in {"error", "align"}:
        raise ValueError(f"Unknown metadata mismatch policy: {policy!r}")
    if not meta_dfs:
        raise ValueError("No behavioral metadata files were provided")

    metadata = pd.concat(meta_dfs, ignore_index=True)
    if onset_column not in metadata.columns:
        raise ValueError(
            f"Behavioral onset column {onset_column!r} not found; it is needed to "
            "verify the trial alignment from inter-trial intervals."
        )

    t = np.sort(np.asarray(trial_onsets, dtype=float))
    b = metadata[onset_column].to_numpy(dtype=float)
    n = len(t)
    excess = len(b) - n
    if excess < 0:
        raise RuntimeError(
            f"Metadata has {-excess} fewer rows than trial events "
            f"({len(b)} vs {n}); refusing a positional join."
        )
    if excess > 0 and (policy == "error" or excess > max_excess):
        raise RuntimeError(
            f"Metadata has {excess} more rows than trial events "
            f"({len(b)} vs {n}); refusing a positional join. The 'align' "
            f"policy drops up to {max_excess} surplus rows chosen by "
            "inter-trial-interval matching (TSX_METADATA_MISMATCH_POLICY=align)."
        )
    if n < 2:
        if excess:
            raise RuntimeError(
                f"Cannot place {excess} surplus metadata row(s) with fewer than "
                "two trial events: there are no inter-trial intervals to match."
            )
        return metadata, TrialAlignment((), 0, 0)

    # agree_k[j]: trigger ISI j (trial j -> j+1) matches behavioral rows j+k -> j+k+1.
    dt = np.diff(t)
    agree_0 = np.abs(dt - np.diff(b[:n])) < isi_tol
    agree_k = np.abs(dt - np.diff(b[excess:])) < isi_tol

    if excess == 0:
        # No block dropped: every trigger pairs with the row at offset 0.
        dropped = ()
        n_agree, n_isi = int(agree_0.sum()), n - 1
    else:
        # Dropping rows [p, p + excess) pairs triggers < p with offset 0 and
        # triggers >= p with offset ``excess``; ISI p-1 spans the block.
        # prefix_0[i] = sum(agree_0[:i]) and suffix_k[i] = sum(agree_k[i:]),
        # for i = 0..n (agree_* have n - 1 entries).
        prefix_0 = np.r_[0, np.cumsum(agree_0)]
        suffix_k = np.r_[np.cumsum(agree_k[::-1])[::-1], 0, 0]
        positions = np.arange(n + 1)
        scores = prefix_0[np.maximum(positions - 1, 0)] + suffix_k[positions]
        best = np.flatnonzero(scores == scores.max())
        edges = [q for q in best if q in (0, n)]
        if len(edges) == 1:
            p = int(edges[0])
        elif len(best) == 1:
            p = int(best[0])
        else:
            raise RuntimeError(
                f"Cannot locate the {excess} surplus metadata row(s): dropping them "
                f"at trial positions {best[:10].tolist()} aligns the inter-trial "
                "intervals equally well."
            )
        dropped = tuple(range(p, p + excess))
        n_agree = int(scores[p])
        n_isi = n - 1 - (1 if 0 < p < n else 0)
        metadata = metadata.drop(index=list(dropped)).reset_index(drop=True)

    alignment = TrialAlignment(dropped, n_agree, n_isi)
    if alignment.isi_agreement < min_isi_agreement:
        raise RuntimeError(
            f"Trial triggers and behavioral rows are not aligned: only "
            f"{n_agree}/{n_isi} inter-trial intervals agree within "
            f"{isi_tol * 1e3:.0f} ms (need {min_isi_agreement:.0%}). "
            "Refusing a positional join."
        )
    assert len(metadata) == n
    return metadata, alignment


def first_response_per_trial(
    raw,
    *,
    trial_conditions=("trial",),
    response_conditions=("response/left", "response/right"),
    max_latency=None,
):
    """Pair each trial annotation with the first response that follows it.

    For every trial annotation (taken in chronological order), the first
    response annotation whose onset falls in the trial window
    ``[trial_onset, min(next_trial_onset, trial_onset + max_latency))`` is
    selected.  Responses that are not the first within a trial window (e.g.
    double presses) and responses that fall outside every trial window
    ("orphans" — before the first trial, after the response deadline, or with
    no containing window) are flagged for removal.

    ``max_latency`` should match the task's response deadline: without it a
    press made after the deadline of an unanswered trial (which the task does
    not record as a response), or on a break screen long after the trial, would
    be paired with that trial.

    Trial and response annotations are matched hierarchically (so ``'trial'``
    matches ``'trial/read_read'`` and ``'response/left'`` matches exactly),
    using the same ``mne.event.match_event_names`` logic as
    :func:`count_condition_events_in_raw`.

    The ``raw`` object is **not** modified.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``annotations`` are inspected.
    trial_conditions : iterable of str
        Condition names identifying trial-onset annotations.
    response_conditions : iterable of str
        Condition names identifying response annotations.
    max_latency : float or None
        Response deadline in seconds after trial onset (exclusive).  ``None``
        bounds each window by the next trial onset only.

    Returns
    -------
    trial_has_response : numpy.ndarray of bool
        One entry per trial annotation, in chronological order; ``True`` when a
        response was found in that trial's window.  ``len`` equals the number of
        trial annotations counted by :func:`count_condition_events_in_raw` for
        ``trial_conditions``.
    keep_ann_idx : list of int
        Indices into ``raw.annotations`` of the kept (first-per-trial) responses,
        in chronological order.
    drop_ann_idx : list of int
        Indices into ``raw.annotations`` of the responses to remove (extra
        presses within a window plus orphans), in chronological order.
    keep_onsets : numpy.ndarray of float
        Onsets (seconds) of the kept responses, in chronological order.
    """
    ann = raw.annotations
    n_ann = len(ann)
    if n_ann == 0:
        return np.zeros(0, dtype=bool), [], [], np.zeros(0, dtype=float)

    unique_names = sorted(set(ann.description))

    def _matched(conds):
        try:
            names = mne.event.match_event_names(
                event_names=unique_names,
                keys=list(conds),
                on_missing="ignore",
            )
        except KeyError:
            return set()
        return set(names)

    trial_names = _matched(trial_conditions)
    response_names = _matched(response_conditions)

    onsets = np.asarray(ann.onset, dtype=float)
    descriptions = np.asarray(ann.description)

    trial_glob = np.array(
        [i for i, d in enumerate(descriptions) if d in trial_names], dtype=int
    )
    response_glob = np.array(
        [i for i, d in enumerate(descriptions) if d in response_names], dtype=int
    )

    # Sort both by onset (annotations are usually already sorted, but be safe).
    if trial_glob.size:
        trial_glob = trial_glob[np.argsort(onsets[trial_glob], kind="stable")]
    if response_glob.size:
        response_glob = response_glob[np.argsort(onsets[response_glob], kind="stable")]

    trial_onsets = onsets[trial_glob]
    resp_onsets = onsets[response_glob]

    n_trials = len(trial_glob)
    trial_has_response = np.zeros(n_trials, dtype=bool)
    keep_ann_idx: list[int] = []

    # Upper bound of each trial window is the next trial onset (inf for the
    # last), capped at the response deadline.
    next_onsets = np.full(n_trials, np.inf, dtype=float)
    if n_trials > 1:
        next_onsets[:-1] = trial_onsets[1:]
    if max_latency is not None:
        next_onsets = np.minimum(next_onsets, trial_onsets + float(max_latency))

    used = np.zeros(len(response_glob), dtype=bool)
    for j in range(n_trials):
        lo, hi = trial_onsets[j], next_onsets[j]
        in_window = (resp_onsets >= lo) & (resp_onsets < hi) & (~used)
        cand = np.where(in_window)[0]
        if cand.size:
            # resp_onsets is sorted ascending, so the first candidate is earliest.
            first = int(cand[0])
            used[first] = True
            trial_has_response[j] = True
            keep_ann_idx.append(int(response_glob[first]))

    keep_set = set(keep_ann_idx)
    drop_ann_idx = [int(i) for i in response_glob if int(i) not in keep_set]
    keep_onsets = (
        onsets[np.asarray(keep_ann_idx, dtype=int)]
        if keep_ann_idx
        else np.zeros(0, dtype=float)
    )

    return trial_has_response, keep_ann_idx, drop_ann_idx, keep_onsets


def trial_following_event(
    raw,
    *,
    event_conditions,
    trial_conditions=("trial",),
):
    """Pair each event annotation with the first trial annotation that follows it.

    Used by event-locked analyses whose epochs are not trials (e.g. the TSX
    switch cue, ``CSI``) but whose metadata is per-trial: the returned indices
    select, from the per-trial metadata, the row of the trial each event
    precedes.  The positional join is only valid when every event is followed
    by its own trial, so two events with no trial between them (a duplicated or
    spurious trigger) and an event with no later trial are errors rather than
    being silently paired.

    Event and trial annotations are matched hierarchically, using the same
    ``mne.event.match_event_names`` logic as :func:`first_response_per_trial`.

    The ``raw`` object is **not** modified.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``annotations`` are inspected.
    event_conditions : iterable of str
        Condition names identifying the locking events (e.g. ``('CSI',)``).
    trial_conditions : iterable of str
        Condition names identifying trial-onset annotations.

    Returns
    -------
    trial_idx : numpy.ndarray of int
        One entry per event annotation, in chronological order: the index, in
        the chronologically ordered trial annotations, of the first trial whose
        onset is strictly after the event.  ``len`` equals the number of events
        counted by :func:`count_condition_events_in_raw` for
        ``event_conditions``.
    lags : numpy.ndarray of float
        Seconds from each event to its trial.

    Raises
    ------
    RuntimeError
        If an event has no later trial, or two events share the same trial.
    """
    ann = raw.annotations
    if len(ann) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    unique_names = sorted(set(ann.description))

    def _matched(conds):
        try:
            names = mne.event.match_event_names(
                event_names=unique_names,
                keys=list(conds),
                on_missing="ignore",
            )
        except KeyError:
            return set()
        return set(names)

    event_names = _matched(event_conditions)
    trial_names = _matched(trial_conditions)
    overlap = event_names & trial_names
    if overlap:
        raise ValueError(
            f"Annotations {sorted(overlap)} match both event_conditions "
            f"{list(event_conditions)} and trial_conditions {list(trial_conditions)}"
        )

    onsets = np.asarray(ann.onset, dtype=float)
    descriptions = np.asarray(ann.description)
    event_onsets = np.sort(
        onsets[[d in event_names for d in descriptions]], kind="stable"
    )
    trial_onsets = np.sort(
        onsets[[d in trial_names for d in descriptions]], kind="stable"
    )

    trial_idx = np.searchsorted(trial_onsets, event_onsets, side="right")
    orphans = np.flatnonzero(trial_idx >= len(trial_onsets))
    if orphans.size:
        raise RuntimeError(
            f"{orphans.size} event(s) {list(event_conditions)} have no later trial "
            f"annotation (first at t={event_onsets[orphans[0]]:.3f} s)."
        )
    shared = np.flatnonzero(np.diff(trial_idx) == 0)
    if shared.size:
        raise RuntimeError(
            f"{shared.size} pair(s) of consecutive events {list(event_conditions)} "
            f"have no trial between them (first at t="
            f"{event_onsets[shared[0]]:.3f} s and "
            f"{event_onsets[shared[0] + 1]:.3f} s); refusing to pair them with "
            "the same trial."
        )
    lags = trial_onsets[trial_idx] - event_onsets
    return trial_idx.astype(int), lags


def trial_response_side_keep_first(
    raw,
    *,
    trial_conditions=("trial",),
    response_conditions=("response/left", "response/right"),
    max_latency=None,
    return_latencies=False,
):
    """Per-trial first-response **side** via :func:`mne.epochs.make_metadata`.

    Epochs on each trial annotation (the "row event") and records the *side*
    (e.g. ``'left'`` / ``'right'``) of the **first** response that follows it,
    using MNE's hierarchical-event-descriptor ``keep_first`` aggregation.  The
    per-trial window is ``[trial_onset, min(next_trial_onset, trial_onset +
    max_latency))`` — exactly the window used by :func:`first_response_per_trial`.
    ``make_metadata`` bounds the window by the next trial; a first response at
    or beyond ``max_latency`` is then discarded (no earlier response exists, so
    the trial has none).

    This deliberately uses a different code path than
    :func:`first_response_per_trial` so the two can be cross-checked against one
    another, and against the trial-wise responses recorded in the behavioral
    metadata, to confirm that the trial and response triggers are aligned.

    All trial annotations are collapsed to a single event type before building
    the events array.  This is required for the ``[trial_onset, next_trial_onset)``
    windowing: ``make_metadata(..., tmax=None)`` bounds each window by the next
    event **of the same type**, so trials carrying distinct hierarchical names
    (``trial/read_read`` vs ``trial/listen_read``) must share one id, otherwise
    a window would run to the end of the recording and steal a later trial's
    response.

    The ``raw`` object is **not** modified.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose ``annotations`` are inspected.
    trial_conditions : iterable of str
        Condition names identifying trial-onset annotations (matched
        hierarchically).
    response_conditions : iterable of str
        Condition names identifying response annotations (matched
        hierarchically).  These must share a single top-level group (the part
        before the first ``'/'``), e.g. ``'response'`` for
        ``('response/left', 'response/right')``.
    max_latency : float or None
        Response deadline in seconds after trial onset (exclusive).  ``None``
        bounds each window by the next trial onset only.
    return_latencies : bool
        Also return each first response's latency after its trial onset.

    Returns
    -------
    sides : list of (str or None)
        One entry per trial annotation, in chronological order; the response
        side (the part after the group prefix, lower-cased, e.g. ``'left'`` /
        ``'right'``) or ``None`` when the trial had no response in its window.
    latencies : numpy.ndarray of float
        Only if ``return_latencies``: the first response's latency (seconds)
        per trial, ``nan`` where ``sides`` is ``None``.
    """

    def _out(sides, latencies):
        if return_latencies:
            return sides, np.asarray(latencies, dtype=float)
        return sides

    ann = raw.annotations
    if len(ann) == 0:
        return _out([], [])

    unique_names = sorted(set(ann.description))

    def _matched(conds):
        try:
            names = mne.event.match_event_names(
                event_names=unique_names,
                keys=list(conds),
                on_missing="ignore",
            )
        except KeyError:
            return set()
        return set(names)

    trial_names = _matched(trial_conditions)
    response_names = _matched(response_conditions)

    sfreq = float(raw.info["sfreq"])

    # Collapse all trials to a single id; map each response to its own id under
    # a shared group so ``keep_first`` can aggregate them.
    groups = {name.split("/", 1)[0] for name in response_names}
    if len(groups) > 1:
        raise ValueError(
            "trial_response_side_keep_first requires response_conditions to "
            f"share a single top-level group; got {sorted(groups)}"
        )
    group = groups.pop() if groups else "response"

    event_id = {"trial": 1}
    for i, name in enumerate(sorted(response_names)):
        event_id[name] = i + 2

    onsets = np.asarray(ann.onset, dtype=float)
    descriptions = np.asarray(ann.description)

    rows = []
    for onset, desc in zip(onsets, descriptions):
        if desc in trial_names:
            code = 1
        elif desc in response_names:
            code = event_id[desc]
        else:
            continue
        rows.append((int(round(onset * sfreq)), 0, code))

    n_trials = sum(1 for _, _, code in rows if code == 1)
    if n_trials == 0:
        return _out([], [])

    # No responses anywhere: every trial is unanswered.
    if not response_names:
        return _out([None] * n_trials, [np.nan] * n_trials)

    events = np.asarray(sorted(rows), dtype=int)

    metadata, _, _ = mne.epochs.make_metadata(
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=None,
        sfreq=sfreq,
        row_events=["trial"],
        keep_first=[group],
    )

    # ``make_metadata`` names the group's first-event latency column after the
    # group itself, and the first event's name ``first_<group>``.
    first_col = f"first_{group}"
    latencies = metadata[group].to_numpy(dtype=float, copy=True)
    sides = []
    for i, value in enumerate(metadata[first_col].tolist()):
        missing = value is None or (isinstance(value, float) and np.isnan(value))
        if not missing and max_latency is not None and latencies[i] >= max_latency:
            missing = True
        if missing:
            sides.append(None)
            latencies[i] = np.nan
        else:
            sides.append(str(value).strip().lower())
    return _out(sides, latencies)


class ResponseAlignment(NamedTuple):
    """Outcome of :func:`assert_response_alignment`.

    Attributes
    ----------
    n_checked : int
        Trials whose trigger and behavioral responses were compared.
    n_skipped : int
        Anticipatory trials excluded from the comparison (see
        :func:`assert_response_alignment`).
    valid : numpy.ndarray of bool
        One entry per trial; ``False`` where the behavioral response is invalid
        because a held button made the task log a button release.
    """

    n_checked: int
    n_skipped: int
    valid: np.ndarray

    @property
    def n_invalid(self):
        """Number of trials whose behavioral response is invalid."""
        return int((~self.valid).sum())


def assert_response_alignment(
    raw,
    meta_df,
    column,
    *,
    trial_conditions=("trial",),
    response_conditions=("response/left", "response/right"),
    response_left=(),
    response_right=(),
    max_latency=None,
    rt_column=None,
    trigger_lag=0.03,
    max_invalid_fraction=0.01,
    context="",
    max_preview=10,
):
    """Verify per-trial trigger responses agree with a behavioral metadata column.

    Cross-checks the trigger stream against the behavioral log: for every trial
    annotation (in chronological order) the *side* of the first response that
    follows it (via :func:`trial_response_side_keep_first`) must equal the
    response side recorded in ``meta_df[column]``.  A disagreement means the
    trial/response triggers and the behavioral log are misaligned, which would
    silently corrupt the positional trial<->epoch metadata join performed by
    mne-bids-pipeline, so a :class:`RuntimeError` is raised to halt the run.

    With ``rt_column`` given, two trial-level disagreements that do *not*
    indicate misalignment are recognised from the behavioral reaction time
    ``rt`` and the trigger latency ``lat`` of the first response.  Response
    triggers lead the behavioral clock by a small hardware lag (5-23 ms in the
    TSX data), bounded by ``trigger_lag``:

    * **anticipatory** — the trigger stream has no response but the behavior
      does, with ``rt < trigger_lag``.  The press landed just *before* the
      trial trigger, so the trigger pairing gave it to the previous trial.  The
      trial is skipped (not compared) and counted in ``n_skipped``.
    * **held button** — both record a response but on different sides, and
      ``rt - lat > trigger_lag``: the behavior logged a *later* button event
      than the press.  With one button held down, a task that only accepts
      single-button states ignores the other button's press and logs its
      release as a press of the held button.  The behavioral response is
      invalid; the trial is marked ``False`` in ``valid``.

    A row shift makes about half of all sides disagree, so held-button trials
    above ``max_invalid_fraction`` of the trials raise instead.

    ``meta_df`` must be the **full per-trial** metadata (one row per trial,
    including unanswered trials), *not* the response-aligned subset — the
    per-trial first-response sides include ``None`` for unanswered trials, so
    the row counts only match before the response mask is applied.

    The ``raw`` object is **not** modified.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose annotations carry the trial and response triggers.
    meta_df : pandas.DataFrame
        Per-trial behavioral metadata (one row per trial).
    column : str
        Name of the column in ``meta_df`` holding the recorded response side.
    trial_conditions, response_conditions : iterable of str
        Condition names identifying trial / response annotations (matched
        hierarchically), forwarded to :func:`trial_response_side_keep_first`.
    response_left, response_right : str or iterable of str
        Metadata token(s) denoting a left / right response (e.g. ``'z'`` /
        ``'r'``).  Used to map recorded values onto ``'left'`` / ``'right'``
        before comparison.  A scalar string is treated as a single token.
    max_latency : float or None
        Response deadline (seconds), forwarded to
        :func:`trial_response_side_keep_first`; must match the one used to
        select the responses.
    rt_column : str or None
        Column in ``meta_df`` holding the behavioral reaction time (seconds).
        ``None`` disables the anticipatory / held-button classification, so
        every disagreement raises.
    trigger_lag : float
        Upper bound (seconds) on how far response triggers lead the
        behavioral reaction time.
    max_invalid_fraction : float
        Largest fraction of trials that may be classified as held-button.
    context : str
        Optional label prepended to log / error messages.
    max_preview : int
        Maximum number of mismatching trials to list in the error message.

    Returns
    -------
    alignment : ResponseAlignment
        Counts of compared / skipped trials and the per-trial validity mask.

    Raises
    ------
    ValueError
        If ``column`` or ``rt_column`` is absent from ``meta_df``.
    RuntimeError
        If the trial count disagrees with the metadata row count, any trial's
        first-response side does not match the metadata column (and is not
        explained as above), or too many trials are held-button.
    """
    prefix = f"[{context}] " if context else ""

    for name in (column, rt_column):
        if name is not None and name not in meta_df.columns:
            raise ValueError(
                f"{prefix}response-metadata column {name!r} not found in metadata "
                f"columns: {list(meta_df.columns)}"
            )

    def _tokens(value):
        items = [value] if isinstance(value, str) else list(value)
        return {str(t).strip().lower() for t in items if t is not None}

    left_tokens = _tokens(response_left)
    right_tokens = _tokens(response_right)

    def _norm(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        text = str(value).strip().lower()
        if text in ("", "none", "nan", "n/a", "na"):
            return None
        if text in left_tokens:
            return "left"
        if text in right_tokens:
            return "right"
        return text

    sides, latencies = trial_response_side_keep_first(
        raw,
        trial_conditions=trial_conditions,
        response_conditions=response_conditions,
        max_latency=max_latency,
        return_latencies=True,
    )
    recorded = list(meta_df[column])

    if len(sides) != len(recorded):
        raise RuntimeError(
            f"{prefix}trial/metadata count mismatch — {len(sides)} trial events "
            f"(keep_first) vs {len(recorded)} metadata rows in column "
            f"{column!r}. Trials and responses are not aligned."
        )

    if rt_column is None:
        rts = np.full(len(sides), np.nan)
    else:
        rts = pd.to_numeric(meta_df[rt_column], errors="coerce").to_numpy(dtype=float)

    valid = np.ones(len(sides), dtype=bool)
    n_skipped = 0
    mismatches = []
    for i, (trigger, rec) in enumerate(zip(sides, recorded)):
        trig_side, rec_side = _norm(trigger), _norm(rec)
        if trig_side == rec_side:
            continue
        if trig_side is None and rec_side is not None and rts[i] < trigger_lag:
            n_skipped += 1  # anticipatory: press preceded the trial trigger
        elif (
            trig_side is not None
            and rec_side is not None
            and rts[i] - latencies[i] > trigger_lag
        ):
            valid[i] = False  # held button: behavior logged a later release
        else:
            mismatches.append((i, trigger, rec))

    if mismatches:
        preview = ", ".join(
            f"row {i}: keep_first={trigger!r} vs metadata={rec!r}"
            for i, trigger, rec in mismatches[:max_preview]
        )
        raise RuntimeError(
            f"{prefix}{len(mismatches)}/{len(sides)} trial(s) where the "
            f"keep_first response side disagrees with metadata column "
            f"{column!r}. Trials and responses are not aligned. "
            f"First mismatches: {preview}"
        )

    n_invalid = int((~valid).sum())
    if n_invalid > max_invalid_fraction * len(sides):
        rows = np.flatnonzero(~valid)[:max_preview].tolist()
        raise RuntimeError(
            f"{prefix}{n_invalid}/{len(sides)} trial(s) where the behavioral "
            f"response in {column!r} is a later, different button event than the "
            f"first response trigger — more than {max_invalid_fraction:.1%} of "
            f"trials, which suggests misalignment rather than held buttons. "
            f"First rows: {rows}"
        )

    return ResponseAlignment(len(sides) - n_skipped, n_skipped, valid)


def drop_response_rows_from_events_tsv(
    events_tsv_path,
    keep_onsets,
    response_conditions=("response/left", "response/right"),
    *,
    tol_sec: float = 0.02,
) -> int:
    """Trim response rows from a BIDS events.tsv down to a kept set of onsets.

    Counterpart to :func:`trim_raw_to_events_tsv`: instead of removing raw
    annotations that are absent from the events.tsv, this removes *response*
    rows from the events.tsv that are absent from ``keep_onsets`` (the first
    response per trial).  All non-response rows are left untouched.

    Each kept onset greedily claims the nearest unclaimed response row within
    ``tol_sec``; any response row left unclaimed is dropped.  The file is
    rewritten in place, preserving the original cell formatting (read as strings).

    Parameters
    ----------
    events_tsv_path : str or Path
        Path to a BIDS ``*_events.tsv`` file (modified in place).
    keep_onsets : array-like of float
        Onsets (seconds) of the responses to keep.
    response_conditions : iterable of str
        Condition names identifying response rows (matched hierarchically).
    tol_sec : float
        Maximum onset difference (seconds) for a kept onset and a TSV row to be
        considered the same event (default 0.02 s).

    Returns
    -------
    n_removed : int
        Number of response rows removed from the events.tsv.
    """
    events_path = Path(events_tsv_path)
    if not events_path.exists():
        return 0

    # Read as strings (keep_default_na=False) so the round-trip preserves the
    # exact original formatting, including BIDS "n/a" cells.
    events_df = pd.read_csv(events_path, sep="\t", dtype=str, keep_default_na=False)
    if "trial_type" not in events_df.columns or "onset" not in events_df.columns:
        return 0

    descriptions = events_df["trial_type"].astype(str)
    unique_names = sorted(set(descriptions[descriptions != "n/a"]))
    try:
        matched_names = set(
            mne.event.match_event_names(
                event_names=unique_names,
                keys=list(response_conditions),
                on_missing="ignore",
            )
        )
    except KeyError:
        matched_names = set()

    if not matched_names:
        return 0

    resp_row_mask = descriptions.isin(matched_names).to_numpy()
    resp_rows = np.where(resp_row_mask)[0]
    if not resp_rows.size:
        return 0
    resp_onsets = events_df["onset"].to_numpy(dtype=float)[resp_rows]

    keep_onsets = np.asarray(list(keep_onsets), dtype=float)

    # Greedy nearest-neighbour: each kept onset claims the closest unclaimed
    # response row within tol_sec.
    claimed = np.zeros(len(resp_rows), dtype=bool)
    for k_on in keep_onsets:
        dists = np.abs(resp_onsets - k_on)
        dists[claimed] = np.inf
        best = int(np.argmin(dists))
        if dists[best] <= tol_sec:
            claimed[best] = True

    drop_rows = {int(resp_rows[i]) for i in np.where(~claimed)[0]}
    n_removed = len(drop_rows)
    if n_removed == 0:
        return 0

    keep_mask = np.array([i not in drop_rows for i in range(len(events_df))])
    events_df.loc[keep_mask].to_csv(events_path, sep="\t", index=False)

    return n_removed


def trim_raw_to_events_tsv(
    raw,
    events_tsv_path,
    conditions=("trial",),
    *,
    tol_sec: float = 0.02,
    context: str = "",
) -> int:
    """Remove condition-matching annotations from *raw* that are absent from
    the events.tsv.

    Recordings sometimes start or end while triggers are still firing,
    producing extra condition annotations in the raw that have no
    corresponding row in the authoritative BIDS events.tsv.  Left
    uncorrected, these orphan annotations cause the derivative FIF to carry
    more events than the metadata, breaking downstream metadata ↔ epochs
    alignment.

    This function greedily matches each events.tsv onset to the nearest
    unmatched raw annotation of the same condition, then removes any
    raw annotations that were not matched.

    The raw is modified **in place** (only ``raw.annotations`` is changed).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose annotations will be trimmed.
    events_tsv_path : str or Path
        Path to a BIDS ``*_events.tsv`` file.  Used as the authoritative list
        of condition onsets.
    conditions : iterable of str
        Condition names matched hierarchically (default ``('trial',)``).
        Non-matching annotations are never removed.
    tol_sec : float
        Maximum onset difference (seconds) for two events to be considered
        the same (default 0.02 s).
    context : str
        Short label for log messages.

    Returns
    -------
    n_removed : int
        Number of annotations removed from *raw*.

    Raises
    ------
    RuntimeError
        If, after trimming, *raw* has fewer condition annotations than the
        events.tsv (which would indicate data loss, not just trailing
        triggers).
    """
    events_tsv_path = Path(events_tsv_path)
    if not events_tsv_path.exists():
        return 0

    tsv_n, tsv_matched = count_condition_events_in_tsv(events_tsv_path, conditions)
    raw_n, matched_names = count_condition_events_in_raw(raw, conditions)

    # If the tsv has no matching rows at all it may simply not exist yet or may
    # not carry condition annotations — skip silently rather than treating every
    # raw annotation as orphan.
    if tsv_n == 0:
        return 0

    if raw_n == tsv_n:
        return 0  # already aligned

    prefix = f"[trim_raw_to_events_tsv{':' + context if context else ''}]"

    if raw_n < tsv_n:
        raise RuntimeError(
            f"{prefix} raw has fewer condition events ({raw_n}) than "
            f"events.tsv ({tsv_n}) — cannot trim; data may be missing."
        )

    # Build authoritative onset list from events.tsv.
    events_df = pd.read_csv(events_tsv_path, sep="\t")
    descriptions = events_df["trial_type"].astype(str)
    tsv_cond_mask = descriptions.isin(matched_names)
    tsv_onsets = events_df.loc[tsv_cond_mask, "onset"].to_numpy(dtype=float)

    # Collect indices of condition-matching annotations in raw.
    ann = raw.annotations
    cond_ann_idx = np.array(
        [i for i, d in enumerate(ann.description) if d in matched_names]
    )
    raw_onsets = ann.onset[cond_ann_idx]

    # Greedy nearest-neighbour matching: each tsv_onset claims the closest
    # unclaimed raw onset within tol_sec.
    used = np.zeros(len(raw_onsets), dtype=bool)
    for t_on in tsv_onsets:
        dists = np.abs(raw_onsets - t_on)
        dists[used] = np.inf
        best = int(np.argmin(dists))
        if dists[best] <= tol_sec:
            used[best] = True

    unmatched_local = np.where(~used)[0]
    n_removed = len(unmatched_local)

    if n_removed == 0:
        return 0

    # Build keep mask over ALL raw annotations.
    unmatched_global = set(cond_ann_idx[unmatched_local])
    keep_mask = np.array([i not in unmatched_global for i in range(len(ann))])
    new_ann = mne.Annotations(
        onset=ann.onset[keep_mask],
        duration=ann.duration[keep_mask],
        description=ann.description[keep_mask],
        orig_time=ann.orig_time,
    )
    raw.set_annotations(new_ann)

    print(
        f"{prefix} removed {n_removed} orphan condition annotation(s) "
        f"(raw had {raw_n}, events.tsv has {tsv_n}); "
        f"raw now has {tsv_n} condition events."
    )
    return n_removed


def verify_event_count_after_write(
    raw_before,
    bids_path,
    conditions=("trial",),
    *,
    context: str = "write",
) -> None:
    """Verify that a freshly-written BIDS/derivative file preserves event counts.

    Reads back the written FIF (and, when available, the matching
    ``events.tsv``) and confirms that the number of condition-matching
    annotations equals the count in ``raw_before``.  Raises
    ``RuntimeError`` with a detailed diagnostic on mismatch.

    Use this immediately after ``mne_bids.write_raw_bids`` or
    ``raw.save()`` to catch silent data loss before downstream steps run.

    Parameters
    ----------
    raw_before : mne.io.BaseRaw
        The raw object that was just written.
    bids_path : BIDSPath
        Path the data was written to.  Must point at the FIF; sidecars
        are derived from it.
    conditions : iterable of str
        Condition names matched hierarchically against annotation
        descriptions (default ``('trial',)``).
    context : str
        Short label for the error message (e.g. ``"format_bids"``,
        ``"bad_segments"``) so the user can locate the offending step.
    """
    expected_n, _ = count_condition_events_in_raw(raw_before, conditions)

    # 1. Re-read the FIF and recount.
    fif_path = Path(bids_path.fpath)
    if not fif_path.exists():
        raise RuntimeError(
            f"[{context}] verify_event_count_after_write: written FIF "
            f"not found at {fif_path}"
        )
    raw_after = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    fif_n, _ = count_condition_events_in_raw(raw_after, conditions)
    del raw_after

    # 2. If an events.tsv exists alongside, recount that too.
    events_tsv = (
        bids_path.copy()
        .update(suffix="events", extension=".tsv", split=None, check=False)
        .fpath
    )
    tsv_n = None
    if events_tsv.exists():
        tsv_n, _ = count_condition_events_in_tsv(events_tsv, conditions)

    # 3. Compare; raise on any divergence.
    summary = (
        f"[{context}] event-count verification "
        f"({'/'.join(conditions)} matches):\n"
        f"    raw before write : {expected_n}\n"
        f"    FIF after write  : {fif_n}\n"
        f"    events.tsv       : "
        f"{tsv_n if tsv_n is not None else '(absent)'}\n"
        f"    file             : {fif_path}"
    )

    if fif_n != expected_n or (tsv_n is not None and tsv_n != expected_n):
        raise RuntimeError(
            f"Event-count mismatch detected after write — this will "
            f"break metadata ↔ epochs alignment downstream.\n{summary}"
        )

    # Successful match: log so the user can see the check passed.
    print(summary)


def read_raw_bids_with_retry(bids_path, extra_params=None, max_retries=10):
    """Read raw BIDS data, dispatching to the right reader.

    Derivative files (``suffix="raw"``) are loaded directly with
    ``mne.io.read_raw_fif``, following the mne-bids-pipeline convention
    (see ``_04_frequency_filter.py``).  This avoids ``scans.tsv``
    validation and sidecar-suffix conflicts introduced by ``write_raw_bids``.

    Source files (``suffix="meg"``) use ``mne_bids.read_raw_bids`` with
    retry logic for NFS race conditions.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path to the BIDS file.  The ``suffix`` attribute determines the
        reader: ``"raw"`` → ``read_raw_fif``, anything else → ``read_raw_bids``.
    extra_params : dict, optional
        Extra parameters forwarded to the raw reader (e.g.
        ``{"preload": True}``).
    max_retries : int
        Maximum number of attempts for ``read_raw_bids`` (default 10).

    Returns
    -------
    raw : mne.io.Raw
        The loaded raw object.
    """
    if extra_params is None:
        extra_params = {}

    # Derivative files — load directly; no BIDS sidecar validation needed.
    if getattr(bids_path, "suffix", None) == "raw":
        return mne.io.read_raw_fif(bids_path.fpath, **extra_params)

    # Source files — retry on transient NFS read errors.
    for attempt in range(max_retries):
        try:
            return mne_bids.read_raw_bids(bids_path, extra_params=extra_params)
        except (IndexError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                delay = (2**attempt) + random.uniform(0, 2)
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
    events_tsv: Path = bp.copy().update(suffix="events", extension=".tsv").fpath
    events_json: Path = bp.copy().update(suffix="events", extension=".json").fpath

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
                    _delay = (2**_attempt) + random.uniform(0, 2)
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


def _is_relative_to(path: Path, other: Path) -> bool:
    """Return True if ``path`` equals ``other`` or lives beneath it.

    ``pathlib.Path.is_relative_to`` was only added in Python 3.9; this helper
    keeps the comparison explicit and resolves both sides first so that
    symlinks and ``..`` segments cannot smuggle a write past the check.
    """
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except ValueError:
        return False


def assert_not_raw_bids_write(target, cfg: SimpleNamespace, context: str = "") -> None:
    """Raise if *target* would write into the raw BIDS data directory.

    Custom preprocessing steps must only ever write into ``deriv_root`` (the
    derivatives tree).  The raw BIDS recordings under ``bids_root`` are the
    canonical, immutable inputs to the pipeline; mutating them silently
    corrupts the source data and makes re-runs non-reproducible.

    ``deriv_root`` conventionally lives *inside* ``bids_root`` (e.g.
    ``<bids_root>/derivatives/<analysis>``), so a path under ``deriv_root`` is
    explicitly allowed even though it is also under ``bids_root``.

    Parameters
    ----------
    target : str or Path or BIDSPath
        The path (or BIDSPath) about to be written.
    cfg : SimpleNamespace
        Configuration object exposing ``bids_root`` and ``deriv_root``.
    context : str
        Short label for the error message (e.g. the calling function).

    Raises
    ------
    RuntimeError
        If *target* resolves to a location inside ``bids_root`` that is not
        inside ``deriv_root``.
    """
    # Resolve a filesystem path from whatever we were handed.
    fpath = getattr(target, "fpath", None)
    target_path = Path(fpath if fpath is not None else target)

    bids_root = getattr(cfg, "bids_root", None)
    if bids_root is None:
        return
    bids_root = Path(bids_root)

    deriv_root = getattr(cfg, "deriv_root", None)
    deriv_root = Path(deriv_root) if deriv_root is not None else None

    # Writes inside the derivatives tree are always allowed.
    if deriv_root is not None and _is_relative_to(target_path, deriv_root):
        return

    # Any other write inside the raw BIDS root is forbidden.
    if _is_relative_to(target_path, bids_root):
        prefix = f"[{context}] " if context else ""
        raise RuntimeError(
            f"{prefix}Refusing to write {target_path} inside the raw BIDS "
            f"data directory ({bids_root}). Custom preprocessing steps must "
            f"write to deriv_root ({deriv_root}); set cfg.custom_proc so that "
            f"outputs are routed to a proc-<label> derivative instead."
        )


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
                suffixes="raw",
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
        suffix="raw",
        check=False,
    )


def _seed_sidecars(source_bp: BIDSPath, output_bp: BIDSPath) -> None:
    """Copy BIDS sidecar files from source to output on first write.

    Copies ``_channels.tsv``, the data JSON, ``_events.tsv``, and
    ``_events.json`` from the source location to the output location.
    Each sidecar is only copied if it exists at the source and does not yet
    exist at the destination, making this safe to call on every save.

    MNE-BIDS searches for the datatype JSON (e.g. ``_meg.json``), including
    when reading a derivative named ``proc-<label>_raw.fif``. Older custom
    derivatives used ``_raw.json``; accept those as a source for migration.
    Copy root-level participants metadata too, so reference-run reads can
    restore subject information without warnings.

    The split entity is stripped from both sides so the lookup is correct
    regardless of whether source_bp refers to a split file.
    """
    def copy_if_missing(src: Path, dst: Path) -> None:
        if src == dst or not src.exists() or dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete file without overwriting a concurrent job's copy.
        with tempfile.TemporaryDirectory(dir=dst.parent) as tmp_dir:
            staged = Path(tmp_dir) / dst.name
            shutil.copy2(src, staged)
            try:
                os.link(staged, dst)
            except FileExistsError:
                pass

    source_datatype = source_bp.datatype or "meg"
    data_suffix = output_bp.datatype or source_datatype

    # (source suffix, destination suffix, extension)
    sidecars = [
        ("channels", "channels", ".tsv"),
        (source_datatype, data_suffix, ".json"),
        ("events", "events", ".tsv"),
        ("events", "events", ".json"),
    ]
    for src_suffix, dst_suffix, ext in sidecars:
        src = (
            source_bp.copy()
            .update(suffix=src_suffix, extension=ext, split=None, check=False)
            .fpath
        )
        dst = (
            output_bp.copy()
            .update(suffix=dst_suffix, extension=ext, split=None, check=False)
            .fpath
        )
        if ext == ".json" and src_suffix == source_datatype and not src.exists():
            inherited = source_bp.find_matching_sidecar(
                suffix=source_datatype, extension=".json", on_error="ignore"
            )
            src = (
                Path(inherited)
                if inherited is not None
                else source_bp.copy().update(
                    suffix="raw", extension=".json", split=None, check=False
                ).fpath
            )
        copy_if_missing(src, dst)

    for name in ("participants.tsv", "participants.json"):
        copy_if_missing(Path(source_bp.root) / name, Path(output_bp.root) / name)


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
        src = source_bp.copy().update(suffix="events", extension=ext, check=False).fpath
        dst = output_bp.copy().update(suffix="events", extension=ext, check=False).fpath
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

    **Derivative mode** (``custom_proc`` is set):
    Follows the mne-bids-pipeline save pattern (see ``_04_frequency_filter.py``):
    seeds BIDS sidecars from the source on the first write, then saves the FIF
    data with ``raw.save(split_naming="bids")``.  This avoids the
    ``suffix=datatype`` override in ``write_raw_bids`` and all associated
    ``scans.tsv`` conflicts.

    **Legacy mode** (``custom_proc`` is None):
    Writes back to the source location via :func:`write_raw_bids_preserve_events`.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data to write.
    cfg : SimpleNamespace
        Configuration object.
    source_bp : BIDSPath
        Path the data was read from (used as base for output path and
        sidecar seeding).
    empty_room : BIDSPath or None
        Empty-room association.  Only used in legacy mode; silently ignored
        in derivative mode (the ``emptyroommatch.json`` is seeded once from
        the source and not updated thereafter).
    **extra_write_kwargs
        Extra keyword arguments forwarded to
        :func:`write_raw_bids_preserve_events` in legacy mode only.

    Returns
    -------
    output_bp : BIDSPath
        The BIDSPath the data was written to (base path, split=None).
    """
    output_bp = get_custom_output_path(cfg, source_bp)
    proc = get_custom_proc(cfg)

    # Safety invariant: a custom step must never overwrite the raw BIDS data.
    # This guards against a misconfigured custom_proc (or a future change to
    # the routing helpers) silently writing back into bids_root.
    assert_not_raw_bids_write(output_bp, cfg, context="write_raw_bids_custom_step")

    if proc is not None:
        # Derivative mode — follow the mne-bids-pipeline save pattern:
        #   - Seed BIDS sidecars (channels.tsv, meg.json, events.*) from the
        #     source on first write; subsequent writes are a no-op here.
        #   - Save FIF data with raw.save() + split_naming="bids" so that
        #     FIF split-file headers reference _split-02_raw.fif, not _meg.fif.
        #     This avoids all scans.tsv / suffix conflicts from write_raw_bids.
        # See _04_frequency_filter.py in mne-bids-pipeline for the same pattern.
        _seed_sidecars(source_bp, output_bp)

        # Before writing, trim any condition annotations that are absent from
        # the seeded events.tsv.  Recordings sometimes carry trailing or leading
        # trigger events that were never part of the task (e.g. the recording
        # started/stopped while triggers were firing).  These orphan annotations
        # would make the derivative FIF disagree with the authoritative
        # events.tsv, breaking metadata ↔ epochs alignment downstream.
        #
        # Always trim BOTH the epoch-locked conditions (cfg.conditions) AND
        # trial-onset annotations (_trial_conditions, default "trial").  The
        # epoch-locked conditions handle stray response presses; the trial
        # conditions handle extra trigger pulses from the OPM recording
        # system that appear in raw.annotations but were filtered out during
        # the initial BIDS conversion and are absent from the events.tsv.
        task_name_for_trim = getattr(output_bp, "task", None) or ""
        if not task_name_for_trim.startswith("noise"):
            _epoch_conds = tuple(getattr(cfg, "conditions", ("trial",)) or ("trial",))
            _trial_conds = tuple(
                getattr(cfg, "_trial_conditions", ("trial",)) or ("trial",)
            )
            conditions_for_trim = tuple(set(_epoch_conds + _trial_conds))
            _events_tsv_for_trim = (
                output_bp.copy()
                .update(suffix="events", extension=".tsv", split=None, check=False)
                .fpath
            )
            trim_raw_to_events_tsv(
                raw,
                _events_tsv_for_trim,
                conditions=conditions_for_trim,
                context=f"write_raw_bids_custom_step:{task_name_for_trim or '?'}",
            )

        output_bp.split = None
        output_bp.fpath.parent.mkdir(parents=True, exist_ok=True)
        raw.save(output_bp.fpath, overwrite=True, split_naming="bids")

        # With split_naming="bids", MNE writes _split-01_raw.fif (not the
        # nominal unsplit path) when data is large enough to be split.
        # Resolve the actual first file so verification and callers can find it.
        if not output_bp.fpath.exists():
            split01_bp = output_bp.copy().update(split="01", check=False)
            if split01_bp.fpath.exists():
                output_bp = split01_bp
    else:
        # Legacy mode: write back to the source location unchanged.
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

    # Post-write verification — catch silent annotation loss before later
    # pipeline steps fail with a confusing metadata-mismatch error.  We skip
    # the noise/empty-room task because it carries no condition annotations.
    task_name = getattr(output_bp, "task", None) or ""
    if not task_name.startswith("noise"):
        _epoch_conds = tuple(getattr(cfg, "conditions", ("trial",)) or ("trial",))
        _trial_conds = tuple(
            getattr(cfg, "_trial_conditions", ("trial",)) or ("trial",)
        )
        conditions = tuple(set(_epoch_conds + _trial_conds))
        verify_event_count_after_write(
            raw,
            output_bp,
            conditions=conditions,
            context=f"write_raw_bids_custom_step:{task_name or '?'}",
        )

    return output_bp


def get_ica_bids_path(cfg: SimpleNamespace) -> BIDSPath:
    """Resolve the pipeline's shared ICA path, with legacy task-file support.

    Current mne-bids-pipeline fits ICA across tasks and omits task and run
    entities. Prefer that solution when present. Older task-specific fits
    remain usable; an existing components TSV also identifies their layout
    when saving an ICA object for the first time.
    """
    subject = cfg.subjects[0] if isinstance(cfg.subjects, list) else cfg.subjects
    session = cfg.sessions[0] if isinstance(cfg.sessions, list) else cfg.sessions
    shared = BIDSPath(
        root=cfg.deriv_root,
        subject=subject,
        session=session,
        acquisition=getattr(cfg, "acq", None),
        recording=getattr(cfg, "rec", None),
        space=getattr(cfg, "space", None),
        datatype=getattr(cfg, "datatype", "meg"),
        suffix="ica",
        processing="ica",
        extension=".fif",
        check=False,
    )
    legacy = shared.copy().update(task=cfg.task)
    for path in (shared, legacy):
        if path.fpath.exists():
            return path
    for path in (shared, legacy):
        if path.copy().update(suffix="components", extension=".tsv").fpath.exists():
            return path
    return shared


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
    ica_path = get_ica_bids_path(cfg)
    tsv_path = ica_path.copy().update(suffix="components", extension=".tsv")

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
                    print(
                        f"Warning: 'status_description' column not found in {tsv_path.fpath}. Skipping description update."
                    )
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
            bids_path = get_bids_path_for_task(cfg, task=task, from_derivatives=False)
        else:
            bids_path = (
                get_bids_path_for_task(cfg, task=task, from_derivatives=False)
                .copy()
                .update(
                    root=cfg.deriv_root,
                    processing=proc,
                    check=False,
                )
            )

    # Never mark channels on the raw BIDS recordings — only on derivatives.
    assert_not_raw_bids_write(bids_path, cfg, context="mark_bad_channels_bids")

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
