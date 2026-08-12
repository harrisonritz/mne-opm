"""Generic two-condition trial schedule for the synthetic subject.

The design is deliberately *not* modelled on any particular experiment: it is
the smallest structure that exercises the pipeline's event machinery.

Each trial is

    ITI  ->  stimulus (``trial/cond_a`` or ``trial/cond_b``)  ->  response
             (``response/left`` or ``response/right``)        ->  feedback

with a small fraction of trials left unanswered so that
``select_trial_response`` has both matched and unmatched trials to handle, and
a rest break between blocks long enough for ``find_breaks`` to annotate.

Events reach the recording the way they do on a Cerca system: as eight
parallel-port bits.  ``format_bids.convert_triggers`` packs those bits back
into integers and maps them to annotation descriptions via ``trigger_desc``,
so :data:`TRIGGER_DESC` here and the ``trigger_desc`` in the shipped
``sub-XXX_config-bids.py`` must agree.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ["TRIGGER_DESC", "TrialSchedule", "build_schedule", "trigger_waveforms"]


#: Trigger code -> annotation description.  Powers of two so each event drives
#: exactly one of the eight trigger lines.
TRIGGER_DESC: dict[int, str] = {
    1: "ITI",
    2: "feedback",
    4: "trial/cond_a",
    8: "trial/cond_b",
    16: "response/left",
    32: "response/right",
}

#: Voltage a Cerca trigger line sits at while asserted (``convert_triggers``
#: thresholds at 2 V).
TRIGGER_HIGH_VOLTS = 5.0
TRIGGER_PULSE_SEC = 0.010


@dataclass
class TrialSchedule:
    """Event times, codes and per-trial metadata for one recording."""

    duration: float
    onsets: np.ndarray  # seconds, one entry per event
    codes: np.ndarray  # trigger code, aligned with ``onsets``
    metadata: "object"  # pandas.DataFrame, one row per trial

    @property
    def n_trials(self) -> int:
        return len(self.metadata)


def build_schedule(
    duration: float,
    *,
    seed: int = 0,
    n_blocks: int = 2,
    break_sec: float = 9.0,
    trial_sec: float = 2.2,
    iti_sec: float = 0.6,
    feedback_sec: float = 1.2,
    miss_rate: float = 0.05,
) -> TrialSchedule:
    """Lay out trials across ``duration`` seconds.

    Parameters
    ----------
    duration : float
        Total recording length, seconds.  Trials are placed until the last
        one's feedback would overrun; the remainder is trailing rest.
    seed : int
        Seed for conditions, responses and reaction times.
    n_blocks : int
        Number of blocks; ``break_sec`` of rest is inserted between them.
    break_sec : float
        Length of the between-block rest.  Keep this above the pipeline's
        ``min_break_duration`` so ``find_breaks`` has something to find.
    trial_sec : float
        Stimulus-onset asynchrony.
    iti_sec : float
        Gap between the ITI marker and the stimulus.
    feedback_sec : float
        Delay from stimulus to feedback.
    miss_rate : float
        Fraction of trials with no response.

    Returns
    -------
    schedule : TrialSchedule
    """
    import pandas as pd

    rng = np.random.default_rng(seed + 4242)

    lead_in = 3.0
    tail = 3.0
    usable = duration - lead_in - tail - break_sec * (n_blocks - 1)
    n_trials = max(int(usable // trial_sec), 1)
    per_block = max(n_trials // n_blocks, 1)

    onsets: list[float] = []
    codes: list[int] = []
    rows: list[dict] = []

    t = lead_in
    trial = 0
    for block in range(n_blocks):
        if block:
            t += break_sec
        for _ in range(per_block):
            stim = t + iti_sec
            if stim + feedback_sec + 0.5 > duration - tail + break_sec:
                break

            condition = "A" if rng.random() < 0.5 else "B"
            responded = rng.random() >= miss_rate
            # Response hand is informative about condition but not perfectly,
            # so a decoder has signal without the problem being trivial.
            correct = "left" if condition == "A" else "right"
            accurate = bool(rng.random() < 0.85)
            response = correct if accurate else ("right" if correct == "left" else "left")
            rt = float(rng.gamma(shape=6.0, scale=0.075) + 0.25)

            onsets.append(t)
            codes.append(1)  # ITI
            onsets.append(stim)
            codes.append(4 if condition == "A" else 8)
            if responded:
                onsets.append(stim + rt)
                codes.append(16 if response == "left" else 32)
            onsets.append(stim + feedback_sec)
            codes.append(2)  # feedback

            rows.append(
                dict(
                    trial=trial,
                    block=block + 1,
                    run=1,
                    condition=condition,
                    stim_onset=round(stim, 4),
                    response_onset=round(stim + rt, 4) if responded else np.nan,
                    response=response if responded else "none",
                    rt=round(rt, 4) if responded else np.nan,
                    accuracy=int(accurate) if responded else 0,
                    responded=int(responded),
                )
            )
            trial += 1
            t += trial_sec

    order = np.argsort(onsets, kind="stable")
    return TrialSchedule(
        duration=duration,
        onsets=np.asarray(onsets, float)[order],
        codes=np.asarray(codes, int)[order],
        metadata=pd.DataFrame(rows),
    )


def trigger_annotations(trigger_data: np.ndarray, sfreq: float):
    """Annotate each trigger-line rising edge with that line's channel name.

    A Cerca recording arrives already annotated this way (see
    ``cMEG_mne.set_events_annotations``), and ``format_bids.convert_triggers``
    relies on it: it strips descriptions containing a trigger channel name
    before appending the decoded event annotations.  The synthetic recording
    therefore has to carry them too.
    """
    import mne

    from .sensors import TRIGGER_CHANNELS

    high = trigger_data > 0.5
    onsets: list[float] = []
    descriptions: list[str] = []
    for line, name in enumerate(TRIGGER_CHANNELS):
        edges = np.flatnonzero(np.diff(high[line].astype(int)) == 1) + 1
        onsets.extend(edges / sfreq)
        descriptions.extend([name] * len(edges))

    order = np.argsort(onsets, kind="stable")
    return mne.Annotations(
        onset=np.asarray(onsets, float)[order],
        duration=np.zeros(len(onsets)),
        description=np.asarray(descriptions, dtype=object)[order],
    )


def trigger_waveforms(schedule: TrialSchedule, sfreq: float, n_times: int) -> np.ndarray:
    """Render a schedule onto eight parallel-port trigger lines.

    Returns
    -------
    data : ndarray, shape (8, n_times)
        Volts.  Bit ``i`` of a trigger code drives ``"Trigger i+1"``.
    """
    data = np.zeros((8, n_times), dtype=np.float64)
    width = max(int(round(TRIGGER_PULSE_SEC * sfreq)), 1)
    for onset, code in zip(schedule.onsets, schedule.codes, strict=True):
        start = int(round(onset * sfreq))
        stop = min(start + width, n_times)
        if start >= n_times:
            continue
        for bit in range(8):
            if code & (1 << bit):
                data[bit, start:stop] = TRIGGER_HIGH_VOLTS
    return data
