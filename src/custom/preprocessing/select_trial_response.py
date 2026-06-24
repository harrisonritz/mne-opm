"""Select the first response following each trial for response-locked analyses.

A response-locked analysis epochs on ``response/left`` / ``response/right``
annotations.  In the raw recordings the number of response events does **not**
match the number of trials: subjects sometimes press more than once per trial
(double presses), and some trials receive no response at all.  Because
mne-bids-pipeline joins per-trial behavioral metadata to epochs **positionally
by row count**, the responses must first be reduced to exactly **one per trial**
— the first response following each trial event — so that each remaining response
epoch corresponds 1:1 with the trial it belongs to.

This step rewrites the ``proc-<custom_proc>`` derivative so that only the first
response within each trial window ``[trial_onset, next_trial_onset)`` survives.
Extra presses and orphan responses (before the first trial / outside any window)
are removed from both ``raw.annotations`` **and** the derivative ``events.tsv``,
keeping the two consistent so the event-count guards in
:func:`custom.preprocessing._io.write_raw_bids_custom_step` pass.

The per-trial selection is performed by
:func:`custom.preprocessing._io.first_response_per_trial`, which is the single
source of truth shared with ``config-trialResponse.py`` (the config re-derives
the same per-trial mask from this reduced derivative to subset its metadata).

Alignment check
---------------
Verifying that the trial and response triggers are actually aligned with the
recorded behavior is a separate concern handled by
:func:`custom.preprocessing._io.assert_response_alignment`, which is invoked
from ``config-trialResponse.py`` once the per-trial metadata has been built.
That is the only place the **full per-trial** metadata (one row per trial,
including unanswered trials) and the trigger raw coexist — this step only sees
the raw, and the metadata it could reach via ``cfg.epochs_custom_metadata`` is
the response-aligned *subset*, whose row count no longer matches the per-trial
first-response sides.  Keeping the check in the config avoids that mismatch.

Gating
------
The step is a **no-op** unless ``cfg._select_trial_response`` is truthy.  This
makes it safe to call unconditionally from the shared ``run_preproc.sh``: the
trial- and response-locked analyses that do not set the flag are untouched.

Placement
---------
Run **after** the spatial-filter steps (HFC/ZCA) and **before**
``mne_bids_pipeline --steps=preprocessing`` (epoching).  Maxwell/frequency
filtering inside mne-bids-pipeline preserve annotations, so the reduced response
set propagates to epoching.

Usage
-----
CLI::

    python src/custom/custom_preproc.py --analysis=select_trial_response --config=config.py

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne
import numpy as np

from ._base import BaseAnalysis
from ._io import (
    _seed_sidecars,
    drop_response_rows_from_events_tsv,
    find_custom_input_paths,
    first_response_per_trial,
    get_custom_output_path,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


# Default condition labels used to identify trial and response annotations.
# Overridable via cfg._trial_conditions / cfg._response_conditions.
_DEFAULT_TRIAL_CONDITIONS = ("trial",)
_DEFAULT_RESPONSE_CONDITIONS = ("response/left", "response/right")


class SelectTrialResponseAnalysis(BaseAnalysis):
    """Reduce response annotations to the first response per trial.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'selecttrialresponse'
    ANALYSIS_NAME : str
        'select_trial_response'
    """

    ANALYSIS_KEY = "selecttrialresponse"
    ANALYSIS_NAME = "select_trial_response"

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__(cfg)
        self.trial_conditions = tuple(
            getattr(cfg, "_trial_conditions", _DEFAULT_TRIAL_CONDITIONS)
            or _DEFAULT_TRIAL_CONDITIONS
        )
        self.response_conditions = tuple(
            getattr(cfg, "_response_conditions", _DEFAULT_RESPONSE_CONDITIONS)
            or _DEFAULT_RESPONSE_CONDITIONS
        )

    # ------------------------------------------------------------------
    # BaseAnalysis interface
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Run only when ``cfg._select_trial_response`` is truthy.

        Returns
        -------
        enabled : bool
            True if response selection should run, False to skip (the default
            for trial-locked / standard response-locked analyses).
        """
        return bool(getattr(self.cfg, "_select_trial_response", False))

    def load_data(self) -> Dict[str, Any]:
        """Load the task raw derivative (empty-room/noise carries no responses).

        Returns
        -------
        data : dict
            Mapping of task name to its raw object.
        """
        paths = find_custom_input_paths(self.cfg, task=self.cfg.task)
        if not paths:
            raise FileNotFoundError(
                f"No raw data found for task={self.cfg.task}"
            )

        raw = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
        self.log(f"Loaded raw data for task={self.cfg.task} at {paths[0].fpath}")
        return {self.cfg.task: raw}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Drop all but the first response within each trial window.

        Parameters
        ----------
        data : dict
            Mapping from load_data() of task name to raw object.

        Returns
        -------
        results : dict
            Mapping of task name to ``{"raw": raw, "keep_onsets": ndarray}`` for
            save_results().
        """
        results: Dict[str, Any] = {}

        for task, raw in data.items():
            (
                trial_has_response,
                keep_ann_idx,
                drop_ann_idx,
                keep_onsets,
            ) = first_response_per_trial(
                raw,
                trial_conditions=self.trial_conditions,
                response_conditions=self.response_conditions,
            )

            n_trials = len(trial_has_response)
            n_responded = int(trial_has_response.sum())
            n_no_response = n_trials - n_responded
            n_kept = len(keep_ann_idx)
            n_dropped = len(drop_ann_idx)
            self.log(
                f"task={task}: {n_trials} trials | "
                f"{n_responded} with a response, {n_no_response} without | "
                f"kept {n_kept} first responses, dropped {n_dropped} "
                f"(extra presses + orphans)"
            )

            if drop_ann_idx:
                raw = self._drop_annotations(raw, drop_ann_idx)

            results[task] = {"raw": raw, "keep_onsets": keep_onsets}

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Write the reduced derivative, keeping events.tsv consistent.

        The derivative ``events.tsv`` is trimmed to the kept responses *before*
        the FIF is written so that the response-count trim/verify guards in
        :func:`write_raw_bids_custom_step` see a consistent FIF and events.tsv.

        Parameters
        ----------
        results : dict
            Mapping from run() of task name to ``{"raw", "keep_onsets"}``.
        """
        for task, payload in results.items():
            raw = payload["raw"]
            keep_onsets = payload["keep_onsets"]

            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(
                    f"No file found for task={task} to save to"
                )
            source_bp = paths[0]

            output_bp = get_custom_output_path(self.cfg, source_bp)

            # Seed the derivative sidecars (events.tsv, channels.tsv, …)
            # before trimming.  When select_trial_response is the first
            # custom step to write for this analysis the derivative
            # events.tsv will not yet exist; without seeding it first,
            # drop_response_rows_from_events_tsv returns 0 (file-not-found
            # early exit) and trim_raw_to_events_tsv then raises because
            # the freshly-seeded file still has all response rows while raw
            # has already been reduced.
            _seed_sidecars(source_bp, output_bp)

            events_tsv = output_bp.copy().update(
                suffix="events", extension=".tsv", split=None, check=False
            ).fpath
            n_removed = drop_response_rows_from_events_tsv(
                events_tsv,
                keep_onsets,
                response_conditions=self.response_conditions,
            )
            self.log(
                f"task={task}: removed {n_removed} response row(s) from "
                f"{events_tsv}"
            )

            written = write_raw_bids_custom_step(raw, self.cfg, source_bp)
            self.log(f"Saved task={task} → {written.fpath}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_annotations(
        raw: mne.io.BaseRaw, drop_idx: list[int]
    ) -> mne.io.BaseRaw:
        """Return *raw* with the annotations at ``drop_idx`` removed.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw object to modify (annotations replaced in place).
        drop_idx : list of int
            Indices into ``raw.annotations`` to remove.

        Returns
        -------
        raw : mne.io.BaseRaw
            The same raw object with a rebuilt ``Annotations`` set.
        """
        ann = raw.annotations
        drop = set(int(i) for i in drop_idx)
        keep_mask = np.array([i not in drop for i in range(len(ann))])
        new_ann = mne.Annotations(
            onset=ann.onset[keep_mask],
            duration=ann.duration[keep_mask],
            description=ann.description[keep_mask],
            orig_time=ann.orig_time,
        )
        raw.set_annotations(new_ann)
        return raw


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for the CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = SelectTrialResponseAnalysis(cfg)

    if not analysis.is_enabled():
        print(
            f"\n[{analysis.ANALYSIS_NAME}] _select_trial_response not set; "
            f"exiting (no-op)"
        )
        return

    analysis.execute()
