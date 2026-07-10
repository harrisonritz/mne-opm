"""Bad segment detection for OPM-MEG data.

This module detects and annotates bad segments in raw MEG data using
statistical methods from the OSL-ephys library. Bad segments are time
periods with abnormally high or low signal variance, often caused by
movement artifacts, sensor noise, or other transient issues.

Staged Detection
----------------
The module supports two stages, called at different points in the
preprocessing pipeline (see ``run_preproc.sh``):

  * **Stage 1** (``--analysis=bad_segments_1``, before spatial filtering):
    Coarse detection with a lenient threshold.  Catches gross artifacts
    such as sensor dropouts, large head movements, or environmental
    transients.  Longer segment windows are appropriate here.

  * **Stage 2** (``--analysis=bad_segments_2``, after spatial filtering):
    Fine detection with a stricter threshold.  Catches smaller transients
    (muscle bursts, brief sensor pops) that become visible after HFC/ZCA
    has removed external interference.  Shorter segment windows help
    localise these brief events.

The legacy ``--analysis=bad_segments`` is still supported and falls back
to the Stage-2 defaults if ``_bad_segments_params`` is not configured.

Detection is performed on **bandpass-filtered** data (using the global
``l_freq``/``h_freq``) so that detection focuses on the frequency range
of interest.  The resulting annotations are then transferred back to the
**unfiltered** raw data that is saved to BIDS.

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    l_freq : float
        High-pass filter frequency used for detection filtering.
    h_freq : float
        Low-pass filter frequency used for detection filtering.
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.

Optional:
    _bad_segments_params : dict
        Per-stage parameters.  Keys are ``"1"`` and ``"2"``.
        Each value is a dict with:
            channel_threshold : float
                Fraction of channels that must be outliers for a segment
                to be marked bad.  Higher = more lenient.
            noise_channel_threshold : float
                Same, but for noise (empty-room) recordings.
            segment_len_sec : float
                Segment window length in seconds.
        See *config-trial.py* for a full example.
    process_empty_room : bool
        Also process empty room noise recording. Default: False.
    find_breaks : bool
        Annotate recording breaks before detection. Default: False.

Usage
-----
CLI::

    python src/custom/custom_preproc.py --analysis=bad_segments_1 --config=config.py
    python src/custom/custom_preproc.py --analysis=bad_segments_2 --config=config.py

Programmatic::

    >>> from preprocessing.bad_segments import run
    >>> run(cfg)  # uses cfg._bad_segments_stage if set

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import mne

from osl_ephys.preprocessing.osl_wrappers import bad_segments as osl_bad_segments

import mne_bids

from ._base import BaseAnalysis, SEGMENT_LEN_SEC
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


# -- Default per-stage parameters (used when config has no _bad_segments_params)
_DEFAULT_STAGE_PARAMS: Dict[str, Dict[str, float]] = {
    "1": {
        # Stage 1 (pre-spatial filter): lenient, long windows
        "channel_threshold": 0.20,
        "noise_channel_threshold": 0.50,
        "segment_len_sec": 1.0,
    },
    "2": {
        # Stage 2 (post-spatial filter): strict, shorter windows
        "channel_threshold": 0.05,
        "noise_channel_threshold": 0.30,
        "segment_len_sec": 0.5,
    },
}


class BadSegmentsAnalysis(BaseAnalysis):
    """Detect and annotate bad segments in raw MEG data.

    Uses OSL-ephys statistical detection to identify time segments with
    abnormal signal characteristics.  Detected segments are annotated
    as ``BAD_`` in the raw data annotations.

    Detection is performed on a **bandpass-filtered copy** of the data
    (using the global ``l_freq`` / ``h_freq`` from the config).  Only the
    resulting annotations are transferred back to the original unfiltered
    raw object that is saved to BIDS.

    Parameters are looked up from ``cfg._bad_segments_params[stage]`` when
    a stage is specified (via ``cfg._bad_segments_stage``).

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badsegments'
    ANALYSIS_NAME : str
        'bad_segments'
    stage : str or None
        Stage identifier (``"1"``, ``"2"``, or ``None`` for legacy mode).

    See Also
    --------
    osl_ephys.preprocessing.osl_wrappers.bad_segments : OSL detection function.
    """

    ANALYSIS_KEY = "badsegments"
    ANALYSIS_NAME = "bad_segments"

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__(cfg)
        self.stage: str | None = getattr(cfg, "_bad_segments_stage", None)
        if self.stage:
            self.ANALYSIS_NAME = f"bad_segments_{self.stage}"

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_stage_params(self) -> Dict[str, float]:
        """Return the parameter dict for the current stage.

        Resolution order:
          1. ``cfg._bad_segments_params[stage]``  (user-specified)
          2. Module-level ``_DEFAULT_STAGE_PARAMS[stage]``
          3. If no stage is set, fall back to stage-2 defaults.

        Returns
        -------
        params : dict
            Keys: ``channel_threshold``, ``noise_channel_threshold``,
            ``segment_len_sec``.
        """
        cfg_params = getattr(self.cfg, "_bad_segments_params", None) or {}

        if self.stage and self.stage in cfg_params:
            return cfg_params[self.stage]
        if self.stage and self.stage in _DEFAULT_STAGE_PARAMS:
            return _DEFAULT_STAGE_PARAMS[self.stage]
        # Legacy / unspecified stage → use stage-2 defaults
        return _DEFAULT_STAGE_PARAMS["2"]

    # ------------------------------------------------------------------
    # BaseAnalysis interface
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Check if bad segment detection is enabled.

        Returns
        -------
        enabled : bool
            Always True (no config flag required for this analysis).
        """
        return True

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for bad segment detection.

        Returns
        -------
        data : dict
            Dictionary with raw data per task.
        """
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        # Determine which tasks to load
        tasks = [self.cfg.task]
        if getattr(self.cfg, "process_empty_room", False):
            tasks.insert(0, "noise")

        for task in tasks:
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No raw data found for task={task}")

            raw = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
            data[task] = raw
            self.log(f"Loaded raw data for task={task} at {paths[0].fpath}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bad segment detection on all loaded data.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with annotated raw data for each task.
        """
        params = self._get_stage_params()
        self.log(
            f"Stage={self.stage or 'legacy'} | "
            f"channel_threshold={params['channel_threshold']}, "
            f"segment_len_sec={params['segment_len_sec']}"
        )

        results: Dict[str, Any] = {"bads": []}

        for task, raw in data.items():
            is_noise = task == "noise"
            self.log(f"Processing task={task} (is_noise={is_noise})")

            cleaned = self._detect_bad_segments(raw, is_noise=is_noise)
            results[task] = cleaned

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save annotated data back to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary with annotated raw data per task.
        """
        self.log("Saving results...")

        # Separate task data from metadata
        tasks = {k: v for k, v in results.items() if k not in {"bads"}}

        # Process noise FIRST (when present) so its saved location can be
        # used as the empty-room association when saving the task.
        ordered_tasks = sorted(tasks.items(), key=lambda kv: kv[0] != "noise")

        er_output_bp = None
        for task, raw in ordered_tasks:
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No file found for task={task} to save to")

            source_bp = paths[0]
            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task} → {output_bp.fpath}")

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def _detect_bad_segments(
        self, raw: mne.io.BaseRaw, is_noise: bool = False
    ) -> mne.io.BaseRaw:
        """Detect bad segments on filtered data; annotate the unfiltered original.

        A bandpass-filtered **copy** is used for the statistical detection
        so that low-frequency drifts and high-frequency noise do not
        contaminate the metric.  The ``BAD_`` annotations produced by OSL
        are then transferred back to the original (unfiltered) raw object.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process (returned with new annotations, unfiltered).
        is_noise : bool
            If True, this is empty room noise data.

        Returns
        -------
        raw : mne.io.BaseRaw
            The *same* unfiltered raw object, with bad-segment annotations
            appended.
        """
        params = self._get_stage_params()
        sfreq = raw.info["sfreq"]
        segment_len = round(sfreq * params["segment_len_sec"])

        # Select threshold
        if is_noise:
            threshold = params.get("noise_channel_threshold", 0.50)
        else:
            threshold = params["channel_threshold"]

        self.log(
            f"Detecting bad segments "
            f"(filter: {self.cfg.l_freq}-{self.cfg.h_freq} Hz, "
            f"seg={params['segment_len_sec']}s, thresh={threshold})"
        )

        # --- Annotate breaks (stage 1 only, task data only) --------------
        if (
            not is_noise
            and self.stage in (None, "1")
            and getattr(self.cfg, "find_breaks", False)
        ):
            mne.preprocessing.annotate_break(
                raw,
                min_break_duration=self.cfg.min_break_duration,
                t_start_after_previous=self.cfg.t_break_annot_start_after_previous_event,
                t_stop_before_next=self.cfg.t_break_annot_stop_before_next_event,
            )

        # --- Filter a copy for detection ---------------------------------
        filt = raw.copy().filter(
            l_freq=self.cfg.l_freq,
            h_freq=self.cfg.h_freq,
            method="iir",
        )

        # filt already inherits annotations from raw.copy() (breaks, prior BADs)
        # so osl_bad_segments will respect them during metric computation.

        n_annots_before = len(filt.annotations)

        # --- Run OSL detection on filtered copy --------------------------
        detected = osl_bad_segments(
            filt,
            picks=self.cfg.ch_types[0],
            ref_meg=False,
            metric="std",
            detect_zeros=False,
            channel_wise=True,
            segment_len=segment_len,
            channel_threshold=threshold,
        )

        # --- Transfer new annotations to unfiltered raw ------------------
        n_annots_after = len(detected.annotations)
        n_new = n_annots_after - n_annots_before

        if n_new > 0:
            new_onsets = detected.annotations.onset[n_annots_before:]
            new_durations = detected.annotations.duration[n_annots_before:]
            new_descriptions = list(detected.annotations.description[n_annots_before:])
            raw.annotations.append(new_onsets, new_durations, new_descriptions)

        # Count total bad annotations on the original raw
        bad_annots = [a for a in raw.annotations if a["description"].startswith("BAD")]
        self.log(
            f"Added {n_new} new bad-segment annotations "
            f"({len(bad_annots)} total BAD annotations on raw)"
        )

        del filt
        return raw


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = BadSegmentsAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
