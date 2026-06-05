"""Bad channel detection for OPM-MEG data.

This module detects bad channels in raw MEG data by combining several
complementary detectors and a configurable consensus vote.  The available
detectors are:

* **gesd** (full-recording): the Generalized Extreme Studentized Deviate
  (GESD) test from OSL-ephys, applied to each channel's standard deviation
  computed over the whole recording.  Catches channels that are bad for the
  entire session.
* **timeresolved**: a windowed GESD test that flags channels which are
  variance outliers in a sufficiently large *fraction* of short time windows.
  This catches **intermittent** bad channels — channels that are only
  disruptive for part of the recording and therefore have a near-normal
  whole-recording standard deviation (these are typically what surface later
  as single-channel ICA components).
* **psd**: OSL-ephys' PSD/noise-floor detector, flagging channels with
  abnormal spectra.
* **lof**: MNE's Local Outlier Factor detector, flagging channels that are
  anomalous relative to their spatial neighbours.

Each enabled detector contributes a *vote* for a channel.  A channel is
confirmed bad (marked ``status="bad"`` in ``channels.tsv`` and added to
``raw.info['bads']``) when its vote count reaches ``_bad_channel_consensus_n``.
Channels with at least one vote but fewer than the consensus threshold are
written to a ``*_badchannel-candidates.tsv`` sidecar for manual confirmation
in the ``manual_channel`` step, but are **not** removed automatically.

Detection is performed on bandpass-filtered data (using the global
``l_freq``/``h_freq``) to focus on the frequency range of interest and avoid
contamination from low-frequency drifts or high-frequency noise.  The PSD
detector runs on the unfiltered data so it can see the full spectrum / noise
floor.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=bad_channels --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.bad_channels import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    l_freq : float
        High-pass filter frequency for detection.
    h_freq : float
        Low-pass filter frequency for detection.
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.

Optional (with conservative defaults):
    process_empty_room : bool
        Also process empty room noise recording. Default: False.
    _bad_channel_methods : list of str
        Detectors to run. Default: ['gesd', 'timeresolved', 'psd', 'lof'].
    _bad_channel_consensus_n : int
        Number of detectors that must agree to auto-mark a channel bad.
        Default: 2.  When set to 1, *any* flag marks a channel bad (and no
        candidates are produced).
    _bad_channel_significance_level : float
        GESD significance level for the full-recording and time-resolved
        detectors. Default: 0.05.
    _bad_channel_window_sec : float
        Window length (seconds) for the time-resolved detector. Default: 2.0.
    _bad_channel_frac_threshold : float
        Fraction of windows in which a channel must be an outlier for the
        time-resolved detector to flag it. Default: 0.20.
    _bad_channel_psd_fmin, _bad_channel_psd_fmax : float
        Frequency band (Hz) for the PSD detector. Defaults: 1.0, 100.0.
    _bad_channel_psd_nfft : int
        FFT length for the PSD detector. Default: 2000.
    _bad_channel_lof_neighbors : int
        Number of neighbours for the LOF detector. Default: 20.
    _bad_channel_lof_threshold : float
        LOF threshold. Default: 1.5.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple

import mne
import numpy as np
import pandas as pd

from osl_ephys.preprocessing.osl_wrappers import (
    bad_channels as osl_bad_channels,
    detect_bad_channels_psd as osl_detect_bad_channels_psd,
    gesd as osl_gesd,
)

import mne_bids

from ._base import BaseAnalysis
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)


# Default detectors and consensus parameters.
_DEFAULT_METHODS: List[str] = ["gesd", "timeresolved", "psd", "lof"]
_DEFAULT_CONSENSUS_N: int = 2
_DEFAULT_SIGNIFICANCE: float = 0.05
_DEFAULT_WINDOW_SEC: float = 2.0
_DEFAULT_FRAC_THRESHOLD: float = 0.20
_DEFAULT_PSD_FMIN: float = 1.0
_DEFAULT_PSD_FMAX: float = 100.0
_DEFAULT_PSD_NFFT: int = 2000
_DEFAULT_LOF_NEIGHBORS: int = 20
_DEFAULT_LOF_THRESHOLD: float = 1.5

# Suffix used for the per-recording candidates sidecar.
_CANDIDATES_SUFFIX = "_badchannel-candidates.tsv"


def candidates_sidecar_path(bids_path: mne_bids.BIDSPath) -> Path:
    """Return the candidates-TSV path that pairs with a given BIDSPath.

    The sidecar lives next to the (output) data file and is named after the
    BIDS basename so that downstream steps (e.g. ``manual_channel``) can locate
    it from their own input path.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path of the data file the candidates relate to.

    Returns
    -------
    pathlib.Path
        Path to the ``*_badchannel-candidates.tsv`` sidecar.
    """
    return Path(bids_path.directory) / f"{bids_path.basename}{_CANDIDATES_SUFFIX}"


class BadChannelsAnalysis(BaseAnalysis):
    """Detect bad channels via a consensus of complementary detectors.

    Runs the enabled detectors (full-recording GESD, time-resolved GESD, PSD
    noise-floor, and LOF), tallies a vote per channel, and splits the result
    into *confirmed* bad channels (vote count >= consensus threshold) and
    *candidate* bad channels (>= 1 vote but below the threshold).  Confirmed
    channels are marked bad in BIDS; candidates are written to a sidecar for
    manual review.

    When processing multiple tasks (e.g. main task + noise), votes are tallied
    per recording and the union of confirmed channels is marked in all
    recordings; the union of candidates (excluding confirmed) is written to the
    sidecar.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badchannels'
    ANALYSIS_NAME : str
        'bad_channels'

    See Also
    --------
    osl_ephys.preprocessing.osl_wrappers.bad_channels : OSL GESD detection.
    osl_ephys.preprocessing.osl_wrappers.detect_bad_channels_psd : PSD detector.
    mne.preprocessing.find_bad_channels_lof : Local Outlier Factor detector.
    """

    ANALYSIS_KEY = "badchannels"
    ANALYSIS_NAME = "bad_channels"

    def is_enabled(self) -> bool:
        """Check if bad channel detection is enabled.

        Returns
        -------
        enabled : bool
            Always True (no config flag required for this analysis).
        """
        return True

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _methods(self) -> List[str]:
        """Return the list of enabled detectors (lower-cased)."""
        methods = getattr(self.cfg, "_bad_channel_methods", None) or _DEFAULT_METHODS
        return [str(m).lower() for m in methods]

    def _consensus_n(self) -> int:
        """Return the consensus vote threshold (>= 1)."""
        return max(1, int(getattr(self.cfg, "_bad_channel_consensus_n", _DEFAULT_CONSENSUS_N)))

    def _significance(self) -> float:
        """Return the GESD significance level."""
        return float(
            getattr(self.cfg, "_bad_channel_significance_level", _DEFAULT_SIGNIFICANCE)
        )

    # ------------------------------------------------------------------
    # BaseAnalysis interface
    # ------------------------------------------------------------------

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for bad channel detection.

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
            # Search for raw files (handles runs, splits, etc.); honours
            # cfg.custom_proc so subsequent custom steps read from deriv.
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No raw data found for task={task}")

            raw = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
            data[task] = raw
            self.log(f"Loaded raw data for task={task} at {paths[0].fpath}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all detectors on each task and combine via consensus voting.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with raw data per task plus:
              * ``bads`` : sorted list of confirmed bad channels (union).
              * ``bad_methods`` : {channel -> sorted method list} for confirmed.
              * ``candidates`` : {channel -> sorted method list} for candidates.
        """
        consensus_n = self._consensus_n()
        self.log(
            f"Detectors={self._methods()} | consensus_n={consensus_n} | "
            f"significance={self._significance()}"
        )

        results: Dict[str, Any] = {}
        confirmed_methods: Dict[str, Set[str]] = {}
        candidate_methods: Dict[str, Set[str]] = {}

        for task, raw in data.items():
            self.log(f"Processing task={task}")
            method_results = self._detect_all_methods(raw)
            confirmed, candidates = self._combine_votes(method_results, consensus_n)

            for ch, methods in confirmed.items():
                confirmed_methods.setdefault(ch, set()).update(methods)
            for ch, methods in candidates.items():
                candidate_methods.setdefault(ch, set()).update(methods)

            results[task] = raw

        # A channel confirmed in any recording outranks a candidacy elsewhere.
        for ch in confirmed_methods:
            candidate_methods.pop(ch, None)

        results["bads"] = sorted(confirmed_methods)
        results["bad_methods"] = {
            ch: sorted(m) for ch, m in confirmed_methods.items()
        }
        results["candidates"] = {
            ch: sorted(m) for ch, m in candidate_methods.items()
        }

        self.log(
            f"Confirmed bad channels: {len(results['bads'])} "
            f"({results['bads']}); candidates: {len(results['candidates'])} "
            f"({sorted(results['candidates'])})"
        )

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results: mark confirmed bads in BIDS, write candidates sidecar.

        Parameters
        ----------
        results : dict
            Output of :meth:`run`.
        """
        self.log("Saving results...")

        confirmed = sorted(set(results.get("bads", []) or []))
        bad_methods: Dict[str, List[str]] = results.get("bad_methods", {}) or {}
        candidates: Dict[str, List[str]] = results.get("candidates", {}) or {}

        # Separate task data from metadata
        tasks = {
            k: v
            for k, v in results.items()
            if k not in {"bads", "bad_methods", "candidates"}
        }

        # Process noise FIRST (when present) so the task save can use the
        # already-written noise as its empty-room association.
        ordered_tasks = sorted(tasks.items(), key=lambda kv: kv[0] != "noise")

        er_output_bp = None
        for task, raw in ordered_tasks:
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No file found for task={task}")

            source_bp = paths[0]

            # Merge existing and newly confirmed bad channels into raw.info,
            # which write_raw_bids will reflect in the output channels.tsv.
            if confirmed:
                existing_bads = raw.info.get("bads", [])
                merged_bads = sorted(set(existing_bads) | set(confirmed))
                raw.info["bads"] = merged_bads
                self.log(f"task={task}: {len(merged_bads)} total bad channels")

            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            # Tag the confirmed bad channels in channels.tsv with a per-channel
            # description recording which detectors agreed (e.g. "auto:gesd+lof").
            if confirmed:
                descriptions = [
                    "auto:" + "+".join(bad_methods.get(ch, ["auto"]))
                    for ch in confirmed
                ]
                mne_bids.mark_channels(
                    bids_path=output_bp,
                    ch_names=confirmed,
                    status="bad",
                    descriptions=descriptions,
                )

            # Write the candidates sidecar next to the output for manual review.
            self._write_candidates_sidecar(output_bp, candidates)

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task} → {output_bp.fpath}")

    # ------------------------------------------------------------------
    # Consensus combination
    # ------------------------------------------------------------------

    def _combine_votes(
        self, method_results: Dict[str, Set[str]], consensus_n: int
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Tally per-channel votes and split confirmed vs candidate channels.

        Parameters
        ----------
        method_results : dict
            {method_name -> set of flagged channel names}.
        consensus_n : int
            Number of agreeing detectors required to confirm a channel.

        Returns
        -------
        confirmed : dict
            {channel -> set of methods} for channels with >= consensus_n votes.
        candidates : dict
            {channel -> set of methods} for channels with 1 <= votes < consensus_n.
        """
        votes: Dict[str, Set[str]] = {}
        for method, chans in method_results.items():
            for ch in chans:
                votes.setdefault(ch, set()).add(method)

        confirmed: Dict[str, Set[str]] = {}
        candidates: Dict[str, Set[str]] = {}
        for ch, methods in votes.items():
            if len(methods) >= consensus_n:
                confirmed[ch] = methods
            else:
                candidates[ch] = methods

        return confirmed, candidates

    def _write_candidates_sidecar(
        self, output_bp: mne_bids.BIDSPath, candidates: Dict[str, List[str]]
    ) -> None:
        """Write (or clear) the candidates TSV sidecar for one recording.

        Parameters
        ----------
        output_bp : mne_bids.BIDSPath
            Path the recording was written to.
        candidates : dict
            {channel -> list of methods} for candidate channels.
        """
        try:
            sidecar = candidates_sidecar_path(output_bp)
        except (TypeError, ValueError):  # e.g. non-filesystem BIDSPath in tests
            return

        if not candidates:
            # Remove any stale sidecar from a previous run so it doesn't mislead.
            if sidecar.exists():
                sidecar.unlink()
            return

        rows = [
            {
                "channel": ch,
                "n_votes": len(methods),
                "methods": "+".join(sorted(methods)),
            }
            for ch, methods in sorted(candidates.items())
        ]
        pd.DataFrame(rows).to_csv(sidecar, sep="\t", index=False)
        self.log(
            f"Wrote {len(rows)} candidate channel(s) for manual review → {sidecar}"
        )

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_all_methods(self, raw: mne.io.BaseRaw) -> Dict[str, Set[str]]:
        """Run all enabled detectors on a single recording.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to analyse.

        Returns
        -------
        method_results : dict
            {method_name -> set of flagged channel names}.  Detectors that are
            disabled, unavailable, or error out contribute an empty set.
        """
        picks = self.cfg.ch_types[0]
        methods = self._methods()

        # report the initial number of bad channels so the user can see how many new ones are added by the detectors.
        initial_bads = set(raw.info.get("bads", []))
        self.log(f"  Initial bad channels: {len(initial_bads)} ({sorted(initial_bads)})")

        # Bandpass-filtered copy for the variance/spatial detectors (the PSD
        # detector deliberately runs on the unfiltered data, see below).
        filt = raw.copy().filter(
            l_freq=self.cfg.l_freq, h_freq=self.cfg.h_freq, method="iir"
        )

        results: Dict[str, Set[str]] = {}
        if "gesd" in methods:
            results["gesd"] = self._detect_gesd_full(filt, picks)
        if "timeresolved" in methods:
            results["timeresolved"] = self._detect_timeresolved(filt, picks)
        if "psd" in methods:
            results["psd"] = self._detect_psd(raw)
        if "lof" in methods:
            results["lof"] = self._detect_lof(filt, picks)

        del filt

        # Channels already marked bad upstream stay bad regardless (preserved in
        # save_results); exclude them so the vote tally reflects only newly
        # detected channels.
        existing = set(raw.info.get("bads", []))
        results = {m: (chans - existing) for m, chans in results.items()}

        for method, chans in results.items():
            self.log(f"  [{method}] flagged {len(chans)}: {sorted(chans)}")

        return results

    def _detect_gesd_full(self, filt: mne.io.BaseRaw, picks: str) -> Set[str]:
        """Full-recording GESD on per-channel std (OSL-ephys)."""
        try:
            detected = osl_bad_channels(
                filt.copy(),
                picks=picks,
                significance_level=self._significance(),
            )
            return set(detected.info["bads"])
        except Exception as exc:  # pragma: no cover - defensive
            self.log(f"  [gesd] skipped ({exc})")
            return set()

    def _detect_timeresolved(self, filt: mne.io.BaseRaw, picks: str) -> Set[str]:
        """Windowed GESD: flag channels that are variance outliers in many windows.

        The recording is split into non-overlapping windows.  Within each
        window we GESD the per-channel standard deviation (upper tail only —
        only abnormally *high* variance counts as bad) and tally how often each
        channel is flagged.  Channels flagged in more than
        ``_bad_channel_frac_threshold`` of windows are returned.
        """
        window_sec = float(
            getattr(self.cfg, "_bad_channel_window_sec", _DEFAULT_WINDOW_SEC)
        )
        frac_threshold = float(
            getattr(self.cfg, "_bad_channel_frac_threshold", _DEFAULT_FRAC_THRESHOLD)
        )
        alpha = self._significance()

        ch_idx = np.array(
            mne.pick_types(filt.info, meg=picks, ref_meg=False, exclude="bads")
        )
        if ch_idx.size < 3:
            self.log("  [timeresolved] too few channels; skipping")
            return set()
        names = np.array(filt.ch_names)[ch_idx]

        # Drop already-annotated bad spans so they don't dominate the metric.
        data = filt.get_data(picks=ch_idx, reject_by_annotation="omit")
        n_ch, n_times = data.shape

        win = max(1, int(round(filt.info["sfreq"] * window_sec)))
        n_windows = n_times // win
        if n_windows < 2:
            self.log(
                f"  [timeresolved] only {n_windows} full window(s); skipping"
            )
            return set()

        # Per-channel std in each full window → (n_ch, n_windows).
        trimmed = data[:, : n_windows * win].reshape(n_ch, n_windows, win)
        win_std = trimmed.std(axis=2)

        flag_counts = np.zeros(n_ch, dtype=int)
        for w in range(n_windows):
            mask, _ = osl_gesd(win_std[:, w], alpha=alpha, p_out=0.5, outlier_side=1)
            flag_counts += np.asarray(mask, dtype=int)

        frac = flag_counts / n_windows
        bad = set(names[frac > frac_threshold].tolist())
        if bad:
            for ch in sorted(bad):
                idx = int(np.where(names == ch)[0][0])
                self.log(
                    f"    {ch}: outlier in {frac[idx] * 100:.0f}% of "
                    f"{n_windows} windows"
                )
        return bad

    def _detect_psd(self, raw: mne.io.BaseRaw) -> Set[str]:
        """PSD / noise-floor detector (OSL-ephys), run on unfiltered data."""
        fmin = float(getattr(self.cfg, "_bad_channel_psd_fmin", _DEFAULT_PSD_FMIN))
        fmax = float(getattr(self.cfg, "_bad_channel_psd_fmax", _DEFAULT_PSD_FMAX))
        n_fft = int(getattr(self.cfg, "_bad_channel_psd_nfft", _DEFAULT_PSD_NFFT))
        try:
            raw_data = raw.copy().pick("data", exclude="bads")
            bads = osl_detect_bad_channels_psd(
                raw_data,
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                alpha=self._significance(),
            )
            del raw_data
            return set(bads)
        except Exception as exc:
            self.log(f"  [psd] skipped ({exc})")
            return set()

    def _detect_lof(self, filt: mne.io.BaseRaw, picks: str) -> Set[str]:
        """Local Outlier Factor detector (MNE), comparing spatial neighbours."""
        n_neighbors = int(
            getattr(self.cfg, "_bad_channel_lof_neighbors", _DEFAULT_LOF_NEIGHBORS)
        )
        threshold = float(
            getattr(self.cfg, "_bad_channel_lof_threshold", _DEFAULT_LOF_THRESHOLD)
        )
        n_good = len(
            mne.pick_types(filt.info, meg=picks, ref_meg=False, exclude="bads")
        )
        if n_good < 3:
            self.log("  [lof] too few channels; skipping")
            return set()
        # n_neighbors must be < number of channels considered.
        n_neighbors = min(n_neighbors, n_good - 1)
        try:
            bads = mne.preprocessing.find_bad_channels_lof(
                filt,
                n_neighbors=n_neighbors,
                picks=picks,
                threshold=threshold,
            )
            return set(bads)
        except Exception as exc:
            self.log(f"  [lof] skipped ({exc})")
            return set()


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = BadChannelsAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
