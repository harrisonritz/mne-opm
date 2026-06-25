"""Bad channel detection for OPM-MEG data.

This module detects bad channels by computing several complementary *per-channel
metrics*, combining them with PCA, and flagging outliers with a Šidák-corrected
generalized ESD (GESD) test — the same unified PCA→GESD procedure used for ICA
component selection (see :mod:`custom.preprocessing.pca_gesd`).

Available per-channel metrics (each selectable via ``cfg._channel_metrics``):

* **log_std** — log of the per-channel standard deviation over the recording
  (broadband power / variance).  Catches channels that are bad for the whole
  session.  ``side=+1`` (side=0 for two-tailed).
* **logit_outlier_frac** — logit of the *fraction of short time windows* in which
  a channel is an upper-tail variance outlier (a windowed GESD).  Catches
  **intermittent** bad channels with a near-normal whole-recording std.
  ``side=+1``.
* **kurtosis** — signed square root of the per-channel temporal kurtosis;
  transient/spiky channels have heavy-tailed amplitude distributions.
  ``side=+1``.
* **lof** — log of the Local Outlier Factor (MNE), flagging channels anomalous
  relative to their spatial neighbours.  ``side=+1``.
* **psd** — per-channel mean log10 power over ``[psd_fmin, psd_fmax]``;
  two-tailed (``side=0``) so both dead/low-power and noisy/high-power channels
  are caught.

Each selected metric becomes one row of a (metrics × channels) matrix that is
z-scored, projected onto principal components, and GESD-tested per eigenscore
under a single family-wise error rate.  Channels flagged on any component are
marked ``status="bad"`` in ``channels.tsv`` and added to ``raw.info['bads']``.

Detection runs on bandpass-filtered data (using the global ``l_freq``/``h_freq``)
for the variance/kurtosis/spatial metrics; the PSD metric runs on the unfiltered
data so it sees the full spectrum / noise floor.

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
    l_freq, h_freq : float
        Bandpass filter band for detection.
    bids_root, subjects, sessions, task : see other steps.

Optional (with conservative defaults):
    process_empty_room : bool
        Also process the empty-room noise recording. Default: False.
    _channel_metrics : list of str | None
        Which per-channel metrics to feed into the PCA→GESD. Valid names are in
        ``AVAILABLE_CHANNEL_METRICS``. ``None`` (default) uses all of them; an
        empty list disables detection.
    _bad_channel_significance_level : float
        Family-wise GESD significance level (also the per-window alpha for the
        outlier-fraction metric). Default: 0.05.
    _bad_channel_window_sec : float
        Window length (seconds) for the outlier-fraction metric. Default: 2.0.
    _bad_channel_psd_fmin, _bad_channel_psd_fmax : float
        Frequency band (Hz) for the PSD metric. Defaults: 1.0, 100.0.
    _bad_channel_psd_nfft : int
        FFT length for the PSD metric. Default: 2000.
    _bad_channel_lof_neighbors : int
        Number of neighbours for the LOF metric. Default: 20.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

from osl_ephys.preprocessing.osl_wrappers import gesd as osl_gesd

import mne_bids

from ._base import BaseAnalysis
from ._io import (
    find_custom_input_paths,
    read_raw_bids_with_retry,
    write_raw_bids_custom_step,
)
from .pca_gesd import (
    MetricSpec,
    log_transform,
    logit,
    run_pca_gesd,
    save_pca_gesd_figures,
    signed_sqrt,
)


# Default parameters.
_DEFAULT_SIGNIFICANCE: float = 0.05
_DEFAULT_WINDOW_SEC: float = 2.0
_DEFAULT_PSD_FMIN: float = 1.0
_DEFAULT_PSD_FMAX: float = 100.0
_DEFAULT_PSD_NFFT: int = 2048
_DEFAULT_LOF_NEIGHBORS: int = 20

# Fraction of outliers cap passed to GESD (channels are rarely >50% bad).
_GESD_P_OUT: float = 0.5

# Suffix used for the (legacy) per-recording candidates sidecar. The PCA→GESD
# procedure no longer writes candidates, but the path helper is retained so the
# manual_channel step can look one up (and gracefully find none).
_CANDIDATES_SUFFIX = "_badchannel-candidates.tsv"


def candidates_sidecar_path(bids_path: mne_bids.BIDSPath) -> Path:
    """Return the candidates-TSV path that pairs with a given BIDSPath.

    Retained for compatibility with the ``manual_channel`` step, which looks for
    such a sidecar.  The PCA→GESD bad-channel detector does not write one.

    Parameters
    ----------
    bids_path : mne_bids.BIDSPath
        Path of the data file the candidates would relate to.

    Returns
    -------
    pathlib.Path
        Path to the ``*_badchannel-candidates.tsv`` sidecar.
    """
    return Path(bids_path.directory) / f"{bids_path.basename}{_CANDIDATES_SUFFIX}"


class BadChannelsAnalysis(BaseAnalysis):
    """Detect bad channels via PCA-whitened, Šidák-corrected GESD.

    Computes per-channel metrics (variance, intermittent-outlier fraction,
    kurtosis, LOF, PSD power), combines them with PCA, and flags outlier
    channels with a GESD test whose family-wise error rate is controlled by a
    Šidák correction across the retained principal components.  Flagged channels
    are marked bad in BIDS.

    When processing multiple tasks (e.g. main task + noise), channels are scored
    per recording and the union of flagged channels is marked in all recordings.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'badchannels'
    ANALYSIS_NAME : str
        'bad_channels'

    See Also
    --------
    custom.preprocessing.pca_gesd.run_pca_gesd : the shared PCA→GESD procedure.
    mne.preprocessing.find_bad_channels_lof : Local Outlier Factor scores.
    """

    ANALYSIS_KEY = "badchannels"
    ANALYSIS_NAME = "bad_channels"

    # All available per-channel metric names (used for validation / defaults).
    AVAILABLE_CHANNEL_METRICS = [
        "log_std",
        "logit_outlier_frac",
        "kurtosis",
        "lof",
        "psd",
        "psd_low",
        "psd_med",
        "psd_high",
    ]

    def is_enabled(self) -> bool:
        """Always enabled (no config flag required for this analysis)."""
        return True

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _significance(self) -> float:
        """Return the GESD significance level."""
        return float(
            getattr(self.cfg, "_bad_channel_significance_level", _DEFAULT_SIGNIFICANCE)
        )

    def _channel_metrics(self) -> set:
        """Resolve which per-channel metrics to use.

        Reads ``cfg._channel_metrics``; ``None`` selects all available metrics,
        an empty list disables detection, and unknown names raise ``ValueError``.
        """
        selected = getattr(self.cfg, "_channel_metrics", None)
        if selected is None:
            return set(self.AVAILABLE_CHANNEL_METRICS)
        unknown = set(selected) - set(self.AVAILABLE_CHANNEL_METRICS)
        if unknown:
            raise ValueError(
                f"Unknown _channel_metrics names: {unknown}. "
                f"Available: {self.AVAILABLE_CHANNEL_METRICS}"
            )
        return set(selected)

    # ------------------------------------------------------------------
    # BaseAnalysis interface
    # ------------------------------------------------------------------

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for bad channel detection (one entry per task)."""
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        tasks = [self.cfg.task]
        if getattr(self.cfg, "process_empty_room", False) and getattr(self.cfg, "_bad_channel_emptyroom", False):
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
        """Score each task with PCA→GESD and union the flagged channels.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Raw data per task plus ``bads`` : sorted list of flagged channels.
        """
        metrics = sorted(self._channel_metrics())
        alpha = self._significance()
        self.log(f"Channel metrics={metrics} | alpha={alpha}")

        results: Dict[str, Any] = {}
        confirmed: set = set()
        # Stored per-task (ch_names, PCAGesdResult) for figure generation.
        self._metric_results: Dict[str, Tuple[List[str], Any]] = {}

        for task, raw in data.items():
            self.log(f"\nProcessing task={task}")
            ch_names, specs = self._compute_channel_metrics(raw)
            if not specs:
                self.log(f"  no channel metrics computed; skipping task={task}")
                results[task] = raw
                continue

            result = run_pca_gesd(
                specs, alpha=alpha, p_out=_GESD_P_OUT, log=self.log
            )
            flagged = [
                ch_names[i] for i in np.where(result.flagged)[0].tolist()
            ]
            self.log(f"  flagged {len(flagged)}: {sorted(flagged)}")

            confirmed.update(flagged)
            self._metric_results[task] = (ch_names, result)
            results[task] = raw

        results["bads"] = sorted(confirmed)
        self.log(
            f"Flagged bad channels: {len(results['bads'])} ({results['bads']})"
        )
        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Mark flagged channels bad in BIDS and save the diagnostic figures."""
        self.log("Saving results...")

        confirmed = sorted(set(results.get("bads", []) or []))
        tasks = {k: v for k, v in results.items() if k != "bads"}

        # Process noise first so the task save can reference it as empty-room.
        ordered_tasks = sorted(tasks.items(), key=lambda kv: kv[0] != "noise")

        er_output_bp = None
        for task, raw in ordered_tasks:
            paths = find_custom_input_paths(self.cfg, task=task)
            if not paths:
                raise FileNotFoundError(f"No file found for task={task}")
            source_bp = paths[0]

            if confirmed:
                existing_bads = raw.info.get("bads", [])
                merged_bads = sorted(set(existing_bads) | set(confirmed))
                raw.info["bads"] = merged_bads
                self.log(f"task={task}: {len(merged_bads)} total bad channels")

            empty_room = er_output_bp if task != "noise" else None
            output_bp = write_raw_bids_custom_step(
                raw, self.cfg, source_bp, empty_room=empty_room
            )

            if confirmed:
                mne_bids.mark_channels(
                    bids_path=output_bp,
                    ch_names=confirmed,
                    status="bad",
                    descriptions=["auto:pca-gesd"] * len(confirmed),
                )

            self._save_channel_figures(task, output_bp)

            if task == "noise":
                er_output_bp = output_bp

            self.log(f"Saved task={task} → {output_bp.fpath}")

    # ------------------------------------------------------------------
    # Per-channel metric computation
    # ------------------------------------------------------------------

    def _compute_channel_metrics(
        self, raw: mne.io.BaseRaw
    ) -> Tuple[List[str], List[MetricSpec]]:
        """Compute the selected per-channel metrics for one recording.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to analyse.

        Returns
        -------
        ch_names : list of str
            Names of the (good) channels scored, in metric-array order.
        specs : list of MetricSpec
            One metric per selected, transformed entry.
        """
        picks = self.cfg.ch_types[0]
        selected = self._channel_metrics()

        ch_idx = np.array(
            mne.pick_types(raw.info, meg=picks, ref_meg=False, eyetrack=False, exclude=[""])
        )
        ch_names = [raw.ch_names[i] for i in ch_idx]
        if not selected:
            self.log("  no channel metrics selected; nothing to compute")
            return ch_names, []
        if ch_idx.size < 3:
            self.log("  too few channels; skipping channel metrics")
            return ch_names, []
        
        self.log(f"initial bad channels ({len(raw.info["bads"])}): {raw.info["bads"]}")
        self.log(f"number of channels to test: {ch_idx.size}")


        # Bandpass-filtered copy for variance/kurtosis/spatial metrics.
        filt = raw.copy().filter(
            l_freq=self.cfg.l_freq, h_freq=self.cfg.h_freq, method="iir"
        )
        data = filt.get_data(picks=ch_idx, reject_by_annotation="omit")

        specs: List[MetricSpec] = []

        if "log_std" in selected:
            std = data.std(axis=1)
            # Two-tailed: dead (low) and noisy (high) channels both flagged.
            specs.append(MetricSpec("log_std", log_transform(std), 0))

        if "logit_outlier_frac" in selected:
            frac = self._outlier_fraction(data, float(filt.info["sfreq"]))
            specs.append(MetricSpec("logit_outlier_frac", logit(frac), 1))

        if "kurtosis" in selected:
            kurt = scipy_kurtosis(data, axis=1, fisher=True)
            specs.append(MetricSpec("kurtosis", signed_sqrt(kurt), 1))

        if "lof" in selected:
            lof = self._lof_scores(filt, ch_idx)
            if lof is not None:
                specs.append(MetricSpec("lof", log_transform(lof), 1))

        if "psd" in selected or "psd_low" in selected or "psd_med" in selected or "psd_high" in selected:
            (psd, psd_low, psd_med, psd_high) = self._psd_logpower(raw, ch_idx)
            # Two-tailed: dead (low) and noisy (high) channels both flagged.
            if psd is not None and "psd" in selected:
                specs.append(MetricSpec("psd", psd, 0))
            if psd_low is not None and "psd_low" in selected:
                specs.append(MetricSpec("psd_low", psd_low, 0))
            if psd_med is not None and "psd_med" in selected:
                specs.append(MetricSpec("psd_med", psd_med, 0))
            if psd_high is not None and "psd_high" in selected:
                specs.append(MetricSpec("psd_high", psd_high, 0))

        del filt
        for s in specs:
            self.log(
                f"  metric {s.name}: mean={np.nanmean(s.values):.3f}, "
                f"std={np.nanstd(s.values):.3f}"
            )
        return ch_names, specs

    def _outlier_fraction(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Fraction of windows in which each channel is a variance outlier.

        The recording is split into non-overlapping windows; within each window
        a one-sided GESD on the per-channel std flags abnormally high-variance
        channels.  Returns the per-channel flag fraction in [0, 1].
        """
        window_sec = float(
            getattr(self.cfg, "_bad_channel_window_sec", _DEFAULT_WINDOW_SEC)
        )
        alpha = self._significance()
        n_ch, n_times = data.shape

        win = max(1, int(round(sfreq * window_sec)))
        n_windows = n_times // win
        if n_windows < 2:
            self.log(
                f"  [outlier_frac] only {n_windows} full window(s); returning zeros"
            )
            return np.zeros(n_ch)

        trimmed = data[:, : n_windows * win].reshape(n_ch, n_windows, win)
        win_std = trimmed.std(axis=2)

        counts = np.zeros(n_ch, dtype=float)
        for w in range(n_windows):
            mask, _ = osl_gesd(win_std[:, w], alpha=alpha, p_out=0.5, outlier_side=1)
            counts += np.asarray(mask, dtype=float)
        return counts / n_windows

    def _lof_scores(
        self, filt: mne.io.BaseRaw, ch_idx: np.ndarray
    ) -> "np.ndarray | None":
        """Per-channel Local Outlier Factor (higher = more outlying).

        MNE returns *negative* outlier factors (≈ -1 for inliers, more negative
        for outliers); we negate so high = bad.  Returns ``None`` on failure.
        """
        n_neighbors = int(
            getattr(self.cfg, "_bad_channel_lof_neighbors", _DEFAULT_LOF_NEIGHBORS)
        )
        n_good = int(ch_idx.size)
        if n_good < 3:
            return None
        n_neighbors = min(n_neighbors, n_good - 1)
        try:
            _, scores = mne.preprocessing.find_bad_channels_lof(
                filt,
                n_neighbors=n_neighbors,
                picks=ch_idx,
                return_scores=True,
            )
        except Exception as exc:
            self.log(f"  [lof] skipped ({exc})")
            return None
        scores = np.asarray(scores, dtype=float)
        if scores.shape[0] != n_good:
            self.log(
                f"  [lof] score length {scores.shape[0]} != {n_good}; skipping"
            )
            return None
        # Negate so larger = more outlying; shift to stay positive for the log.
        lof = -scores
        return lof - lof.min() + 1.0

    def _psd_logpower(
        self, raw: mne.io.BaseRaw, ch_idx: np.ndarray
    ) -> "np.ndarray | None":
        """Per-channel mean log10 PSD power over the configured band."""
        fmin = float(getattr(self.cfg, "_bad_channel_psd_fmin", _DEFAULT_PSD_FMIN))
        fmax = float(getattr(self.cfg, "_bad_channel_psd_fmax", _DEFAULT_PSD_FMAX))
        n_fft = int(getattr(self.cfg, "_bad_channel_psd_nfft", _DEFAULT_PSD_NFFT))
        try:
            psd = raw.compute_psd(
                picks=ch_idx,
                exclude=[""],
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                reject_by_annotation=True,
                verbose=False,
            )
            pow_all = psd.get_data(picks="all", exclude=[""])  # (n_ch, n_freqs)
            pow_low = psd.get_data(picks="all", exclude=[""], fmin=0, fmax=10)  # (n_ch, low freqs)
            pow_med = psd.get_data(picks="all", exclude=[""], fmin=10, fmax=20)  # (n_ch, med freqs)
            pow_high = psd.get_data(picks="all", exclude=[""], fmin=20, fmax=fmax)  # (n_ch, high freqs)
        except Exception as exc:
            self.log(f"  [psd] skipped ({exc})")
            return None
        return np.log10(pow_all + 1e-32).mean(axis=1), np.log10(pow_low + 1e-32).mean(axis=1), np.log10(pow_med + 1e-32).mean(axis=1), np.log10(pow_high + 1e-32).mean(axis=1),
        

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _save_channel_figures(
        self, task: str, output_bp: mne_bids.BIDSPath
    ) -> None:
        """Save the PCA→GESD diagnostic figures for one recording."""
        res = getattr(self, "_metric_results", {}).get(task)
        if res is None:
            return
        ch_names, result = res
        if result is None or result.n_pcs == 0:
            return
        try:
            out_dir = Path(output_bp.directory) / "badchannels"
            basename = output_bp.basename
        except (TypeError, ValueError):  # e.g. non-filesystem BIDSPath in tests
            return
        save_pca_gesd_figures(
            result,
            out_dir,
            basename,
            item_label="channel",
            item_names=ch_names,
            log=self.log,
        )


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
