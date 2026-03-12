"""General sensor regression for OPM-MEG data.

This module regresses out a configurable list of sensor signals from MEG data
channels to remove noise captured by those sensors. Two regression methods are
supported:

1. **Standard regression**: Computes a single set of regression weights
   over the entire recording using MNE's EOGRegression.

2. **Time-varying regression**: Uses a sliding window approach to compute
   time-varying regression weights, better suited for non-stationary noise.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=regress --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.regress import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    bids_root : str
        Root directory of BIDS dataset.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.
    _regress_preds : list
        Channel names or channel-type strings to use as predictors
        (e.g., ['ref_meg'] or ['MEG0001', 'MEG0002']).

Optional:
    _regress : bool
        Enable/disable this analysis. Default: False.
    _regress_timevarying : bool
        Use time-varying regression instead of standard. Default: False.
    _regress_window : float
        Window size in seconds for time-varying regression. Default: 100.0.
    _regress_freqs : list of tuple
        Frequency bands for filtering predictor channels.
        Example: [(None, 5.0), (5.0, 15.0)]. Default: None (use raw + squared).
    _regress_lags : int
        Number of past time-lags for delay-embedded regression. When > 0,
        time-shifted copies of each predictor channel (lag-1 … lag-N samples)
        are added as extra regressors and ridge regression is used instead of
        EOGRegression. Default: 0 (no delay embedding).
    _regress_plot : bool
        Show PSD comparison plots before/after. Default: False.
    process_empty_room : bool
        Also process empty room noise recording. Default: False.
    find_breaks : bool
        Annotate recording breaks before regression. Default: False.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

import mne
import mne_bids
from mne._fiff.pick import _picks_to_idx
from scipy import stats
from scipy.linalg import qr

from ._base import BaseAnalysis
from ._io import write_raw_bids_preserve_events


class RegressAnalysis(BaseAnalysis):
    """Regress out a configurable set of sensor signals from MEG data.

    This analysis removes noise captured by a user-specified list of sensors
    from the primary MEG channels using linear regression. The predictor
    channels are set via ``cfg._regress_preds`` and can be channel type
    strings (e.g., ``'ref_meg'``) or individual channel names.

    The sliding window (time-varying) method is recommended for recordings
    with non-stationary noise, as it allows the regression weights to adapt
    over time.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'regress'
    ANALYSIS_NAME : str
        'regress'

    See Also
    --------
    mne.preprocessing.EOGRegression : MNE's standard regression method.
    """

    ANALYSIS_KEY = "regress"
    ANALYSIS_NAME = "regress"

    def is_enabled(self) -> bool:
        """Check if regression is enabled.

        Returns
        -------
        enabled : bool
            True if cfg._regress is True.
        """
        return getattr(self.cfg, "_regress", False)

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for regression.

        Loads the main task data and optionally empty room data if
        process_empty_room is enabled.

        Returns
        -------
        data : dict
            Dictionary with keys:
            - cfg.task : Raw data for the main task
            - 'noise' : Raw data for empty room (if process_empty_room=True)
        """
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        # Determine which tasks to load
        tasks = [self.cfg.task]
        if getattr(self.cfg, "process_empty_room", False):
            tasks.insert(0, "noise")

        for task in tasks:
            # Search for raw files (handles runs, splits, etc.)
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks=task,
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if not paths:
                raise FileNotFoundError(f"No raw data found for task={task}")

            raw = mne_bids.read_raw_bids(paths[0], extra_params={"preload": True})
            data[task] = raw
            self.log(f"Loaded raw data for task={task}")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regression on all loaded data.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data per task.

        Returns
        -------
        results : dict
            Dictionary with cleaned raw data for each task.
        """
        results: Dict[str, Any] = {"bads": []}

        for task, raw in data.items():
            is_noise = task == "noise"
            self.log(f"Processing task={task} (is_noise={is_noise})")

            cleaned = self._regress(raw, is_noise=is_noise)
            results[task] = cleaned

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save regression results back to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary with cleaned raw data per task.
        """
        self.log("Saving results...")

        # Separate task data from metadata
        tasks = {k: v for k, v in results.items() if k not in {"bads"}}

        # Find empty room path if needed
        er_bids_path = None
        if "noise" in tasks:
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks="noise",
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if paths:
                er_bids_path = paths[0]

        for task, raw in tasks.items():
            # Find existing file
            paths = mne_bids.find_matching_paths(
                root=self.cfg.bids_root,
                subjects=self.cfg.subjects,
                tasks=task,
                sessions=self.cfg.sessions,
                datatypes="meg",
                extensions=".fif",
                ignore_nosub=True,
            )
            if not paths:
                raise FileNotFoundError(f"No file found for task={task}")

            bp = paths[0]
            bp.split = None  # Clear split to write to base file

            # Associate with empty room for non-noise tasks
            write_kwargs = dict(
                raw=raw,
                bids_path=bp,
                allow_preload=True,
                overwrite=True,
                format="FIF",
            )
            if er_bids_path and task != "noise":
                write_kwargs["empty_room"] = er_bids_path
            write_raw_bids_preserve_events(**write_kwargs)
            self.log(f"Saved task={task}")

    def _regress(
        self, raw: mne.io.BaseRaw, is_noise: bool = False
    ) -> mne.io.BaseRaw:
        """Perform regression on raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw MEG data to process.
        is_noise : bool
            If True, this is empty room noise data (skip break annotation).

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Raw data with predictor signals regressed out.
        """
        self.log(f"Regressing out channels: {self.cfg._regress_preds}")

        # Check whether predictor channels exist in this recording
        try:
            _picks_to_idx(raw.info, self.cfg._regress_preds)
        except (ValueError, KeyError):
            warnings.warn(
                f"Predictor channels {self.cfg._regress_preds} not found "
                f"in noise recording — skipping regression."
            )
            return raw

        # Annotate breaks if configured (not for noise recordings)
        if getattr(self.cfg, "find_breaks", False) and not is_noise:
            mne.preprocessing.annotate_break(
                raw,
                min_break_duration=self.cfg.min_break_duration,
                t_start_after_previous=self.cfg.t_break_annot_start_after_previous_event,
                t_stop_before_next=self.cfg.t_break_annot_stop_before_next_event,
            )

        # Choose regression method
        if getattr(self.cfg, "_regress_timevarying", False):
            raw_clean = self._regress_timevarying(raw)
        else:
            raw_clean = self._regress_standard(raw)

        # Optional PSD comparison plot
        if getattr(self.cfg, "_regress_plot", False):
            self._plot_psd_comparison(raw, raw_clean)

        self.log("Regression complete!")
        return raw_clean

    def _regress_standard(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Standard (time-invariant) regression.

        Uses MNE's EOGRegression when no lags are requested, or a custom
        ridge regression with delay-embedded predictors when ``_regress_lags > 0``.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process.

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Cleaned raw data.
        """
        n_lags = getattr(self.cfg, "_regress_lags", 0)

        if n_lags <= 0:
            self.log("Using standard (time-invariant) regression")

            weights = mne.preprocessing.EOGRegression(
                picks=self.cfg.ch_types[0],
                picks_artifact=self.cfg._regress_preds,
                proj=True,
            ).fit(raw)

            raw_clean = weights.apply(raw, copy=True)
            del weights

        else:
            self.log(
                f"Using delay-embedded ridge regression ({n_lags} lags, "
                f"{n_lags / raw.info['sfreq'] * 1000:.1f} ms)"
            )
            
            # # print channel names for debugging
            # self.log(f"  Target channels: {[raw.ch_names[i] for i in _picks_to_idx(raw.info, self.cfg.ch_types[0])]}")
            # # print all channel names
            # self.log(f"  Info: {raw.info}")
            # self.log(f"  All channels: {raw.ch_names}")
            # self.log(f"  Predictor channels: {[raw.ch_names[i] for i in _picks_to_idx(raw.info, self.cfg._regress_preds)]}")
            
            raw_clean = self._regress_delay_embedded(raw, n_lags)

        return raw_clean

    def _regress_delay_embedded(
        self, raw: mne.io.BaseRaw, n_lags: int
    ) -> mne.io.BaseRaw:
        """Ridge regression with delay-embedded artifact predictors.

        Builds a design matrix from the artifact channels and their
        time-shifted (lagged) copies, then solves a ridge regression
        to remove the artifact subspace from the target channels.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process.
        n_lags : int
            Number of past lags to include (lag-1 … lag-n_lags).

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Cleaned raw data.
        """
        # --- resolve channel indices ---
        mag_idx = _picks_to_idx(raw.info, self.cfg.ch_types[0])
        pred_idx = _picks_to_idx(raw.info, self.cfg._regress_preds)

        # --- build delay-embedded design matrix ---
        pred_data = raw.get_data(picks=pred_idx)  # (n_pred, n_times)
        n_pred, n_times = pred_data.shape

        embedded = [pred_data]  # lag-0
        for lag in range(1, n_lags + 1):
            shifted = np.zeros_like(pred_data)
            shifted[:, lag:] = pred_data[:, :-lag]
            embedded.append(shifted)

        X = np.vstack(embedded)  # (n_pred * (n_lags + 1), n_times)
        del embedded, pred_data

        # mean-center each predictor row
        X -= X.mean(axis=1, keepdims=True)

        n_features = X.shape[0]
        self.log(f"  Design matrix: {n_features} features x {n_times} samples")

        # --- ridge regression ---
        cov_xx = X @ X.T  # (n_features, n_features)
        alpha = 1e-6 * np.trace(cov_xx) / n_features
        cov_xx[np.diag_indices_from(cov_xx)] += alpha

        raw_data = raw.get_data().copy()
        target = raw_data[mag_idx, :]  # (n_target, n_times)

        cov_xy = X @ target.T  # (n_features, n_target)
        beta = np.linalg.solve(cov_xx, cov_xy)  # (n_features, n_target)

        # subtract predicted artifact
        raw_data[mag_idx, :] -= beta.T @ X

        raw_clean = mne.io.RawArray(raw_data, raw.info)
        del raw_data, X, beta

        return raw_clean

    def _regress_timevarying(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Time-varying (sliding window) regression.

        Computes regression weights in overlapping windows to handle
        non-stationary noise. Uses QR decomposition for efficient
        computation with ridge regularization.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process.

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Cleaned raw data.
        """
        self.log("Using time-varying (sliding window) regression")

        # Extract data arrays
        raw_data = raw.get_data()
        info = raw.info

        # Prepare predictor data
        pred_data = self._prepare_predictor_data(raw)

        # Get channel indices and dimensions
        mag_idx = _picks_to_idx(info, self.cfg.ch_types[0])
        n_channels, n_times = raw_data.shape
        n_pred, _ = pred_data.shape
        sfreq = info["sfreq"]

        # Window parameters
        window_size = int(sfreq * getattr(self.cfg, "_regress_window", 100.0))
        step_size = int(window_size // 2)  # 50% overlap
        n_windows = int(np.ceil((n_times - window_size) / step_size))

        # Ridge regression prior (regularization)
        prior = np.diag(np.repeat([np.sqrt(1e-4)], n_pred))

        self.log(
            f"Processing {n_windows} windows "
            f"({window_size / sfreq:.1f}s window, {step_size / sfreq:.1f}s step)"
        )

        # Sliding window regression
        for w in range(n_windows):
            start = w * step_size
            end = min(start + window_size, n_times)

            # Build design matrix with ridge prior
            data_x = pred_data[:, start:end].T
            X = np.vstack([data_x, prior])

            # QR decomposition with column pivoting for numerical stability
            Q, _, _ = qr(X, pivoting=True, mode="economic")
            Qd = Q[: (end - start), :]  # Extract data portion

            # Regress out predictors from MEG channels
            raw_data[mag_idx, start:end] -= (
                raw_data[mag_idx, start:end] @ Qd
            ) @ Qd.T

            # Progress logging (every 10%)
            if w % max(n_windows // 10, 1) == 0:
                self.log(f"Processed {w}/{n_windows} windows")

        # Create cleaned Raw object with original info
        raw_clean = mne.io.RawArray(raw_data, info)
        del raw_data, pred_data

        return raw_clean

    def _prepare_predictor_data(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """Prepare predictor channel data for regression.

        If ``_regress_freqs`` is set, filters predictor channels in
        frequency bands. Otherwise, uses raw predictor data plus
        squared values to capture nonlinear relationships.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data containing the predictor channels.

        Returns
        -------
        pred_data : ndarray, shape (n_pred_features, n_times)
            Prepared predictor data, z-scored along time axis.
        """
        preds = self.cfg._regress_preds
        freq_bands = getattr(self.cfg, "_regress_freqs", None)

        if freq_bands is not None:
            # Filter predictors in multiple frequency bands
            pred_data_list = []
            for l_freq, h_freq in freq_bands:
                pred_filt = (
                    raw.copy()
                    .pick(preds)
                    .filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        fir_window="blackman",
                        h_trans_bandwidth=5.0,
                    )
                )
                pred_data_list.append(pred_filt.get_data(picks=preds))
                del pred_filt
        else:
            # Use raw predictors + squared (captures nonlinearity)
            pred_raw = raw.get_data(picks=preds)
            pred_raw -= np.mean(pred_raw, axis=1, keepdims=True)
            pred_data_list = [pred_raw, pred_raw**2]
            del pred_raw

        # Stack and z-score normalize
        pred_data = stats.zscore(np.vstack(pred_data_list), axis=1)
        del pred_data_list

        return pred_data

    def _plot_psd_comparison(
        self, raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw
    ) -> None:
        """Plot PSD comparison before and after regression.

        Shows:
        1. Interactive raw data browser for before/after
        2. PSD overlay plot comparing before/after
        3. PSD difference plot showing noise reduction

        Parameters
        ----------
        raw_before : mne.io.BaseRaw
            Original raw data.
        raw_after : mne.io.BaseRaw
            Cleaned raw data.
        """
        self.log("Plotting PSD before/after regression...")

        # Interactive data browsing
        plot_kwargs = dict(
            precompute=True,
            n_channels=64,
            show_options=True,
            show=True,
            block=True,
            lowpass=60.0,
            decim=4,
            scalings=dict(mag=1e-11, eyegaze=0.01, pupil=0.01),
        )
        raw_before.plot(**plot_kwargs)
        raw_after.plot(**plot_kwargs)

        # Compute PSDs
        picks = _picks_to_idx(raw_before.info, self.cfg.ch_types[0])

        def compute_psd(raw, picks):
            data, freqs = raw.compute_psd(
                n_fft=2048,
                fmin=0,
                fmax=100,
                tmin=raw.times[int(raw.n_times * 0.25)],
                tmax=raw.times[int(raw.n_times * 0.75)],
                proj=True,
                picks=picks,
                n_jobs=-1,
            ).get_data(return_freqs=True)
            return data, freqs

        power_to_db = lambda power: 10 * np.log10(power)

        self.log("Computing PSD for before...")
        data_before, freqs_before = compute_psd(raw_before, picks)
        self.log("Computing PSD for after...")
        data_after, freqs_after = compute_psd(raw_after, picks)
        self.log("PSD computation complete")

        mean_psd_before = np.mean(data_before, axis=0)
        mean_psd_after = np.mean(data_after, axis=0)

        # Create figure with two subplots
        plt.switch_backend("qt5agg")
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # Left: Overlaid PSDs
        ax = axes[0]
        for ch in range(data_before.shape[0]):
            ax.semilogy(freqs_before, data_before[ch], color="red", alpha=0.1, lw=1)
            ax.semilogy(freqs_after, data_after[ch], color="blue", alpha=0.1, lw=1)
        ax.semilogy(
            freqs_before, mean_psd_before, color="red", alpha=1.0, lw=2, label="Before"
        )
        ax.semilogy(
            freqs_after, mean_psd_after, color="blue", alpha=1.0, lw=2, label="After"
        )
        ax.set_title("PSD Before/After Regression")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (V²/Hz)")
        ax.legend()

        # Right: PSD difference
        ax = axes[1]
        for ch in range(data_before.shape[0]):
            ax.plot(
                freqs_before,
                power_to_db(data_after[ch]) - power_to_db(data_before[ch]),
                color="green",
                alpha=0.1,
                lw=1,
            )
        ax.plot(
            freqs_before,
            power_to_db(mean_psd_after) - power_to_db(mean_psd_before),
            color="black",
            lw=2,
        )
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("PSD Difference (After - Before)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (dB)")

        plt.tight_layout()
        plt.show()


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = RegressAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
