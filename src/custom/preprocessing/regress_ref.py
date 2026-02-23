"""Reference regression analysis for OPM-MEG data.

This module regresses out reference channel signals from MEG data channels
to remove environmental noise captured by the reference sensors. Two
regression methods are supported:

1. **Standard regression**: Computes a single set of regression weights
   over the entire recording using MNE's EOGRegression.

2. **Time-varying regression**: Uses a sliding window approach to compute
   time-varying regression weights, better suited for non-stationary
   environmental noise.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=regress_ref --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.regress_ref import run
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

Optional:
    _regress_ref : bool
        Enable/disable this analysis. Default: False.
    _regress_ref_timevarying : bool
        Use time-varying regression instead of standard. Default: False.
    _regress_ref_window : float
        Window size in seconds for time-varying regression. Default: 100.0.
    _regress_ref_freqs : list of tuple
        Frequency bands for filtering reference channels.
        Example: [(None, 5.0), (5.0, 15.0)]. Default: None (use raw + squared).
    _regress_ref_plot : bool
        Show PSD comparison plots before/after. Default: False.
    process_empty_room : bool
        Also process empty room noise recording. Default: False.
    find_breaks : bool
        Annotate recording breaks before regression. Default: False.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

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


class RegressReferenceAnalysis(BaseAnalysis):
    """Regress out reference channels from MEG data.

    This analysis removes environmental noise captured by reference sensors
    from the primary MEG channels using linear regression. Reference channels
    typically measure background magnetic fields that contaminate the brain
    signals of interest.

    The sliding window (time-varying) method is recommended for recordings
    with non-stationary noise, as it allows the regression weights to adapt
    over time.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'regressref'
    ANALYSIS_NAME : str
        'regress_ref'

    See Also
    --------
    mne.preprocessing.EOGRegression : MNE's standard regression method.
    """

    ANALYSIS_KEY = "regressref"
    ANALYSIS_NAME = "regress_ref"

    def is_enabled(self) -> bool:
        """Check if reference regression is enabled.

        Returns
        -------
        enabled : bool
            True if cfg._regress_ref is True.
        """
        return getattr(self.cfg, "_regress_ref", False)

    def load_data(self) -> Dict[str, Any]:
        """Load raw data for reference regression.

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
        """Execute reference regression on all loaded data.

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

            cleaned = self._regress_reference(raw, is_noise=is_noise)
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

    def _regress_reference(
        self, raw: mne.io.BaseRaw, is_noise: bool = False
    ) -> mne.io.BaseRaw:
        """Perform reference regression on raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw MEG data to process.
        is_noise : bool
            If True, this is empty room noise data (skip break annotation).

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Raw data with reference signals regressed out.
        """
        self.log("Regressing out reference channels...")

        # Annotate breaks if configured (not for noise recordings)
        if getattr(self.cfg, "find_breaks", False) and not is_noise:
            mne.preprocessing.annotate_break(
                raw,
                min_break_duration=self.cfg.min_break_duration,
                t_start_after_previous=self.cfg.t_break_annot_start_after_previous_event,
                t_stop_before_next=self.cfg.t_break_annot_stop_before_next_event,
            )

        # Choose regression method
        if getattr(self.cfg, "_regress_ref_timevarying", False):
            raw_clean = self._regress_timevarying(raw)
        else:
            raw_clean = self._regress_standard(raw)

        # Optional PSD comparison plot
        if getattr(self.cfg, "_regress_ref_plot", False):
            self._plot_psd_comparison(raw, raw_clean)

        self.log("Reference regression complete!")
        return raw_clean

    def _regress_standard(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Standard (time-invariant) reference regression.

        Uses MNE's EOGRegression to compute a single set of regression
        weights over the entire recording.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to process.

        Returns
        -------
        raw_clean : mne.io.BaseRaw
            Cleaned raw data.
        """
        self.log("Using standard (time-invariant) regression")

        weights = mne.preprocessing.EOGRegression(
            picks=self.cfg.ch_types[0],
            picks_artifact="ref_meg",
            proj=True,
        ).fit(raw)

        raw_clean = weights.apply(raw, copy=True)
        del weights

        return raw_clean

    def _regress_timevarying(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Time-varying (sliding window) reference regression.

        Computes regression weights in overlapping windows to handle
        non-stationary environmental noise. Uses QR decomposition for
        efficient computation with ridge regularization.

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

        # Prepare reference data
        ref_data = self._prepare_reference_data(raw)

        # Get channel indices and dimensions
        mag_idx = _picks_to_idx(info, self.cfg.ch_types[0])
        n_channels, n_times = raw_data.shape
        n_ref, _ = ref_data.shape
        sfreq = info["sfreq"]

        # Window parameters
        window_size = int(sfreq * getattr(self.cfg, "_regress_ref_window", 100.0))
        step_size = int(window_size // 2)  # 50% overlap
        n_windows = int(np.ceil((n_times - window_size) / step_size))

        # Ridge regression prior (regularization)
        prior = np.diag(np.repeat([np.sqrt(1e-4)], n_ref))

        self.log(
            f"Processing {n_windows} windows "
            f"({window_size / sfreq:.1f}s window, {step_size / sfreq:.1f}s step)"
        )

        # Sliding window regression
        for w in range(n_windows):
            start = w * step_size
            end = min(start + window_size, n_times)

            # Build design matrix with ridge prior
            data_x = ref_data[:, start:end].T
            X = np.vstack([data_x, prior])

            # QR decomposition with column pivoting for numerical stability
            Q, _, _ = qr(X, pivoting=True, mode="economic")
            Qd = Q[: (end - start), :]  # Extract data portion

            # Regress out reference from MEG channels
            raw_data[mag_idx, start:end] -= (
                raw_data[mag_idx, start:end] @ Qd
            ) @ Qd.T

            # Progress logging (every 10%)
            if w % max(n_windows // 10, 1) == 0:
                self.log(f"Processed {w}/{n_windows} windows")

        # Create cleaned Raw object with original info
        raw_clean = mne.io.RawArray(raw_data, info)
        del raw_data, ref_data

        return raw_clean

    def _prepare_reference_data(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """Prepare reference channel data for regression.

        If _regress_ref_freqs is set, filters reference channels in
        frequency bands. Otherwise, uses raw reference data plus
        squared values to capture nonlinear relationships.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data containing reference channels.

        Returns
        -------
        ref_data : ndarray, shape (n_ref_features, n_times)
            Prepared reference data, z-scored along time axis.
        """
        freq_bands = getattr(self.cfg, "_regress_ref_freqs", None)

        if freq_bands is not None:
            # Filter reference in multiple frequency bands
            ref_data_list = []
            for l_freq, h_freq in freq_bands:
                ref_filt = (
                    raw.copy()
                    .pick("ref_meg")
                    .filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        fir_window="blackman",
                        h_trans_bandwidth=5.0,
                    )
                )
                ref_data_list.append(ref_filt.get_data(picks="ref_meg"))
                del ref_filt
        else:
            # Use raw reference + squared (captures nonlinearity)
            ref_raw = raw.get_data(picks="ref_meg")
            ref_raw -= np.mean(ref_raw, axis=1, keepdims=True)
            ref_data_list = [ref_raw, ref_raw**2]
            del ref_raw

        # Stack and z-score normalize
        ref_data = stats.zscore(np.vstack(ref_data_list), axis=1)
        del ref_data_list

        return ref_data

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
    analysis = RegressReferenceAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
