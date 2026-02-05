"""Automatic ICA component labeling for OPM-MEG data.

This module automatically identifies and labels bad ICA components using
multiple detection strategies:

1. **Reference sensor correlation**: Identifies components that correlate
   with reference sensor ICA (environmental noise).

2. **GESD outlier detection**: Uses statistical tests on component
   properties (kurtosis, variance) to identify artifact components.

These automatic methods reduce the burden of manual ICA component
selection while catching common artifact patterns.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=auto_ica --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.auto_ica import run
    >>> run(cfg)

Configuration Attributes
------------------------
Required:
    ch_types : list
        Channel types to process (e.g., ['mag']).
    deriv_root : str
        Root directory of derivatives.
    subjects : list
        Subject IDs to process.
    sessions : list
        Session IDs to process.
    task : str
        Task name.
    spatial_filter : str
        Must be 'ica' to enable ICA analyses.

Optional:
    _auto_ica : bool
        Enable/disable automatic ICA labeling. Default: False.
    ref_bads : bool
        Use reference sensor ICA correlation. Default: True.
    gesd_bads : bool
        Use GESD outlier detection. Default: True.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
from scipy import stats
from scipy.stats import kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

import mne
import mne_bids
from mne_bids import BIDSPath, find_matching_paths
from osl_ephys.preprocessing.osl_wrappers import gesd as osl_gesd

from ._base import BaseAnalysis
from ._io import save_ica_bids


class AutoICAAnalysis(BaseAnalysis):
    """Automatic ICA component labeling.

    Uses multiple strategies to identify artifact components:
    - Reference sensor correlation (if ref_bads=True)
    - GESD outlier detection on kurtosis/variance (if gesd_bads=True)

    Components identified by any method are added to ica.exclude
    and will be removed when ICA is applied to the data.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'autoica'
    ANALYSIS_NAME : str
        'auto_ica'

    See Also
    --------
    mne.preprocessing.ICA : MNE's ICA class.
    osl_ephys.preprocessing.osl_wrappers.gesd : GESD outlier detection.
    """

    ANALYSIS_KEY = "autoica"
    ANALYSIS_NAME = "auto_ica"

    def is_enabled(self) -> bool:
        """Check if automatic ICA is enabled.

        Requires both _auto_ica=True and spatial_filter='ica'.

        Returns
        -------
        enabled : bool
            True if both conditions are met.
        """
        auto_enabled = getattr(self.cfg, "_auto_ica", False)
        ica_enabled = getattr(self.cfg, "spatial_filter", None) == "ica"
        return auto_enabled and ica_enabled

    def load_data(self) -> Dict[str, Any]:
        """Load raw data and ICA solution.

        Returns
        -------
        data : dict
            Dictionary with:
            - cfg.task : Raw data
            - 'ica' : ICA solution
        """
        self.log("Loading data...")

        # Construct BIDSPath for cleaned raw data
        subject = self.cfg.subjects[0] if isinstance(self.cfg.subjects, list) else self.cfg.subjects
        session = self.cfg.sessions[0] if isinstance(self.cfg.sessions, list) else self.cfg.sessions

        # Find cleaned raw files using mne_bids
        matching_files = find_matching_paths(
            root=self.cfg.deriv_root,
            subjects=subject,
            sessions=session,
            tasks=self.cfg.task,
            processings="clean",
            suffixes="raw",
            extensions=".fif",
            datatypes="meg",
            check=False,  # Allow non-standard suffix 'raw' for derivatives
        )

        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No cleaned raw files found for sub-{subject}/ses-{session}/task-{self.cfg.task}"
            )
        elif len(matching_files) > 1:
            runs_found = [f.run for f in matching_files]
            raise ValueError(
                f"Multiple runs found: {runs_found}. "
                "Please specify which run to use in the configuration."
            )

        # Use the matched file's run
        run = matching_files[0].run
        self.log(f"Detected run: {run}")

        bp_raw = matching_files[0]
        raw = mne.io.read_raw_fif(bp_raw.fpath, preload=True)
        self.log("Loaded cleaned raw data")

        # Load ICA solution (note: ICA files don't include run in filename)
        bp_ica = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="ica",
            processing="ica",
            extension=".fif",
            check=False,  # Allow non-standard suffix 'ica'
        )
        ica = mne.preprocessing.read_ica(bp_ica.fpath)
        self.log(f"Loaded ICA solution with {ica.n_components_} components")

        return {self.cfg.task: raw, "ica": ica}

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run automatic ICA component labeling.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw data and ICA.

        Returns
        -------
        results : dict
            Dictionary with labeled ICA and raw data.
        """
        raw = data[self.cfg.task]
        ica = data["ica"]

        self.log("Running automatic ICA component labeling...")

        # Apply labeling methods
        ica = self._auto_ica(ica, raw)

        return {self.cfg.task: raw, "ica": ica}

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save ICA solution with updated exclusions.

        Parameters
        ----------
        results : dict
            Dictionary with labeled ICA.
        """
        self.log("Saving ICA results...")

        ica = results["ica"]
        save_ica_bids(ica, self.cfg)

        self.log(f"Saved ICA with {len(ica.exclude)} excluded components")

    def _auto_ica(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Apply automatic ICA component labeling.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution to label.
        raw : mne.io.BaseRaw
            Raw data for computing component sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with updated exclude list.
        """
        # Reference sensor correlation method
        if getattr(self.cfg, "ref_bads", True):
            ica = self._label_by_reference(ica, raw)

        # GESD outlier detection method (using new PCA-whitened approach)
        if getattr(self.cfg, "gesd_bads", True):
            ica = self._label_by_gesd_new(ica, raw)

        # Remove duplicates from exclude list
        ica.exclude = sorted(set(ica.exclude))

        self.log(f"Total excluded components: {len(ica.exclude)}")
        self.log(f"Excluded: {ica.exclude}")

        return ica

    def _label_by_reference(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Identify bad components by correlation with reference sensor ICA.

        Fits a separate ICA on reference sensors and identifies main ICA
        components that correlate with reference ICA sources.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            Main ICA solution.
        raw : mne.io.BaseRaw
            Raw data containing reference channels.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with reference-correlated components added to exclude.
        """
        self.log("Identifying bad components from reference sensors...")

        # Fit ICA on reference sensors
        ref_raw = raw.copy().pick("ref_meg").filter(l_freq=1, h_freq=None)

        ref_ica = mne.preprocessing.ICA(
            n_components=0.99,
            method="picard",
            max_iter=256,
            allow_ref_meg=True,
        )
        ref_ica.fit(ref_raw, decim=2, reject_by_annotation=True)

        # Get reference ICA sources and add to raw
        ref_src = ref_ica.get_sources(ref_raw)
        ref_src.rename_channels(lambda x: f"REF_{x}")
        raw.add_channels([ref_src], force_update_info=True)

        # Find main ICA components correlated with reference
        ref_idx, _ = ica.find_bads_ref(inst=raw, method="separate")

        self.log(f"Found {len(ref_idx)} reference-correlated components: {ref_idx}")

        ica.exclude.extend(ref_idx)

        # Cleanup
        del ref_raw, ref_ica, ref_src

        return ica

    def _label_by_gesd_old(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Identify bad components using GESD outlier detection (original method).

        Uses multiple metrics (kurtosis, variance) to identify components
        with unusual statistical properties.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Raw data for computing sources.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with outlier components added to exclude.
        """
        self.log("Identifying bad components using GESD...")

        # Get ICA sources
        sources = ica.get_sources(raw).get_data()
        n_comps = sources.shape[0]

        # Check if enough components remain for GESD
        n_remaining = n_comps - len(ica.exclude)
        if n_remaining < 5:
            self.log(
                f"Too few components remaining ({n_remaining}) for GESD; skipping"
            )
            return ica

        # Compute statistics for each component
        kurtosis_scores = stats.kurtosis(sources, axis=1)
        std_scores = np.std(sources, axis=1, ddof=1)
        std_diff_scores = np.linalg.norm(np.diff(sources, axis=1), axis=1)

        # Apply GESD to each metric
        self.log(f"Before GESD: {len(ica.exclude)} excluded components")

        metrics = [
            (kurtosis_scores, "kurtosis"),
            (std_scores, "std"),
            (std_diff_scores, "std_diff"),
        ]

        for scores, name in metrics:
            gesd_mask, _ = osl_gesd(scores, p_out=1.0)

            if gesd_mask.sum() == 0:
                self.log(f"{name}: no outliers found")
            else:
                outlier_idx = np.where(gesd_mask)[0].tolist()
                ica.exclude.extend(outlier_idx)
                self.log(f"{name}: found {len(outlier_idx)} outliers: {outlier_idx}")

        self.log(f"After GESD: {len(ica.exclude)} excluded components")

        return ica

    def _label_by_gesd_new(
        self,
        ica: mne.preprocessing.ICA,
        raw: mne.io.BaseRaw,
        alpha: float = 0.05,
        p_out: float = 1.0,
        n_pcs: int | None = None,
    ) -> mne.preprocessing.ICA:
        """Identify bad components using PCA-whitened GESD (new method).

        This method applies PCA to orthogonalize multiple diagnostic metrics,
        then runs GESD on each principal component with Šidák correction for
        family-wise error control.

        Steps:
            1. Compute metrics matrix (k metrics × n components)
            2. Standardize each metric
            3. PCA to get orthogonal composite metrics
            4. GESD on each PC with Šidák correction
            5. Union of flagged components

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Raw data for computing sources.
        alpha : float
            Overall significance level for GESD (default 0.05).
        p_out : float
            Maximum fraction of outliers to detect per PC (default 1.0).
        n_pcs : int | None
            Number of principal components to use. If None, uses all (k).

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with outlier components added to exclude.
        """
        self.log("Identifying bad components using PCA-whitened GESD...")

        # Check if enough components for analysis
        n_comps = ica.n_components_
        n_remaining = n_comps - len(ica.exclude)
        if n_remaining < 5:
            self.log(
                f"Too few components remaining ({n_remaining}) for PCA-GESD; skipping"
            )
            return ica

        self.log(f"Before PCA-GESD: {len(ica.exclude)} excluded components")

        # Compute diagnostic metrics
        diagnostics = self._ica_component_diagnostics(ica, raw)
        metrics_list = self._prepare_metrics_for_gesd(diagnostics)

        # Print raw metric statistics
        self.log("=== Diagnostic Metrics Summary ===")
        for name, vals, side in metrics_list:
            direction = "high=bad" if side == 1 else "low=bad"
            self.log(
                f"  {name}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
                f"min={np.min(vals):.3f}, max={np.max(vals):.3f} ({direction})"
            )

        # Build metrics matrix: k × n
        M = np.vstack([vals for name, vals, side in metrics_list])
        k, n = M.shape
        self.log(f"Metrics matrix shape: {k} metrics × {n} components")

        # Store outlier directions for later interpretation
        outlier_sides = np.array([side for name, vals, side in metrics_list])

        # Standardize each metric (row-wise)
        M_std = StandardScaler().fit_transform(M.T).T  # shape still k × n

        # PCA on metrics
        if n_pcs is None:
            n_pcs = k

        # Compute PCA via SVD of standardized matrix
        U, s, Vt = np.linalg.svd(M_std, full_matrices=False)

        # PC loadings (how each PC weights the original metrics)
        loadings = U[:, :n_pcs]  # k × n_pcs

        # PC scores for each component
        scores = loadings.T @ M_std  # n_pcs × n

        # Variance explained (for diagnostics)
        var_explained = (s**2) / (s**2).sum()
        self.log("=== PCA Variance Explained ===")
        for p in range(n_pcs):
            self.log(f"  PC{p + 1}: {var_explained[p] * 100:.1f}%")
        self.log(f"  Total ({n_pcs} PCs): {var_explained[:n_pcs].sum() * 100:.1f}%")

        # Print PC loadings
        metric_names = [m[0] for m in metrics_list]
        self.log("=== PC Loadings (metric weights) ===")
        for p in range(n_pcs):
            loading_strs = [
                f"{metric_names[i]}={loadings[i, p]:.2f}" for i in range(k)
            ]
            self.log(f"  PC{p + 1}: {', '.join(loading_strs)}")

        # Determine outlier direction for each PC based on loadings
        # If a PC loads positively on metrics where high=bad, then high PC score=bad
        pc_outlier_sides = np.sign(loadings.T @ outlier_sides)
        # pc_outlier_sides[pc_outlier_sides == 0] = 1  # default to upper tail

        # Šidák correction (exact under independence)
        alpha_per_pc = 1 - (1 - alpha) ** (1 / n_pcs)
        self.log(
            f"Šidák-corrected alpha: {alpha:.3f} → {alpha_per_pc:.4f} per PC"
        )

        # GESD on each PC
        flagged = np.zeros(n, dtype=bool)
        self.log("=== GESD Results per PC ===")

        for p in range(n_pcs):
            side = int(pc_outlier_sides[p])
            if side == 1:
                side_str = "upper" 
            elif side == -1:
                side_str = "lower"
            else:
                side_str = "both"

            # osl_gesd returns (outlier_mask, cleaned_data)
            flags, _ = osl_gesd(
                scores[p, :], alpha=alpha_per_pc, p_out=p_out, outlier_side=side
            )
            flagged |= flags

            n_flagged = flags.sum()
            if n_flagged > 0:
                flagged_idx = np.where(flags)[0].tolist()
                self.log(
                    f"  PC{p + 1} ({side_str} tail): {n_flagged} outliers → {flagged_idx}"
                )
            else:
                self.log(f"  PC{p + 1} ({side_str} tail): no outliers")

        # Get final list of flagged components
        outlier_idx = np.where(flagged)[0].tolist()

        if len(outlier_idx) > 0:
            self.log(f"=== Total flagged components: {outlier_idx} ===")
            ica.exclude.extend(outlier_idx)
        else:
            self.log("=== No components flagged by PCA-GESD ===")

        self.log(f"After PCA-GESD: {len(ica.exclude)} excluded components")

        return ica

    def _ica_component_diagnostics(
        self,
        ica: mne.preprocessing.ICA,
        inst: mne.io.BaseRaw,
        sfreq: float | None = None,
        fmin: float = 7,
        fmax: float = 45,
    ) -> dict:
        """Compute diagnostic metrics for ICA components.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        inst : mne.io.BaseRaw
            Raw data or epochs for computing sources.
        sfreq : float | None
            Sampling frequency. If None, extracted from inst.info.
        fmin : float
            Minimum frequency for spectral slope calculation (default 7 Hz).
        fmax : float
            Maximum frequency for spectral slope calculation (default 45 Hz).

        Returns
        -------
        diagnostics : dict
            Dictionary with raw metrics for each component.
        """
        if sfreq is None:
            sfreq = float(inst.info["sfreq"])

        # === Get topographies ===
        topos = ica.get_components()  # (n_channels, n_components)

        # Undo pre-whitening for sensor-space metrics
        if ica.pre_whitener_.ndim == 1 or ica.pre_whitener_.shape[1] == 1:
            topos_sensor = topos * ica.pre_whitener_.ravel()[:, np.newaxis]
        else:
            topos_sensor = np.linalg.solve(ica.pre_whitener_, topos)

        topo_norms_sq = np.sum(topos_sensor**2, axis=0)

        # === Get source time courses ===
        sources = ica.get_sources(inst).get_data()
        if sources.ndim == 3:  # Epochs
            sources = sources.reshape(sources.shape[1], -1)

        n_components, n_times = sources.shape

        # === Raw metrics ===

        # Variance-based
        source_vars = np.var(sources, axis=1)
        sensor_var = topo_norms_sq * source_vars

        source_deriv_vars = np.var(np.diff(sources, axis=1), axis=1)
        sensor_deriv_var = topo_norms_sq * source_deriv_vars

        hf_ratio = sensor_deriv_var / (sensor_var + 1e-20)

        # Temporal kurtosis
        source_kurt = kurtosis(sources, axis=1, fisher=True)

        # 1-lag autocorrelation
        s_centered = sources - sources.mean(axis=1, keepdims=True)
        autocorr_num = np.sum(s_centered[:, :-1] * s_centered[:, 1:], axis=1)
        autocorr_denom = np.sum(s_centered**2, axis=1)
        autocorr_1lag = autocorr_num / autocorr_denom

        # Spectral slope
        nperseg = min(n_times, int(2 * sfreq))
        freqs, psd = welch(sources, sfreq, nperseg=nperseg, axis=1)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        log_freqs = np.log10(freqs[freq_mask])
        log_psd = np.log10(psd[:, freq_mask] + 1e-20)
        X = np.column_stack([log_freqs, np.ones_like(log_freqs)])
        spectral_slope = np.linalg.lstsq(X, log_psd.T, rcond=None)[0][0]

        # Spatial kurtosis
        spatial_kurt = kurtosis(topos, axis=0, fisher=True)

        return {
            "sensor_var": sensor_var,
            "sensor_deriv_var": sensor_deriv_var,
            "hf_ratio": hf_ratio,
            "source_kurtosis": source_kurt,
            "autocorr_1lag": autocorr_1lag,
            "spectral_slope": spectral_slope,
            "spatial_kurtosis": spatial_kurt,
        }

    def _prepare_metrics_for_gesd(self, diagnostics: dict) -> list:
        """Transform metrics and specify outlier direction for GESD.

        Parameters
        ----------
        diagnostics : dict
            Dictionary from _ica_component_diagnostics.

        Returns
        -------
        metrics : list
            List of (metric_name, transformed_values, outlier_side) tuples.
            outlier_side: 1 = high values are bad, -1 = low values are bad.
        """
        metrics = []

        # 1. Log HF ratio: high = high-frequency artifact (muscle)
        log_hf = np.log(diagnostics["hf_ratio"] + 1e-10)
        metrics.append(("log_hf_ratio", log_hf, 1))

        # 2. |Temporal kurtosis|: high = transient artifact
        abs_kurt = np.abs(diagnostics["source_kurtosis"])
        metrics.append(("abs_temporal_kurtosis", abs_kurt, 1))

        # 3. Autocorrelation: low = white noise artifact
        #    Neural signals have high autocorr; noise has low
        metrics.append(("autocorr_1lag", diagnostics["autocorr_1lag"], -1))

        # 4. Spectral slope: high (less negative) = flat spectrum = muscle/noise
        metrics.append(("spectral_slope", diagnostics["spectral_slope"], 1))

        # 5. Spatial kurtosis: high = focal/single-channel artifact
        metrics.append(("abs_spatial_kurtosis", diagnostics["spatial_kurtosis"], 1))

        return metrics


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = AutoICAAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
