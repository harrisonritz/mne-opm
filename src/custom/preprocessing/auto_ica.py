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

from pathlib import Path
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
    - Spatial template matching via corrmap (if _corrmap_bads=True)

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

        # Spatial template matching via corrmap
        if getattr(self.cfg, "_corrmap_bads", False):
            ica = self._label_by_corrmap(ica, raw)

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

    def _label_by_corrmap(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Identify bad components by matching against pre-computed spatial templates.

        Loads SVD-based topographic templates produced by ``make_ica_template.py``
        and uses ``mne.preprocessing.corrmap`` to find ICA components whose
        spatial patterns correlate with each template.  This is useful for
        detecting EOG / ECG artifacts when no physiological reference channel
        is available.

        Template directory layout (produced by ``make_ica_template.py``)::

            <_corrmap_template_dir>/
                eog_channel_names.npy    # 1-D array of channel-name strings
                eog_templates.npy        # 2-D (n_channels, n_templates) matrix
                ecg_channel_names.npy    # (optional, same format for ECG)
                ecg_templates.npy        # (optional)

        Channel alignment
        -----------------
        Each artifact type has its own channel-name file.  The current ICA may
        have a different (usually smaller) channel set due to sensor dropout.
        For each template column, a per-channel lookup maps reference values
        onto the ICA's channel order; channels in the ICA that are absent from
        the reference get a template weight of 0 and are effectively ignored
        in the correlation.

        Configuration attributes
        ------------------------
        _corrmap_bads : bool
            Master enable switch (default ``False``).
        _corrmap_template_dir : str
            Path to the template directory.
        _n_eog_templates : int
            Number of EOG template columns to use (default 3).  Set to 0 to
            skip EOG matching entirely.
        _n_ecg_templates : int
            Number of ECG template columns to use (default 3).  Set to 0 to
            skip ECG matching entirely.
        _corrmap_threshold : float | 'auto'
            Correlation threshold passed to corrmap (default ``'auto'``).

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Raw data (used for channel-type metadata).

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with matched components added to ``ica.exclude`` and
            ``ica.labels_``.
        """
        self.log("Identifying bad components using corrmap template matching...")

        # --- Validate template directory ---
        template_dir_str = getattr(self.cfg, "_corrmap_template_dir", "")
        if not template_dir_str:
            self.log("_corrmap_template_dir not set; skipping")
            return ica

        template_dir = Path(template_dir_str)
        if not template_dir.is_dir():
            self.log(f"Template directory not found: {template_dir}; skipping")
            return ica

        # --- Determine which artifact types and how many templates to use ---
        threshold = getattr(self.cfg, "_corrmap_threshold", "auto")
        ch_type = (
            self.cfg.ch_types[0]
            if hasattr(self.cfg, "ch_types") and self.cfg.ch_types
            else "mag"
        )

        type_configs = []
        n_eog = getattr(self.cfg, "_n_eog_templates", 3)
        if n_eog > 0:
            type_configs.append(("eog", n_eog))
        n_ecg = getattr(self.cfg, "_n_ecg_templates", 3)
        if n_ecg > 0:
            type_configs.append(("ecg", n_ecg))

        if not type_configs:
            self.log("No artifact types enabled for corrmap; skipping")
            return ica

        ica_channels = ica.ch_names

        # --- Run corrmap for each artifact type ---
        for artifact_type, n_templates in type_configs:

            # Load channel names for this artifact type
            ch_names_path = template_dir / f"{artifact_type}_channel_names.npy"
            if not ch_names_path.exists():
                self.log(
                    f"{ch_names_path.name} not found in {template_dir}; "
                    f"skipping {artifact_type}"
                )
                continue

            ref_channels = np.load(
                str(ch_names_path), allow_pickle=True
            ).tolist()
            ref_channel_to_idx = {
                name: i for i, name in enumerate(ref_channels)
            }

            # Report channel alignment
            n_shared = sum(
                1 for ch in ica_channels if ch in ref_channel_to_idx
            )
            n_total = len(ica_channels)
            self.log(
                f"{artifact_type} channel alignment: {n_shared}/{n_total} ICA "
                f"channels found in reference set ({len(ref_channels)} channels)"
            )
            if n_shared < 0.9 * n_total:
                self.log(
                    f"Warning: only {n_shared}/{n_total} ICA channels are in "
                    "the reference set — template matching may be unreliable."
                )

            # Load templates matrix
            templates_path = template_dir / f"{artifact_type}_templates.npy"
            if not templates_path.exists():
                self.log(
                    f"{templates_path.name} not found in {template_dir}; "
                    f"skipping {artifact_type}"
                )
                continue

            templates_matrix = np.load(str(templates_path))
            if templates_matrix.ndim == 1:
                templates_matrix = templates_matrix[:, np.newaxis]
            n_available = templates_matrix.shape[1]
            n_use = min(n_templates, n_available)

            self.log(
                f"Running corrmap: {n_use} {artifact_type} template(s) "
                f"(of {n_available} available), threshold={threshold!r}"
            )

            # Accumulate matched indices across all templates for this type.
            # Each corrmap call may overwrite ica.labels_[artifact_type], so we
            # collect the union manually.
            accumulated_idx: set[int] = set(
                ica.labels_.get(artifact_type, [])
            )

            for ti in range(n_use):
                template_ref = templates_matrix[:, ti]

                # Align template to ICA channel order.
                # For channels not in the reference, weight = 0 (ignored).
                template_aligned = np.array(
                    [
                        template_ref[ref_channel_to_idx[ch]]
                        if ch in ref_channel_to_idx
                        else 0.0
                        for ch in ica_channels
                    ],
                    dtype=float,
                )

                # Clear the label so we can detect only the new matches from
                # this template call.
                ica.labels_.pop(artifact_type, None)

                try:
                    mne.preprocessing.corrmap(
                        [ica],
                        template=template_aligned,
                        label=artifact_type,
                        threshold=threshold,
                        ch_type=ch_type,
                        plot=False,
                        show=False,
                        verbose=False,
                    )
                except Exception as exc:
                    self.log(
                        f"  template {ti}: corrmap raised "
                        f"{type(exc).__name__}({exc}); skipping this template"
                    )
                    continue

                newly_matched = set(ica.labels_.get(artifact_type, []))
                new_finds = newly_matched - accumulated_idx
                accumulated_idx |= newly_matched

                self.log(
                    f"  template {ti}: "
                    + (
                        f"{len(new_finds)} new → {sorted(new_finds)}"
                        if new_finds
                        else "no new matches"
                    )
                )

            # Write back accumulated labels and extend exclude list.
            final_idx = sorted(accumulated_idx)
            ica.labels_[artifact_type] = final_idx
            if final_idx:
                self.log(
                    f"{artifact_type}: adding components {final_idx} to exclude"
                )
                ica.exclude.extend(final_idx)
            else:
                self.log(f"{artifact_type}: no components matched")

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
            if side == 0:
                direction = "both tails"
            elif side == 1:
                direction = "high=bad"
            else:
                direction = "low=bad"
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


        # print("SKIPPING PCA-GESD FLAGGING FOR NOW TO CHECK FOR FALSE POSITIVES")
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

        # Mean absolute gradient (FASTER: Nolan et al. 2010)
        source_diffs = np.diff(sources, axis=1)
        mean_abs_grad = np.mean(np.abs(source_diffs), axis=1)

        # Temporal kurtosis
        source_kurt = kurtosis(sources, axis=1, fisher=True)

        # 1-lag autocorrelation
        s_centered = sources - sources.mean(axis=1, keepdims=True)
        autocorr_num = np.sum(s_centered[:, :-1] * s_centered[:, 1:], axis=1)
        autocorr_denom = np.sum(s_centered**2, axis=1)
        autocorr_1lag = autocorr_num / autocorr_denom

        # Spectral slope, derivative, and residual metrics
        nperseg = min(n_times, int(2 * sfreq))
        freqs, psd = welch(sources, sfreq, nperseg=nperseg, axis=1)

        # High-frequency power ratio: fraction of power above fmin Hz.
        # Muscle/noise has flat broadband spectra; brain has steep 1/f rolloff.
        hf_power = np.sum(psd[:, freqs >= fmin], axis=1)
        total_power = np.sum(psd, axis=1)
        hf_ratio = hf_power / (total_power + 1e-20)

        # --- Spectral derivative kurtosis (all Welch frequencies) ---
        # d(log_psd)/df: fractional change in power per Hz.
        # A boxcar artifact creates two sharp steps (onset/offset) → heavy-tailed
        # derivative distribution → high kurtosis.
        df = freqs[1] - freqs[0]  # uniform frequency spacing from Welch
        log_psd_all = np.log10(psd + 1e-20)  # (n_components, n_freqs)
        spectral_deriv = np.diff(log_psd_all, axis=1) / df  # (n_components, n_freqs-1)
        spectral_deriv_kurtosis = kurtosis(spectral_deriv, axis=1, fisher=True)

        # --- Spectral slope and residual kurtosis (fmin-fmax band) ---
        # Fit a power-law (1/f^n) baseline in log-log space, then take the
        # kurtosis of the residuals. Narrow-band artifacts create a concentrated
        # bump above the baseline → high residual kurtosis.
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        log_freqs = np.log10(freqs[freq_mask])
        log_psd = np.log10(psd[:, freq_mask] + 1e-20)
        X = np.column_stack([log_freqs, np.ones_like(log_freqs)])
        coeffs = np.linalg.lstsq(X, log_psd.T, rcond=None)[0]  # (2, n_components)
        spectral_slope = coeffs[0]  # (n_components,)

        # Residuals: how much each frequency deviates from the 1/f baseline
        log_psd_fit = (X @ coeffs).T  # (n_components, n_freqs_masked)
        residuals = log_psd - log_psd_fit  # (n_components, n_freqs_masked)
        spectral_resid_kurtosis = kurtosis(residuals, axis=1, fisher=True)

        # Spatial kurtosis
        spatial_kurt = kurtosis(topos, axis=0, fisher=True)

        return {
            "sensor_var": sensor_var,
            "hf_ratio": hf_ratio,
            "mean_abs_gradient": mean_abs_grad,
            "source_kurtosis": source_kurt,
            "autocorr_1lag": autocorr_1lag,
            "spectral_slope": spectral_slope,
            "spatial_kurtosis": spatial_kurt,
            "spectral_deriv_kurtosis": spectral_deriv_kurtosis,
            "spectral_resid_kurtosis": spectral_resid_kurtosis,
        }

    # All available GESD metric names (used for validation).
    AVAILABLE_GESD_METRICS = [
        "log_hf_ratio",
        "temporal_kurtosis_sqrt",
        "autocorr_fisher_z",
        "spectral_slope",
        "spatial_kurtosis_sqrt",
        "spectral_deriv_kurtosis_sqrt",
        "spectral_resid_kurtosis_sqrt",
        "log_mean_abs_gradient",
    ]

    def _prepare_metrics_for_gesd(self, diagnostics: dict) -> list:
        """Transform metrics and specify outlier direction for GESD.

        Which metrics are included can be controlled by setting
        ``cfg._gesd_metrics`` to a list of metric name strings.  When the
        attribute is absent or ``None``, all metrics are used.

        Available metric names:
            ``log_hf_ratio``, ``temporal_kurtosis_sqrt``,
            ``autocorr_fisher_z``, ``spectral_slope``,
            ``spatial_kurtosis_sqrt``, ``spectral_deriv_kurtosis_sqrt``,
            ``spectral_resid_kurtosis_sqrt``, ``log_mean_abs_gradient``.

        Parameters
        ----------
        diagnostics : dict
            Dictionary from _ica_component_diagnostics.

        Returns
        -------
        metrics : list
            List of (metric_name, transformed_values, outlier_side) tuples.
            outlier_side: 1 = high values are bad, -1 = low values are bad.

        Notes
        -----
        GESD assumes normality. Transforms are chosen to improve normality:
        - Log transforms for positive-valued, right-skewed metrics
        - Raw values when already approximately normal
        - Signed sqrt for kurtosis to preserve sign while reducing skew
        """
        # Determine which metrics the user wants
        selected = getattr(self.cfg, "_gesd_metrics", None)
        if selected is not None:
            unknown = set(selected) - set(self.AVAILABLE_GESD_METRICS)
            if unknown:
                raise ValueError(
                    f"Unknown GESD metric names: {unknown}. "
                    f"Available: {self.AVAILABLE_GESD_METRICS}"
                )
            use = set(selected)
            self.log(f"Using selected GESD metrics: {sorted(use)}")
        else:
            use = set(self.AVAILABLE_GESD_METRICS)

        metrics = []

        # 1. Log HF ratio: high = high-frequency artifact (muscle)
        if "log_hf_ratio" in use:
            log_hf = np.log(diagnostics["hf_ratio"] + 1e-10)
            metrics.append(("log_hf_ratio", log_hf, 1))

        # 2. Temporal kurtosis: high = non-Gaussian artifact
        if "temporal_kurtosis_sqrt" in use:
            source_kurt = diagnostics["source_kurtosis"]
            signed_sqrt_kurt = np.sign(source_kurt) * np.sqrt(np.abs(source_kurt)  + 1e-10)
            metrics.append(("temporal_kurtosis_sqrt", signed_sqrt_kurt, 1))

        # 3. Autocorrelation: low = white noise artifact
        if "autocorr_fisher_z" in use:
            autocorr = diagnostics["autocorr_1lag"]
            autocorr_clipped = np.clip(autocorr, -0.999, 0.999)
            fisher_z = np.arctanh(autocorr_clipped)  # Fisher z-transform
            metrics.append(("autocorr_fisher_z", fisher_z, -1))

        # 4. Spectral slope: high (less negative) = flat spectrum = muscle/noise
        if "spectral_slope" in use:
            metrics.append(("spectral_slope", diagnostics["spectral_slope"], 1))

        # 5. Spatial kurtosis: high = focal/single-channel artifact
        if "spatial_kurtosis_sqrt" in use:
            spatial_kurt = diagnostics["spatial_kurtosis"]
            signed_sqrt_spatial = np.sign(spatial_kurt) * np.sqrt(np.abs(spatial_kurt) + 1e-10)
            metrics.append(("spatial_kurtosis_sqrt", signed_sqrt_spatial, 1))

        # 6. Spectral derivative kurtosis: high = sharp narrow-band transitions.
        # d(log_psd)/df has heavy tails when the spectrum has sudden onset/offset
        # edges (boxcar-like artifact). Natural 1/f spectra are smooth → low kurtosis.
        if "spectral_deriv_kurtosis_sqrt" in use:
            spec_deriv_kurt = diagnostics["spectral_deriv_kurtosis"]
            signed_sqrt_spec_deriv = np.sign(spec_deriv_kurt) * np.sqrt(np.abs(spec_deriv_kurt) + 1e-10)
            metrics.append(("spectral_deriv_kurtosis_sqrt", signed_sqrt_spec_deriv, 1))

        # 7. Spectral residual kurtosis: high = concentrated deviation from 1/f.
        # Residuals from the power-law fit are near-Gaussian for typical brain ICs.
        # A narrow-band artifact creates a localized bump above the baseline
        # → heavy-tailed residuals → high kurtosis.
        if "spectral_resid_kurtosis_sqrt" in use:
            spec_resid_kurt = diagnostics["spectral_resid_kurtosis"]
            signed_sqrt_spec_resid = np.sign(spec_resid_kurt) * np.sqrt(np.abs(spec_resid_kurt) + 1e-10)
            metrics.append(("spectral_resid_kurtosis_sqrt", signed_sqrt_spec_resid, 1))

        # 8. Mean absolute gradient (FASTER): high = temporally rough signal.
        # Brain sources are smooth (dominated by low-freq oscillations);
        # muscle and spike artifacts have rapid moment-to-moment fluctuations.
        if "log_mean_abs_gradient" in use:
            metrics.append(("log_mean_abs_gradient",
                            np.log(diagnostics["mean_abs_gradient"] + 1e-10), 1))

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
