"""Automatic ICA component labeling for OPM-MEG data.

This module identifies bad ICA components with a single, unified
PCA-whitened GESD procedure. Every detection strategy contributes a
per-IC *score* vector; all selected scores are z-scored, projected onto
principal components (eigenscores), and a generalized ESD test (GESD) is
applied to each eigenscore with the family-wise error rate controlled by a
Šidák correction across the principal components.

The scores that can be included are:

1. **Diagnostic property metrics**: spectral, kurtosis, autocorrelation,
   gradient and variance statistics of the component time courses/topographies.

2. **EOG correlation** (``find_bads_eog``) against the virtual EOG channels.

3. **ECG correlation** (``find_bads_ecg``, CTPS) against an ECG signal that is
   synthesized from the magnetometers when no ECG channel is present.

4. **Reference-sensor correlation** (``find_bads_ref``) via a separate ICA fit
   on the reference sensors (environmental noise).

5. **Corrmap template correlation**: each component's topography correlated
   against pre-computed EOG/ECG spatial templates.

Folding every detector into one GESD family removes the redundancy and
uncontrolled error rate of running several independent thresholded detectors,
while keeping each score individually selectable through configuration.

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
    _ica_metrics : list[str] | None
        The single list of per-IC scores to feed into the unified GESD. Valid
        names are in ``AVAILABLE_ICA_SCORES`` — the diagnostic property metrics
        plus the artifact-targeted scores ``"eog"``, ``"ecg"``, ``"reference"``,
        ``"corrmap_eog"`` and ``"corrmap_ecg"``. ``None`` (default) selects all
        available scores; an empty list disables the GESD entirely.
    _corrmap_template_dir, _n_eog_templates, _n_ecg_templates :
        Corrmap template location and per-type column counts, used when
        ``"corrmap_eog"`` / ``"corrmap_ecg"`` are selected.
    _auto_ica_overlay : bool
        Save ``ica.plot_overlay`` PNGs (report-style evoked butterfly) after
        each per-PC GESD step plus a final overlay, and the PCA diagnostic
        figures, into the participant's ``meg/ICA`` directory. Default: True.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import welch

import mne
import mne_bids
from mne_bids import BIDSPath, find_matching_paths

from ._base import BaseAnalysis
from ._io import save_ica_bids
from .pca_gesd import (
    MetricSpec,
    PCAGesdResult,
    empty_result,
    fisher_z,
    run_pca_gesd,
    save_pca_gesd_figures,
)


class AutoICAAnalysis(BaseAnalysis):
    """Automatic ICA component labeling.

    Every detection strategy contributes a per-IC score; the selected scores
    are z-scored, projected onto principal components, and a GESD test is run
    on each eigenscore with a single Šidák-controlled family-wise error rate.
    Available scores: diagnostic property metrics, EOG/ECG/reference
    correlation, and corrmap template correlation (see ``_ica_metrics``).

    Components flagged by the unified GESD are added to ica.exclude
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
        """Save ICA solution with updated exclusions and detailed TSV.

        Parameters
        ----------
        results : dict
            Dictionary with labeled ICA.
        """
        self.log("Saving ICA results...")

        ica = results["ica"]

        # Build detailed components TSV with method attribution
        components_df = self._build_components_tsv(ica)
        save_ica_bids(ica, self.cfg, components_df=components_df)

        self.log(f"Saved ICA with {len(ica.exclude)} excluded components")
        self.log(
            f"Components TSV columns: {list(components_df.columns)}"
        )

    def _auto_ica(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> mne.preprocessing.ICA:
        """Apply the unified PCA→GESD ICA component labeling.

        All selected detectors are turned into per-IC scores, projected onto
        principal components, and GESD-tested per eigenscore under one Šidák
        family-wise error rate (see :meth:`_run_unified_gesd`). Report-style
        overlays are written after each per-PC step (cumulative) plus a final
        overlay, and PCA diagnostic figures are saved alongside.

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
        # Initialize per-component label tracking for TSV attribution
        self._component_labels = {i: [] for i in range(ica.n_components_)}
        self._gesd_result: PCAGesdResult | None = None

        # Evoked used for the report-style overlays (built once, only if needed).
        evoked = (
            self._make_overlay_evoked(raw)
            if getattr(self.cfg, "_auto_ica_overlay", True)
            else None
        )

        # Any exclusions already on the loaded ICA (e.g. ICALabel) are preserved.
        cumulative = set(ica.exclude)

        # (a) Overlay before any custom labelling.
        self._save_ica_overlay(ica, evoked, 0, "pre-custom")

        # Compute every selected per-IC score.
        score_specs = self._compute_ic_scores(ica, raw)
        if not score_specs:
            self.log("No scores selected/available; skipping unified GESD.")
            ica.exclude = sorted(cumulative)
            self.log(f"Total excluded components: {len(ica.exclude)}")
            return ica

        # Unified PCA-whitened GESD across all scores.
        ica, gesd = self._run_unified_gesd(ica, raw, score_specs)
        self._gesd_result = gesd

        # (b) Cumulative per-PC (per-eigenscore) overlays.
        for p in range(gesd.n_pcs):
            cumulative |= set(np.where(gesd.per_pc_flagged[p])[0].tolist())
            ica.exclude = sorted(cumulative)
            self._save_ica_overlay(ica, evoked, p + 1, f"gesd-PC{p + 1}")

        # Finalize exclude list (union of all flagged + any pre-existing).
        ica.exclude = sorted(cumulative)

        # (c) Final overlay showing all excluded components.
        self._save_ica_overlay(ica, evoked, gesd.n_pcs + 1, "final")

        # PCA diagnostic figures.
        self._save_gesd_figures(gesd)

        self.log(f"Total excluded components: {len(ica.exclude)}")
        self.log(f"Excluded: {ica.exclude}")

        return ica

    def _overlay_basepath(self) -> tuple[Path, str]:
        """Resolve the output directory and BIDS basename for ICA figures.

        Figures (overlays and PCA diagnostics) are written into the
        participant's ``ICA`` subfolder of the ``meg`` derivatives directory,
        matching where mne-bids-pipeline saves its ICA figures, and sharing the
        ``proc-ica`` prefix.

        Returns
        -------
        out_dir : Path
            ``{deriv_root}/sub-XX/ses-YY/meg/ICA`` directory.
        basename : str
            ``sub-XX_ses-YY_task-<task>_proc-ica`` prefix.
        """
        subject = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        bp = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="ica",
            processing="ica",
            extension=".fif",
            check=False,
        )
        out_dir = Path(bp.directory) / "ICA"
        basename = f"sub-{subject}_ses-{session}_task-{self.cfg.task}_proc-ica"
        return out_dir, basename

    def _make_overlay_evoked(self, raw: mne.io.BaseRaw) -> "mne.Evoked | None":
        """Build the Evoked used for report-style ICA overlays.

        To reproduce the mne-bids-pipeline report overlay
        (``ica.plot_overlay(inst=epochs.average())``), the magnetometer
        ``icafit`` epochs are loaded and averaged.  When that file is not
        available (e.g. in unit tests), fall back to averaging fixed-length
        epochs cut from the cleaned raw.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Cleaned raw data (fallback source for the Evoked).

        Returns
        -------
        evoked : mne.Evoked | None
            Magnetometer Evoked, or ``None`` if neither source is usable
            (overlays are then skipped without aborting labelling).
        """
        subject = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        # Preferred: the pipeline's icafit epochs (matches the report overlay).
        try:
            bp_epo = BIDSPath(
                root=self.cfg.deriv_root,
                subject=subject,
                session=session,
                task=self.cfg.task,
                datatype="meg",
                suffix="epo",
                processing="icafit",
                extension=".fif",
                check=False,
            )
            if bp_epo.fpath.exists():
                epochs = mne.read_epochs(bp_epo.fpath, preload=True, verbose="ERROR")
                evoked = epochs.pick("mag").average()
                self.log("Built overlay Evoked from icafit epochs.")
                return evoked
        except Exception as exc:
            self.log(
                f"Could not load icafit epochs for overlay "
                f"({type(exc).__name__}: {exc}); falling back to raw."
            )

        # Fallback: fixed-length epochs from the cleaned raw.
        try:
            raw_mag = raw.copy().pick("mag")
            duration = float(min(2.0, raw_mag.times[-1]))
            epochs = mne.make_fixed_length_epochs(
                raw_mag, duration=duration, preload=True, verbose="ERROR"
            )
            evoked = epochs.average()
            self.log("Built overlay Evoked from fixed-length raw epochs.")
            return evoked
        except Exception as exc:
            self.log(
                f"Could not build overlay Evoked "
                f"({type(exc).__name__}: {exc}); overlays will be skipped."
            )
            return None

    def _save_ica_overlay(
        self,
        ica: mne.preprocessing.ICA,
        evoked: "mne.Evoked | None",
        step_idx: int,
        step_label: str,
    ) -> None:
        """Save a report-style ICA overlay for the current (cumulative) exclusions.

        Renders ``ica.plot_overlay(inst=evoked)`` — the averaged evoked response
        before (red) and after (black) removing the components currently in
        ``ica.exclude`` — matching the mne-bids-pipeline report overlay.  Because
        ``ica.exclude`` grows across the per-PC GESD steps, successive calls
        produce cumulative overlays.  Files are named::

            <basename>_icaOverlay_<NN>_<step_label>.png

        e.g. ``sub-009_ses-01_task-TSX_proc-ica_icaOverlay_01_gesd-PC1.png`` in the
        participant's ``meg/ICA`` folder.

        Disabled by setting ``cfg._auto_ica_overlay = False``.  A missing Evoked
        or any plotting failure is logged and swallowed so it never aborts
        labelling.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution with the current ``exclude`` list.
        evoked : mne.Evoked | None
            Averaged response passed to ``plot_overlay`` as ``inst``.
        step_idx : int
            Step number used for the zero-padded filename ordinal.
        step_label : str
            Short human-readable step label (e.g. ``"gesd-PC1"``).
        """
        if not getattr(self.cfg, "_auto_ica_overlay", True):
            return
        if evoked is None:
            return

        import matplotlib.pyplot as plt

        out_dir, basename = self._overlay_basepath()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{basename}_icaOverlay_{step_idx:02d}_{step_label}.png"

        n_excl = len(ica.exclude)
        title = (
            f"Step {step_idx:02d}: {step_label} — {n_excl} excluded "
            f"(red=before, black=after cleaning)"
        )

        try:
            fig = ica.plot_overlay(
                inst=evoked, show=False, on_baseline="reapply", title=title
            )
        except Exception as exc:  # plotting must never break labelling
            self.log(
                f"Overlay plot '{step_label}' failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        try:
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            self.log(f"Saved ICA overlay: {fname.name}")
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Unified scoring layer
    # ------------------------------------------------------------------

    def _resolve_gesd_scores(self) -> set:
        """Resolve which per-IC scores to feed into the unified GESD.

        Reads the single ``cfg._ica_metrics`` list, which may contain any name
        in ``AVAILABLE_ICA_SCORES`` — the diagnostic property metrics plus the
        artifact-targeted scores ``"eog"``, ``"ecg"``, ``"reference"``,
        ``"corrmap_eog"`` and ``"corrmap_ecg"``.

        - ``None`` (or unset) selects every available score.
        - An empty list disables the GESD entirely (no-op).
        - Unknown names raise ``ValueError``.

        Returns
        -------
        selected : set of str
            Names of the scores to compute and test.
        """
        selected = getattr(self.cfg, "_ica_metrics", None)
        if selected is None:
            self.log("_ica_metrics unset; using all available scores.")
            return set(self.AVAILABLE_ICA_SCORES)

        unknown = set(selected) - set(self.AVAILABLE_ICA_SCORES)
        if unknown:
            raise ValueError(
                f"Unknown _ica_metrics names: {unknown}. "
                f"Available: {self.AVAILABLE_ICA_SCORES}"
            )
        self.log(f"Using configured _ica_metrics: {sorted(set(selected))}")
        return set(selected)

    @staticmethod
    def _reduce_multichannel_scores(scores) -> np.ndarray:
        """Reduce ``find_bads_*`` scores to one |value| per IC.

        ``find_bads_eog``/``find_bads_ref`` return a list of arrays (one per
        physiological/reference channel) when given several channels.  Collapse
        to the elementwise maximum absolute correlation per component (mirroring
        the mne-bids-pipeline EOG handling); take ``|value|`` for a single-array
        result so the score is a non-negative "artifact-ness".
        """
        arr = np.abs(np.asarray(scores, dtype=float))
        if arr.ndim > 1:
            arr = arr.max(axis=0)
        return arr.ravel()

    def _compute_ic_scores(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> list:
        """Compute every selected per-IC score for the unified GESD.

        Combines the diagnostic property metrics (already transformed by
        :meth:`_prepare_metrics_for_gesd`) with the artifact-targeted
        correlation scores (EOG, ECG, reference, corrmap).  The targeted scores
        are maximum-absolute correlations, Fisher z-transformed (``arctanh``)
        for normality before z-scoring, with ``side=+1`` (high = artifact).
        Sanitization and dropping of constant/degenerate metrics happen inside
        :func:`pca_gesd.run_pca_gesd`.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Cleaned raw data (sources, EOG/ECG/reference, topographies).

        Returns
        -------
        specs : list of MetricSpec
            Per-IC scores to feed into the GESD.
        """
        selected = self._resolve_gesd_scores()
        if not selected:
            return []

        specs: list[MetricSpec] = []

        # --- Diagnostic property metrics (each carries its own transform) ---
        diag_names = selected & set(self.AVAILABLE_ICA_METRICS)
        if diag_names:
            diagnostics = self._ica_component_diagnostics(ica, raw)
            for name, vals, side in self._prepare_metrics_for_gesd(
                diagnostics, names=diag_names
            ):
                specs.append(MetricSpec(name, np.asarray(vals, float), side))

        # --- Targeted artifact-correlation scores: atanh(|r|), side=+1 ---
        if "eog" in selected:
            vals = self._score_eog(ica, raw)
            if vals is not None:
                specs.append(MetricSpec("eog", fisher_z(vals), 1))
        if "ecg" in selected:
            vals = self._score_ecg(ica, raw)
            if vals is not None:
                specs.append(MetricSpec("ecg", fisher_z(vals), 1))
        if "reference" in selected:
            vals = self._score_reference(ica, raw)
            if vals is not None:
                specs.append(MetricSpec("reference", fisher_z(vals), 1))
        if "corrmap_eog" in selected or "corrmap_ecg" in selected:
            corr = self._score_corrmap(ica, raw)
            for key in ("corrmap_eog", "corrmap_ecg"):
                if (
                    key in selected
                    and corr is not None
                    and corr.get(key) is not None
                ):
                    specs.append(MetricSpec(key, fisher_z(corr[key]), 1))

        self.log(
            f"Computed {len(specs)} per-IC scores: {[s.name for s in specs]}"
        )
        return specs

    def _score_eog(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "np.ndarray | None":
        """Per-IC EOG correlation score via ``find_bads_eog``.

        Uses ``measure="correlation"`` so the scores are Pearson correlations
        (reduced to max ``|r|`` per IC), suitable for the Fisher z-transform
        applied in :meth:`_compute_ic_scores`.  Returns ``None`` (logged) when
        no EOG channel is present or detection fails.
        """
        try:
            _, scores = ica.find_bads_eog(raw, measure="correlation")
        except Exception as exc:
            self.log(f"EOG scoring skipped ({type(exc).__name__}: {exc}).")
            return None
        return self._reduce_multichannel_scores(scores)

    def _score_ecg(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "np.ndarray | None":
        """Per-IC ECG correlation score via ``find_bads_ecg``.

        Synthesizes an ECG signal from the magnetometers when no ECG channel is
        present, and uses ``method="correlation"`` so the scores are Pearson
        correlations (reduced to max ``|r|`` per IC) for the Fisher z-transform
        applied in :meth:`_compute_ic_scores`.  Returns ``None`` (logged) on
        failure.
        """
        try:
            _, scores = ica.find_bads_ecg(
                raw, method="correlation", measure="correlation"
            )
        except Exception as exc:
            self.log(f"ECG scoring skipped ({type(exc).__name__}: {exc}).")
            return None
        return self._reduce_multichannel_scores(scores)

    def _score_reference(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "np.ndarray | None":
        """Per-IC reference-sensor correlation score via a separate ref-ICA.

        Fits an ICA on the reference sensors, adds the reference sources as
        channels to a *copy* of the raw (so the caller's raw is never mutated),
        and returns the ``find_bads_ref`` per-IC scores.  Returns ``None``
        (logged) when no reference sensors exist or the fit fails.
        """
        try:
            work = raw.copy()
            ref_raw = work.copy().pick("ref_meg").filter(
                l_freq=1, h_freq=None, verbose="ERROR"
            )
            ref_ica = mne.preprocessing.ICA(
                n_components=0.99,
                method="picard",
                fit_params=dict(extended=True),
                max_iter=256,
                allow_ref_meg=True,
            )
            ref_ica.fit(
                ref_raw, decim=2, reject_by_annotation=True, verbose="ERROR"
            )
            ref_src = ref_ica.get_sources(ref_raw)
            ref_src.rename_channels(lambda x: f"REF_{x}")
            work.add_channels([ref_src], force_update_info=True)
            _, scores = ica.find_bads_ref(
                inst=work, method="separate", measure="correlation"
            )
        except Exception as exc:
            self.log(f"Reference scoring skipped ({type(exc).__name__}: {exc}).")
            return None
        return self._reduce_multichannel_scores(scores)

    def _score_corrmap(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "dict | None":
        """Per-IC corrmap template-correlation scores.

        For each artifact type (``eog``/``ecg``) with a template file, aligns
        each template column to the ICA channel order and correlates it
        (Pearson) against every component topography (``ica.get_components()``),
        reducing to the maximum absolute correlation per component across the
        type's template columns.  Returns a dict keyed ``corrmap_eog`` /
        ``corrmap_ecg``; ``None`` when no templates are available.
        """
        template_dir_str = getattr(self.cfg, "_corrmap_template_dir", "")
        if not template_dir_str:
            self.log("_corrmap_template_dir not set; skipping corrmap scores.")
            return None
        template_dir = Path(template_dir_str)
        if not template_dir.is_dir():
            self.log(f"Corrmap template dir not found: {template_dir}; skipping.")
            return None

        type_counts = {
            "eog": getattr(self.cfg, "_n_eog_templates", 3),
            "ecg": getattr(self.cfg, "_n_ecg_templates", 0),
        }
        maps = ica.get_components()  # (n_channels, n_components)
        ica_channels = ica.ch_names

        out: dict = {}
        for artifact_type, n_templates in type_counts.items():
            if n_templates <= 0:
                continue
            ch_names_path = template_dir / f"{artifact_type}_channel_names.npy"
            templates_path = template_dir / f"{artifact_type}_templates.npy"
            if not ch_names_path.exists() or not templates_path.exists():
                self.log(
                    f"Corrmap {artifact_type}: template files missing; skipping."
                )
                continue

            ref_channels = np.load(str(ch_names_path), allow_pickle=True).tolist()
            ref_idx = {name: i for i, name in enumerate(ref_channels)}
            templates_matrix = np.load(str(templates_path))
            if templates_matrix.ndim == 1:
                templates_matrix = templates_matrix[:, np.newaxis]
            n_use = min(n_templates, templates_matrix.shape[1])

            best = np.zeros(maps.shape[1], dtype=float)
            for ti in range(n_use):
                template_ref = templates_matrix[:, ti]
                template_aligned = np.array(
                    [
                        template_ref[ref_idx[ch]] if ch in ref_idx else 0.0
                        for ch in ica_channels
                    ],
                    dtype=float,
                )
                corr = self._pearson_cols(maps, template_aligned)
                best = np.maximum(best, np.abs(corr))

            out[f"corrmap_{artifact_type}"] = best
            self.log(
                f"Corrmap {artifact_type}: scored {n_use} template(s); "
                f"max corr={best.max():.3f}"
            )

        return out if out else None

    @staticmethod
    def _pearson_cols(maps: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Pearson correlation of a template against each column of ``maps``.

        Parameters
        ----------
        maps : np.ndarray
            ``(n_channels, n_components)`` topography matrix.
        template : np.ndarray
            ``(n_channels,)`` template vector.

        Returns
        -------
        corr : np.ndarray
            ``(n_components,)`` correlation per component.
        """
        t = template - template.mean()
        m = maps - maps.mean(axis=0, keepdims=True)
        num = m.T @ t
        den = np.linalg.norm(m, axis=0) * np.linalg.norm(t)
        return num / (den + 1e-30)
    # ------------------------------------------------------------------
    # Unified PCA-whitened GESD (delegates to the generic pca_gesd utility)
    # ------------------------------------------------------------------

    def _run_unified_gesd(
        self,
        ica: mne.preprocessing.ICA,
        raw: mne.io.BaseRaw,
        score_specs: list,
        alpha: float = 0.05,
        p_out: float = 1.0,
        n_pcs: "int | None" = None,
    ) -> tuple:
        """Run the unified PCA-whitened GESD over all per-IC scores.

        Thin wrapper around :func:`custom.preprocessing.pca_gesd.run_pca_gesd`
        that adds the ICA-specific bookkeeping: skipping when too few components
        remain, recording which PC flagged each component (for the TSV), and
        extending ``ica.exclude``.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Unused directly (scores are precomputed); kept for symmetry/testing.
        score_specs : list of MetricSpec
            Per-IC scores to test.
        alpha : float
            Overall family-wise significance level (default 0.05).
        p_out : float
            Maximum fraction of outliers per PC passed to GESD (default 1.0).
        n_pcs : int | None
            Number of PCs to keep; default keeps 99% of the variance.

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA with flagged components added to ``exclude``.
        result : PCAGesdResult
            Detailed PCA/GESD outputs for the TSV and figures.
        """
        # Lazy-init tracking dict so the method can be tested stand-alone.
        if not hasattr(self, "_component_labels"):
            self._component_labels = {i: [] for i in range(ica.n_components_)}

        n_comps = ica.n_components_
        names = [s.name for s in score_specs]
        sides = [s.side for s in score_specs]

        # Skip when too few components remain after existing exclusions.
        n_remaining = n_comps - len(ica.exclude)
        if n_remaining < 5:
            self.log(
                f"Too few components remaining ({n_remaining}) for PCA-GESD; skipping"
            )
            return ica, empty_result(names, sides, n_comps, alpha)

        result = run_pca_gesd(
            score_specs,
            alpha=alpha,
            p_out=p_out,
            n_pcs=n_pcs,
            min_items=5,
            log=self.log,
        )

        # Record per-PC attribution and extend the exclude list.
        for p in range(result.n_pcs):
            for i in np.where(result.per_pc_flagged[p])[0].tolist():
                self._component_labels[i].append(f"GESD_PC{p + 1}")
        outlier_idx = np.where(result.flagged)[0].tolist()
        if outlier_idx:
            ica.exclude.extend(outlier_idx)

        self.log(
            f"After unified GESD: {len(set(ica.exclude))} excluded components"
        )
        return ica, result

    def _save_gesd_figures(self, gesd: "PCAGesdResult | None") -> None:
        """Save the PCA/GESD diagnostic figures into the participant's ICA folder.

        Delegates to :func:`custom.preprocessing.pca_gesd.save_pca_gesd_figures`.
        Disabled by ``cfg._auto_ica_overlay=False``; each figure is isolated so a
        failure never aborts labelling.
        """
        if not getattr(self.cfg, "_auto_ica_overlay", True):
            return
        if gesd is None or gesd.n_pcs == 0:
            self.log("No GESD result/PCs; skipping diagnostic figures.")
            return

        out_dir, basename = self._overlay_basepath()
        item_names = [f"IC{i:03d}" for i in range(gesd.n_items)]
        save_pca_gesd_figures(
            gesd,
            out_dir,
            basename,
            item_label="IC",
            item_names=item_names,
            log=self.log,
        )

    def _ica_component_diagnostics(
        self,
        ica: mne.preprocessing.ICA,
        inst: mne.io.BaseRaw,
        sfreq: float | None = None,
        fmin: float = 7,
        fmax: float = 45,
        line_freq: float | None = None,
        line_bw: float = 1.0,
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
        line_freq : float | None
            Power-line frequency in Hz used for the line-power metric. If None,
            it is read from ``inst.info['line_freq']`` (e.g. 60 Hz in North
            America); if that is also unset, it falls back to 60 Hz.
        line_bw : float
            Half-bandwidth (Hz) of the band centred on ``line_freq`` used to
            measure line-noise power (default 1.0, i.e. ``line_freq ± 1`` Hz).

        Returns
        -------
        diagnostics : dict
            Dictionary with raw metrics for each component.
        """
        if sfreq is None:
            sfreq = float(inst.info["sfreq"])

        # Resolve the power-line frequency: explicit arg > inst.info > 60 Hz.
        if line_freq is None:
            line_freq = inst.info.get("line_freq", None)
        if line_freq is None:
            line_freq = 60.0
            self.log(
                "line_freq not set in info; defaulting to 60.0 Hz for line_ratio"
            )

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

        # Line-frequency power ratio: fraction of power in a narrow band
        # (line_freq ± line_bw Hz) around the mains frequency. Power-line
        # components concentrate spectral power at this frequency, so a high
        # ratio flags line-noise artifacts. The band is empty (ratio 0) if
        # line_freq lies beyond the Welch frequency range (e.g. above Nyquist).
        line_mask = (freqs >= line_freq - line_bw) & (freqs <= line_freq + line_bw)
        line_power = np.sum(psd[:, line_mask], axis=1)
        line_ratio = line_power / (total_power + 1e-20)

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
            "line_ratio": line_ratio,
            "mean_abs_gradient": mean_abs_grad,
            "source_kurtosis": source_kurt,
            "autocorr_1lag": autocorr_1lag,
            "spectral_slope": spectral_slope,
            "spatial_kurtosis": spatial_kurt,
            "spectral_deriv_kurtosis": spectral_deriv_kurtosis,
            "spectral_resid_kurtosis": spectral_resid_kurtosis,
        }

    # All available diagnostic GESD metric names (used for validation).
    AVAILABLE_ICA_METRICS = [
        "log_hf_ratio",
        "log_line_ratio",
        "temporal_kurtosis_sqrt",
        "autocorr_fisher_z",
        "spectral_slope",
        "spatial_kurtosis_sqrt",
        "spectral_deriv_kurtosis_sqrt",
        "spectral_resid_kurtosis_sqrt",
        "log_mean_abs_gradient",
    ]

    # Targeted artifact-correlation scores (in addition to the diagnostics).
    AVAILABLE_ICA_TARGETED = [
        "eog",
        "ecg",
        "reference",
        "corrmap_eog",
        "corrmap_ecg",
    ]

    # Full set of per-IC scores selectable via ``cfg._ica_metrics``.
    AVAILABLE_ICA_SCORES = AVAILABLE_ICA_METRICS + AVAILABLE_ICA_TARGETED

    def _build_components_tsv(
        self, ica: mne.preprocessing.ICA
    ) -> "pd.DataFrame":
        """Build a detailed components TSV from the unified GESD result.

        Reads the existing pipeline-generated TSV (from
        ``_06a2_find_ica_artifacts``) only to preserve any ICALabel
        attributions, then adds the unified-GESD attribution and per-score
        columns.

        Columns produced
        ----------------
        Standard BIDS:
            component, type, description, status, status_description
        Attribution:
            method_gesd (0/1), method_pipeline_icalabel (class label or "n/a")
        GESD detail (only when GESD ran):
            gesd_pcs_flagged, gesd_score_PC1..N, score_<name> per input score,
            gesd_pc_loadings, gesd_var_explained

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA with final exclude list.

        Returns
        -------
        df : pd.DataFrame
            One row per component.
        """
        n_comps = ica.n_components_

        # --- Read existing pipeline TSV for ICALabel attributions only ---
        subject = (
            self.cfg.subjects[0]
            if isinstance(self.cfg.subjects, list)
            else self.cfg.subjects
        )
        session = (
            self.cfg.sessions[0]
            if isinstance(self.cfg.sessions, list)
            else self.cfg.sessions
        )

        tsv_path = BIDSPath(
            root=self.cfg.deriv_root,
            subject=subject,
            session=session,
            task=self.cfg.task,
            datatype="meg",
            suffix="components",
            processing="ica",
            extension=".tsv",
            check=False,
        )

        pipeline_icalabel = ["n/a"] * n_comps

        if tsv_path.fpath.exists():
            existing = pd.read_csv(tsv_path.fpath, sep="\t")
            for _, row in existing.iterrows():
                comp = int(row["component"])
                if comp >= n_comps:
                    continue
                desc = str(row.get("status_description", "n/a"))
                if "(MNE-ICALabel)" in desc:
                    label = (
                        desc.replace("Auto-detected ", "")
                        .replace(" (MNE-ICALabel)", "")
                    )
                    pipeline_icalabel[comp] = label
                    self._component_labels[comp].append(f"icalabel_{label}")
        else:
            self.log(
                "No existing components TSV found; ICALabel attributions "
                "will not be available."
            )

        # --- Build status_description from all labels ---
        status_descriptions = []
        for i in range(n_comps):
            labels = self._component_labels[i]
            status_descriptions.append("; ".join(labels) if labels else "n/a")

        # --- Core columns ---
        data: dict[str, list] = {
            "component": list(range(n_comps)),
            "type": ["ica"] * n_comps,
            "description": ["Independent Component"] * n_comps,
            "status": [
                "bad" if i in set(ica.exclude) else "good"
                for i in range(n_comps)
            ],
            "status_description": status_descriptions,
            # --- Attribution ---
            "method_gesd": [
                int(
                    any(
                        lbl.startswith("GESD_PC")
                        for lbl in self._component_labels[i]
                    )
                )
                for i in range(n_comps)
            ],
            "method_pipeline_icalabel": pipeline_icalabel,
        }

        # --- GESD detail columns (only if the unified GESD produced PCs) ---
        gesd = getattr(self, "_gesd_result", None)
        if gesd is not None and gesd.n_pcs > 0:
            n_pcs = gesd.n_pcs
            metric_names = gesd.metric_names

            # Which PCs flagged each component
            data["gesd_pcs_flagged"] = [
                ";".join(
                    lbl
                    for lbl in self._component_labels[i]
                    if lbl.startswith("GESD_PC")
                )
                or "n/a"
                for i in range(n_comps)
            ]

            # Eigenscore per component per PC
            for p in range(n_pcs):
                data[f"gesd_score_PC{p + 1}"] = np.round(
                    gesd.eigenscores[p], 4
                ).tolist()

            # Raw (pre-standardization) value per input score per component
            for j, sname in enumerate(metric_names):
                data[f"score_{sname}"] = np.round(gesd.M[j], 6).tolist()

            # PC loadings (shared across components, stored once per row
            # for self-contained CSV analysis)
            loading_strs = []
            for p in range(n_pcs):
                parts = [
                    f"{metric_names[j]}:{gesd.loadings[j, p]:.3f}"
                    for j in range(len(metric_names))
                ]
                loading_strs.append(f"PC{p + 1}({','.join(parts)})")
            data["gesd_pc_loadings"] = [";".join(loading_strs)] * n_comps

            # Variance explained
            var_str = ";".join(
                f"PC{p + 1}:{gesd.var_explained[p]:.4f}" for p in range(n_pcs)
            )
            data["gesd_var_explained"] = [var_str] * n_comps

        df = pd.DataFrame(data)
        return df

    def _prepare_metrics_for_gesd(
        self, diagnostics: dict, names: "set | list | None" = None
    ) -> list:
        """Transform diagnostic metrics and specify outlier direction for GESD.

        The diagnostic selection is resolved upstream by
        :meth:`_resolve_gesd_scores` (from ``cfg._ica_metrics``); the
        diagnostic subset is passed here via ``names``.  When ``names`` is
        ``None`` all diagnostic metrics are produced.

        Available metric names:
            ``log_hf_ratio``, ``log_line_ratio``,
            ``temporal_kurtosis_sqrt``,
            ``autocorr_fisher_z``, ``spectral_slope``,
            ``spatial_kurtosis_sqrt``, ``spectral_deriv_kurtosis_sqrt``,
            ``spectral_resid_kurtosis_sqrt``, ``log_mean_abs_gradient``.

        Parameters
        ----------
        diagnostics : dict
            Dictionary from _ica_component_diagnostics.
        names : set | list | None
            Explicit diagnostic-metric selection.  ``None`` produces all of
            them.

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
        # Determine which diagnostic metrics to compute.
        if names is not None:
            unknown = set(names) - set(self.AVAILABLE_ICA_METRICS)
            if unknown:
                raise ValueError(
                    f"Unknown GESD metric names: {unknown}. "
                    f"Available: {self.AVAILABLE_ICA_METRICS}"
                )
            use = set(names)
            self.log(f"Using selected GESD metrics: {sorted(use)}")
        else:
            use = set(self.AVAILABLE_ICA_METRICS)

        metrics = []

        # 1. Log HF ratio: high = high-frequency artifact (muscle)
        if "log_hf_ratio" in use:
            log_hf = np.log(diagnostics["hf_ratio"] + 1e-10)
            metrics.append(("log_hf_ratio", log_hf, 1))

        # 1b. Log line-frequency ratio: high = power-line (50/60 Hz) artifact.
        # Line-noise components concentrate power in a narrow band at the mains
        # frequency, giving a high ratio relative to total power.
        if "log_line_ratio" in use:
            log_line = np.log(diagnostics["line_ratio"] + 1e-10)
            metrics.append(("log_line_ratio", log_line, 1))

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
