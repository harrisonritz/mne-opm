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
    _gesd_scores : list[str] | None
        Which per-IC scores to feed into the unified GESD. Valid names are in
        ``AVAILABLE_GESD_SCORES`` (the diagnostic metrics plus ``"eog"``,
        ``"ecg"``, ``"reference"``, ``"corrmap_eog"``, ``"corrmap_ecg"``).
        If ``None`` (default), the selection is derived for backward
        compatibility from the legacy flags ``_gesd_metrics``, ``ref_bads``,
        ``_corrmap_bads`` and ``_gesd_bads``.
    _gesd_bads : bool
        Legacy master switch. When ``False`` the unified GESD is a no-op
        (only consulted when ``_gesd_scores`` is unset). Default: True.
    ref_bads, _corrmap_bads : bool
        Legacy per-method switches used only to derive ``_gesd_scores`` when it
        is unset. Defaults: True / False.
    _auto_ica_overlay : bool
        Save ``ica.plot_overlay`` PNGs (report-style evoked butterfly) after
        each per-PC GESD step plus a final overlay, into the participant's
        ``meg/ICA`` directory. Default: True.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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


@dataclass
class ScoreSpec:
    """A single per-IC score fed into the unified GESD procedure.

    Parameters
    ----------
    name : str
        Identifier for the score (e.g. ``"log_hf_ratio"``, ``"eog"``).
    values : np.ndarray
        Per-IC score values, shape ``(n_components_,)``.
    side : int
        Outlier direction for GESD: ``1`` = high values are bad,
        ``-1`` = low values are bad, ``0`` = both tails.
    group : str
        ``"diagnostic"`` (signed property metric) or ``"targeted"``
        (artifact-correlation score; stored as ``|value|`` with ``side=1``).
    """

    name: str
    values: np.ndarray
    side: int
    group: str


@dataclass
class GesdResult:
    """Outputs of the unified PCA→GESD procedure, used by the TSV and figures.

    Attributes
    ----------
    score_names : list of str
        Names of the input scores, in row order of ``M``.
    sides : np.ndarray
        Outlier side per input score, shape ``(k,)``.
    M : np.ndarray
        Raw (pre-standardization) score matrix, shape ``(k, n)``.
    M_std : np.ndarray
        Standardized score matrix, shape ``(k, n)``.
    loadings : np.ndarray
        PC loadings (how each score weights each PC), shape ``(k, n_pcs)``.
    eigenscores : np.ndarray
        PC scores (how each IC loads on each PC), shape ``(n_pcs, n)``.
    var_explained : np.ndarray
        Variance fraction explained by each retained PC, shape ``(n_pcs,)``.
    var_explained_all : np.ndarray
        Variance fraction for the full singular spectrum (for the scree plot).
    n_pcs : int
        Number of retained principal components.
    pc_sides : np.ndarray
        Outlier side used for GESD on each PC, shape ``(n_pcs,)``.
    alpha : float
        Overall family-wise significance level.
    alpha_per_pc : float
        Šidák-corrected per-PC significance level.
    per_pc_flagged : list of np.ndarray
        Boolean mask of flagged ICs for each PC, each shape ``(n,)``.
    flagged : np.ndarray
        Union boolean mask of flagged ICs, shape ``(n,)``.
    """

    score_names: List[str]
    sides: np.ndarray
    M: np.ndarray
    M_std: np.ndarray
    loadings: np.ndarray
    eigenscores: np.ndarray
    var_explained: np.ndarray
    var_explained_all: np.ndarray
    n_pcs: int
    pc_sides: np.ndarray
    alpha: float
    alpha_per_pc: float
    per_pc_flagged: List[np.ndarray]
    flagged: np.ndarray


class AutoICAAnalysis(BaseAnalysis):
    """Automatic ICA component labeling.

    Every detection strategy contributes a per-IC score; the selected scores
    are z-scored, projected onto principal components, and a GESD test is run
    on each eigenscore with a single Šidák-controlled family-wise error rate.
    Available scores: diagnostic property metrics, EOG/ECG/reference
    correlation, and corrmap template correlation (see ``_gesd_scores``).

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
        self._gesd_result: GesdResult | None = None

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

        Uses ``cfg._gesd_scores`` when set (validated against
        ``AVAILABLE_GESD_SCORES``).  Otherwise derives the selection for
        backward compatibility from the legacy flags: the diagnostic metrics
        from ``cfg._gesd_metrics`` (or all of them), plus ``"reference"`` when
        ``cfg.ref_bads`` and the corrmap scores when ``cfg._corrmap_bads`` and
        the corresponding template count is positive.  ``cfg._gesd_bads=False``
        forces an empty selection (the GESD becomes a no-op).

        Returns
        -------
        selected : set of str
            Names of the scores to compute and test.
        """
        selected = getattr(self.cfg, "_gesd_scores", None)
        if selected is not None:
            unknown = set(selected) - set(self.AVAILABLE_GESD_SCORES)
            if unknown:
                raise ValueError(
                    f"Unknown GESD score names: {unknown}. "
                    f"Available: {self.AVAILABLE_GESD_SCORES}"
                )
            self.log(f"Using configured _gesd_scores: {sorted(set(selected))}")
            return set(selected)

        # Legacy master switch.
        if not getattr(self.cfg, "_gesd_bads", True):
            self.log("_gesd_bads=False and _gesd_scores unset; no scores selected.")
            return set()

        # Diagnostic metrics from the legacy _gesd_metrics (or all).
        metrics = getattr(self.cfg, "_gesd_metrics", None)
        sel = set(self.AVAILABLE_GESD_METRICS) if metrics is None else set(metrics)

        # Reference / corrmap from legacy switches.
        if getattr(self.cfg, "ref_bads", True):
            sel.add("reference")
        if getattr(self.cfg, "_corrmap_bads", False):
            if getattr(self.cfg, "_n_eog_templates", 3) > 0:
                sel.add("corrmap_eog")
            if getattr(self.cfg, "_n_ecg_templates", 0) > 0:
                sel.add("corrmap_ecg")

        self.log(f"Derived _gesd_scores from legacy flags: {sorted(sel)}")
        return sel

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

    @staticmethod
    def _sanitize_score(values: np.ndarray) -> np.ndarray:
        """Replace non-finite score entries with the finite median (or 0)."""
        vals = np.asarray(values, dtype=float).copy()
        finite = np.isfinite(vals)
        if not finite.all():
            fill = float(np.median(vals[finite])) if finite.any() else 0.0
            vals[~finite] = fill
        return vals

    def _compute_ic_scores(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> list:
        """Compute every selected per-IC score for the unified GESD.

        Combines the diagnostic property metrics with the artifact-targeted
        correlation scores (EOG, ECG, reference, corrmap), filtered by
        :meth:`_resolve_gesd_scores`.  Each returned score is finite (NaNs
        imputed) and has positive variance; constant scores are dropped.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Cleaned raw data (sources, EOG/ECG/reference, topographies).

        Returns
        -------
        specs : list of ScoreSpec
            Per-IC scores to feed into the GESD.
        """
        selected = self._resolve_gesd_scores()
        if not selected:
            return []

        specs: list[ScoreSpec] = []

        # --- Diagnostic property metrics ---
        diag_names = selected & set(self.AVAILABLE_GESD_METRICS)
        if diag_names:
            diagnostics = self._ica_component_diagnostics(ica, raw)
            for name, vals, side in self._prepare_metrics_for_gesd(
                diagnostics, names=diag_names
            ):
                specs.append(
                    ScoreSpec(name, np.asarray(vals, float), side, "diagnostic")
                )

        # --- Targeted artifact-correlation scores (|value|, side=+1) ---
        if "eog" in selected:
            vals = self._score_eog(ica, raw)
            if vals is not None:
                specs.append(ScoreSpec("eog", vals, 1, "targeted"))
        if "ecg" in selected:
            vals = self._score_ecg(ica, raw)
            if vals is not None:
                specs.append(ScoreSpec("ecg", vals, 1, "targeted"))
        if "reference" in selected:
            vals = self._score_reference(ica, raw)
            if vals is not None:
                specs.append(ScoreSpec("reference", vals, 1, "targeted"))
        if "corrmap_eog" in selected or "corrmap_ecg" in selected:
            corr = self._score_corrmap(ica, raw)
            for key in ("corrmap_eog", "corrmap_ecg"):
                if (
                    key in selected
                    and corr is not None
                    and corr.get(key) is not None
                ):
                    specs.append(ScoreSpec(key, corr[key], 1, "targeted"))

        # Sanitize and drop zero-variance / wrong-length scores.
        clean: list[ScoreSpec] = []
        for spec in specs:
            vals = self._sanitize_score(spec.values)
            if vals.size != ica.n_components_:
                self.log(
                    f"Score '{spec.name}' has length {vals.size} != "
                    f"{ica.n_components_}; dropping."
                )
                continue
            if np.std(vals) == 0:
                self.log(f"Score '{spec.name}' is constant; dropping.")
                continue
            clean.append(ScoreSpec(spec.name, vals, spec.side, spec.group))

        self.log(
            f"Computed {len(clean)} per-IC scores: {[s.name for s in clean]}"
        )
        return clean

    def _score_eog(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "np.ndarray | None":
        """Per-IC EOG correlation score via ``find_bads_eog``.

        Returns ``None`` (logged) when no EOG channel is present or detection
        fails — typical when virtual EOG channels are absent.
        """
        try:
            _, scores = ica.find_bads_eog(raw)
        except Exception as exc:
            self.log(f"EOG scoring skipped ({type(exc).__name__}: {exc}).")
            return None
        return self._reduce_multichannel_scores(scores)

    def _score_ecg(
        self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw
    ) -> "np.ndarray | None":
        """Per-IC ECG correlation score via ``find_bads_ecg`` (CTPS).

        Synthesizes an ECG signal from the magnetometers when no ECG channel is
        present.  Returns ``None`` (logged) on failure.
        """
        try:
            _, scores = ica.find_bads_ecg(raw, method="ctps")
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
            _, scores = ica.find_bads_ref(inst=work, method="separate")
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
    # Unified PCA-whitened GESD
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

        Steps:
            1. Build the score matrix (k scores x n components); targeted scores
               use ``|value|`` so high = artifact.
            2. Standardize each score (row-wise).
            3. PCA via SVD; keep enough PCs for 99% of the variance.
            4. GESD on each eigenscore with a Šidák-corrected alpha, the tail
               direction taken from the loadings and per-score sides.
            5. Union of flagged components -> ``ica.exclude``.

        Parameters
        ----------
        ica : mne.preprocessing.ICA
            ICA solution.
        raw : mne.io.BaseRaw
            Unused directly (scores are precomputed); kept for symmetry/testing.
        score_specs : list of ScoreSpec
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
        result : GesdResult
            Detailed PCA/GESD outputs for the TSV and figures.
        """
        self.log("Running unified PCA-whitened GESD...")

        # Lazy-init tracking dict so the method can be tested stand-alone.
        if not hasattr(self, "_component_labels"):
            self._component_labels = {i: [] for i in range(ica.n_components_)}

        names = [s.name for s in score_specs]
        sides = np.array([s.side for s in score_specs], dtype=float)
        n_comps = ica.n_components_

        def _empty_result(n_pcs_val: int = 0) -> GesdResult:
            return GesdResult(
                score_names=names,
                sides=sides,
                M=np.empty((len(score_specs), n_comps)),
                M_std=np.empty((len(score_specs), n_comps)),
                loadings=np.empty((len(score_specs), 0)),
                eigenscores=np.empty((0, n_comps)),
                var_explained=np.empty(0),
                var_explained_all=np.empty(0),
                n_pcs=n_pcs_val,
                pc_sides=np.empty(0),
                alpha=alpha,
                alpha_per_pc=alpha,
                per_pc_flagged=[],
                flagged=np.zeros(n_comps, dtype=bool),
            )

        if not score_specs:
            self.log("No scores provided to GESD; skipping.")
            return ica, _empty_result()

        n_remaining = n_comps - len(ica.exclude)
        if n_remaining < 5:
            self.log(
                f"Too few components remaining ({n_remaining}) for PCA-GESD; skipping"
            )
            return ica, _empty_result()

        # Build score matrix: targeted scores use |value| so high = artifact.
        rows = [
            np.abs(s.values) if s.group == "targeted" else s.values
            for s in score_specs
        ]
        M = np.vstack(rows)  # (k, n)
        k, n = M.shape
        self.log(f"Score matrix shape: {k} scores x {n} components")

        self.log("=== Score Summary ===")
        for i, s in enumerate(score_specs):
            direction = {1: "high=bad", -1: "low=bad", 0: "both tails"}[s.side]
            self.log(
                f"  {s.name}: mean={np.mean(M[i]):.3f}, "
                f"std={np.std(M[i]):.3f} ({direction})"
            )

        # Standardize each score (row-wise).
        M_std = StandardScaler().fit_transform(M.T).T  # (k, n)

        # PCA via SVD of the standardized matrix.
        U, sv, Vt = np.linalg.svd(M_std, full_matrices=False)
        var_explained_all = (sv**2) / (sv**2).sum()

        if n_pcs is None:
            cumvar = np.cumsum(var_explained_all)
            n_pcs = int(np.searchsorted(cumvar, 0.99) + 1)
        n_pcs = max(1, min(n_pcs, k))

        loadings = U[:, :n_pcs]  # (k, n_pcs)
        eigenscores = loadings.T @ M_std  # (n_pcs, n)
        var_explained = var_explained_all[:n_pcs]

        self.log("=== PCA Variance Explained ===")
        for p in range(n_pcs):
            self.log(f"  PC{p + 1}: {var_explained[p] * 100:.1f}%")
        self.log(f"  Total ({n_pcs} PCs): {var_explained.sum() * 100:.1f}%")

        self.log("=== PC Loadings (score weights) ===")
        for p in range(n_pcs):
            loading_strs = [f"{names[i]}={loadings[i, p]:.2f}" for i in range(k)]
            self.log(f"  PC{p + 1}: {', '.join(loading_strs)}")

        # Per-PC tail direction from loadings . sides.
        pc_sides = np.sign(loadings.T @ sides)

        # Šidák correction (exact under independence).
        alpha_per_pc = 1 - (1 - alpha) ** (1 / n_pcs)
        self.log(f"Šidák-corrected alpha: {alpha:.3f} -> {alpha_per_pc:.4f} per PC")

        flagged = np.zeros(n, dtype=bool)
        per_pc_flagged: list = []
        self.log("=== GESD Results per PC ===")
        for p in range(n_pcs):
            side = int(pc_sides[p])
            side_str = {1: "upper", -1: "lower", 0: "both"}.get(side, "both")
            flags, _ = osl_gesd(
                eigenscores[p, :],
                alpha=alpha_per_pc,
                p_out=p_out,
                outlier_side=side,
            )
            flags = np.asarray(flags, dtype=bool)
            per_pc_flagged.append(flags)
            flagged |= flags
            if flags.any():
                idx = np.where(flags)[0].tolist()
                for i in idx:
                    self._component_labels[i].append(f"GESD_PC{p + 1}")
                self.log(
                    f"  PC{p + 1} ({side_str} tail): {flags.sum()} outliers -> {idx}"
                )
            else:
                self.log(f"  PC{p + 1} ({side_str} tail): no outliers")

        outlier_idx = np.where(flagged)[0].tolist()
        if outlier_idx:
            self.log(f"=== Total flagged components: {outlier_idx} ===")
            ica.exclude.extend(outlier_idx)
        else:
            self.log("=== No components flagged by unified GESD ===")

        result = GesdResult(
            score_names=names,
            sides=sides,
            M=M,
            M_std=M_std,
            loadings=loadings,
            eigenscores=eigenscores,
            var_explained=var_explained,
            var_explained_all=var_explained_all,
            n_pcs=n_pcs,
            pc_sides=pc_sides,
            alpha=alpha,
            alpha_per_pc=alpha_per_pc,
            per_pc_flagged=per_pc_flagged,
            flagged=flagged,
        )
        self.log(
            f"After unified GESD: {len(set(ica.exclude))} excluded components"
        )
        return ica, result

    # ------------------------------------------------------------------
    # PCA / GESD diagnostic figures
    # ------------------------------------------------------------------

    def _save_gesd_figures(self, gesd: "GesdResult | None") -> None:
        """Save PCA/GESD diagnostic figures into the participant's ICA folder.

        Figures: score-loadings heatmap (how each score loads on each PC),
        IC-eigenscore heatmap (how each IC loads on each PC), scree plot,
        standardized-score correlation matrix, standardized-score heatmap, and
        per-PC outlier scatter.  Disabled by ``cfg._auto_ica_overlay=False``;
        each figure is isolated so a failure never aborts labelling.
        """
        if not getattr(self.cfg, "_auto_ica_overlay", True):
            return
        if gesd is None or gesd.n_pcs == 0:
            self.log("No GESD result/PCs; skipping diagnostic figures.")
            return

        self._save_fig(self._fig_score_loadings, gesd, "gesdLoadings")
        self._save_fig(self._fig_ic_eigenscores, gesd, "gesdEigenscores")
        self._save_fig(self._fig_scree, gesd, "gesdScree")
        self._save_fig(self._fig_score_corr, gesd, "gesdScoreCorr")
        self._save_fig(self._fig_standardized_scores, gesd, "gesdStdScores")
        self._save_fig(self._fig_pc_outliers, gesd, "gesdOutliers")

    def _save_fig(self, builder, gesd: GesdResult, suffix: str) -> None:
        """Build a figure via ``builder(gesd)`` and save it to the ICA folder."""
        import matplotlib.pyplot as plt

        out_dir, basename = self._overlay_basepath()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{basename}_{suffix}.png"
        try:
            fig = builder(gesd)
        except Exception as exc:  # figures must never break labelling
            self.log(f"Figure '{suffix}' failed: {type(exc).__name__}: {exc}")
            return
        if fig is None:
            return
        try:
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            self.log(f"Saved figure: {fname.name}")
        finally:
            plt.close(fig)

    def _fig_score_loadings(self, gesd: GesdResult):
        """Heatmap of PC loadings (scores x PCs)."""
        import matplotlib.pyplot as plt

        k, n_pcs = gesd.loadings.shape
        fig, ax = plt.subplots(
            figsize=(max(4, 0.6 * n_pcs + 2), max(3, 0.35 * k + 1))
        )
        vmax = float(np.abs(gesd.loadings).max()) or 1.0
        im = ax.imshow(
            gesd.loadings, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        ax.set_xticks(range(n_pcs))
        ax.set_xticklabels([f"PC{p + 1}" for p in range(n_pcs)])
        ax.set_yticks(range(k))
        ax.set_yticklabels(gesd.score_names, fontsize=8)
        ax.set_xlabel("Principal component")
        ax.set_title("Score loadings on PCs")
        fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
        fig.tight_layout()
        return fig

    def _fig_ic_eigenscores(self, gesd: GesdResult):
        """Heatmap of IC eigenscores (ICs x PCs), flagged ICs labelled red."""
        import matplotlib.pyplot as plt

        data = gesd.eigenscores.T  # (n, n_pcs)
        n, n_pcs = data.shape
        fig, ax = plt.subplots(
            figsize=(max(4, 0.6 * n_pcs + 2), max(3, 0.18 * n + 1))
        )
        vmax = float(np.abs(data).max()) or 1.0
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_pcs))
        ax.set_xticklabels([f"PC{p + 1}" for p in range(n_pcs)])
        ax.set_ylabel("Independent component")
        ax.set_xlabel("Principal component")
        ax.set_title("IC eigenscores (flagged ICs in red)")
        flagged_idx = np.where(gesd.flagged)[0]
        if flagged_idx.size:
            ax.set_yticks(flagged_idx)
            ax.set_yticklabels(
                [f"IC{i}" for i in flagged_idx], fontsize=7, color="red"
            )
        fig.colorbar(im, ax=ax, shrink=0.8, label="eigenscore")
        fig.tight_layout()
        return fig

    def _fig_scree(self, gesd: GesdResult):
        """Scree plot: variance explained per PC and cumulative."""
        import matplotlib.pyplot as plt

        ve = gesd.var_explained_all
        x = np.arange(1, len(ve) + 1)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(x, ve * 100, "o-", color="steelblue", label="per PC")
        ax.plot(x, np.cumsum(ve) * 100, "s--", color="gray", label="cumulative")
        ax.axvline(
            gesd.n_pcs + 0.5, color="red", ls=":", label=f"kept {gesd.n_pcs} PCs"
        )
        ax.set_xlabel("Principal component")
        ax.set_ylabel("Variance explained (%)")
        ax.set_title("PCA scree")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def _fig_score_corr(self, gesd: GesdResult):
        """Correlation matrix of the standardized scores."""
        import matplotlib.pyplot as plt

        if gesd.M_std.shape[0] < 2:
            return None
        corr = np.corrcoef(gesd.M_std)
        k = corr.shape[0]
        fig, ax = plt.subplots(
            figsize=(max(4, 0.4 * k + 2), max(4, 0.4 * k + 2))
        )
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(k))
        ax.set_xticklabels(gesd.score_names, rotation=90, fontsize=7)
        ax.set_yticks(range(k))
        ax.set_yticklabels(gesd.score_names, fontsize=7)
        ax.set_title("Score correlation (standardized)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="r")
        fig.tight_layout()
        return fig

    def _fig_standardized_scores(self, gesd: GesdResult):
        """Heatmap of the standardized scores (scores x ICs)."""
        import matplotlib.pyplot as plt

        data = gesd.M_std  # (k, n)
        k, n = data.shape
        fig, ax = plt.subplots(
            figsize=(max(5, 0.18 * n + 2), max(3, 0.35 * k + 1))
        )
        vmax = float(np.abs(data).max()) or 1.0
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(k))
        ax.set_yticklabels(gesd.score_names, fontsize=8)
        ax.set_xlabel("Independent component")
        ax.set_title("Standardized scores")
        fig.colorbar(im, ax=ax, shrink=0.8, label="z")
        fig.tight_layout()
        return fig

    def _fig_pc_outliers(self, gesd: GesdResult):
        """Per-PC scatter of eigenscore vs IC index; flagged ICs in red."""
        import matplotlib.pyplot as plt

        n_pcs = gesd.n_pcs
        ncol = min(3, n_pcs)
        nrow = int(np.ceil(n_pcs / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False
        )
        n = gesd.eigenscores.shape[1]
        x = np.arange(n)
        for p in range(n_pcs):
            ax = axes[p // ncol][p % ncol]
            y = gesd.eigenscores[p]
            flags = gesd.per_pc_flagged[p]
            ax.scatter(x[~flags], y[~flags], s=12, color="steelblue", label="kept")
            if flags.any():
                ax.scatter(x[flags], y[flags], s=24, color="red", label="flagged")
            ax.set_title(
                f"PC{p + 1} (α/PC={gesd.alpha_per_pc:.4f})", fontsize=9
            )
            ax.set_xlabel("IC")
            ax.set_ylabel("eigenscore")
            ax.legend(fontsize=7)
        for j in range(n_pcs, nrow * ncol):
            fig.delaxes(axes[j // ncol][j % ncol])
        fig.suptitle("Per-PC GESD outliers")
        fig.tight_layout()
        return fig

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
    AVAILABLE_GESD_METRICS = [
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
    AVAILABLE_TARGETED_SCORES = [
        "eog",
        "ecg",
        "reference",
        "corrmap_eog",
        "corrmap_ecg",
    ]

    # Full set of per-IC scores selectable via ``cfg._gesd_scores``.
    AVAILABLE_GESD_SCORES = AVAILABLE_GESD_METRICS + AVAILABLE_TARGETED_SCORES

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
            score_names = gesd.score_names

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
            for j, sname in enumerate(score_names):
                data[f"score_{sname}"] = np.round(gesd.M[j], 6).tolist()

            # PC loadings (shared across components, stored once per row
            # for self-contained CSV analysis)
            loading_strs = []
            for p in range(n_pcs):
                parts = [
                    f"{score_names[j]}:{gesd.loadings[j, p]:.3f}"
                    for j in range(len(score_names))
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
        """Transform metrics and specify outlier direction for GESD.

        Which metrics are included can be controlled by passing ``names``
        explicitly (used by the unified scoring layer) or, when ``names`` is
        ``None``, by setting ``cfg._gesd_metrics`` to a list of metric name
        strings.  When neither is given, all diagnostic metrics are used.

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
            Explicit metric selection.  Overrides ``cfg._gesd_metrics`` when
            provided.

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
        # Determine which metrics to compute.
        if names is not None:
            selected = list(names)
        else:
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
