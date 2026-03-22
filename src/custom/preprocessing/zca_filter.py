"""ZCA (Zero-phase Component Analysis) Filter for OPM-MEG data.

This module applies ZCA filtering to MEG data using a forward model-informed
approach. ZCA identifies signal and noise subspaces via generalized
eigendecomposition (GED) of the forward model and external SSS basis,
then projects out the noise components.

ZCA is particularly useful for OPM-MEG data because it leverages the
forward model to distinguish brain signals from environmental interference,
providing more targeted noise suppression than standard HFC.

Usage
-----
CLI:
    python src/custom/custom_preproc.py --analysis=zca_filter --config=/path/to/config.py

Programmatic:
    >>> from preprocessing.zca_filter import run
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
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR path.

Optional:
    _do_ZCA : bool
        Enable/disable ZCA/GEDAI filtering. Default: False.
    _zca_method : str
        Algorithm to use: 'zca' (default) or 'gedai'.
        - 'zca': Geometric approach — GED between the forward model signal
          subspace and the external SSS noise subspace. Purely geometry-based,
          does not use the actual data statistics in the decomposition.
        - 'gedai': Data-adaptive approach (Generalized Eigenvalue Decomposition
          for Artifact Identification). refCOV = U S² Uᵀ where U, S come from
          the noise-whitened, depth-weighted inverse operator. GED between the
          task data covariance (dataCOV) and refCOV + dataCOV identifies
          components whose variance maximally exceeds brain-predicted
          covariance as artifacts.
    _zca_ext_order : int
        Order of external SSS basis (1-3). Used by ZCA method only. Higher
        orders capture more complex external interference patterns. Default: 3.
    _zca_threshold : float
        GED eigenvalue threshold for ZCA signal/noise separation.
        Values closer to 1.0 retain more signal components. Default: 0.99.
    _gedai_threshold : float
        GED eigenvalue threshold for GEDAI artifact identification.
        If <= 1.0, interpreted as a fraction: components with eigenvalue
        above this value (data fraction of total) are artifacts.
        If > 1, interpreted as number of signal dimensions to keep.
        Default: 0.50.
    process_empty_room : bool
        Apply same projections to noise recording. Default: False.

Notes
-----
Both ZCA and GEDAI require:
- BEM solution (*bem-sol.fif) in FreeSurfer subject's bem/ folder
- Source space (*-src.fif) in FreeSurfer subject's bem/ folder
- Head-MRI transform (computed via mne_bids.get_head_mri_trans)
- Noise recording (task="noise") for noise covariance estimation

ZCA algorithm:
1. Computes forward model and inverse operator
2. Builds signal transform from forward model eigendecomposition
3. Builds noise transform from external SSS basis
4. Performs GED (noise vs signal+noise) to separate subspaces
5. Creates SSP projectors from noise subspace
6. Applies projectors to data

GEDAI algorithm:
1. Computes task data covariance (dataCOV) and noise covariance
2. Computes forward solution and inverse operator (depth-weighted, noise-whitened)
3. Builds brain reference covariance: refCOV = U S² Uᵀ from inverse operator
4. Regularizes both via Tikhonov (Cohen 2022 style)
5. Performs GED: eigh(dataCOV, refCOV + dataCOV) — high eigenvalue = artifact
6. Creates SSP projectors from artifact subspace
7. Applies projectors to data

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import mne
import mne_bids
import numpy as np
from mne._fiff.pick import pick_info
from mne.preprocessing.maxwell import _prep_mf_coils, _sss_basis
from scipy.linalg import eigh, null_space

from ._base import BaseAnalysis
from ._io import write_raw_bids_preserve_events, read_raw_bids_with_retry


class ZCAFilterAnalysis(BaseAnalysis):
    """Apply ZCA (Zero-phase Component Analysis) filtering.

    ZCA uses forward model information to distinguish brain signals from
    environmental interference, creating targeted SSP projectors for
    noise suppression.

    Attributes
    ----------
    ANALYSIS_KEY : str
        'zcafilter'
    ANALYSIS_NAME : str
        'zca_filter'

    See Also
    --------
    mne.preprocessing.compute_proj_hfc : Alternative HFC projection method.
    mne.minimum_norm.make_inverse_operator : Used for forward decomposition.
    """

    ANALYSIS_KEY = "zcafilter"
    ANALYSIS_NAME = "zca_filter"

    def is_enabled(self) -> bool:
        """Check if ZCA filtering is enabled.

        Returns
        -------
        enabled : bool
            True if cfg._do_ZCA is True.
        """
        return getattr(self.cfg, "_do_ZCA", False)

    def load_data(self) -> Dict[str, Any]:
        """Load raw data, noise, BEM, source space, and transform.

        Returns
        -------
        data : dict
            Dictionary containing:
            - 'raw': Raw MEG data for task
            - 'noise': Raw noise recording (required)
            - 'bem': BEM solution
            - 'src': Source space
            - 'trans': Head-MRI transform
        """
        self.log("Loading data...")
        data: Dict[str, Any] = {}

        # Get FreeSurfer subject name
        fs_subject = self._get_fs_subject()
        self.log(f"FreeSurfer subject: {fs_subject}")

        # Load main task data
        paths = mne_bids.find_matching_paths(
            root=self.cfg.bids_root,
            subjects=self.cfg.subjects,
            tasks=self.cfg.task,
            sessions=self.cfg.sessions,
            datatypes="meg",
            extensions=".fif",
            ignore_nosub=True,
        )
        if not paths:
            raise FileNotFoundError(f"No raw data found for task={self.cfg.task}")
        
        data["bids_path"] = paths[0]
        data["raw"] = read_raw_bids_with_retry(paths[0], extra_params={"preload": True})
        self.log(f"Loaded raw data for task={self.cfg.task}")

        # Load noise data (required for ZCA)
        paths_noise = mne_bids.find_matching_paths(
            root=self.cfg.bids_root,
            subjects=self.cfg.subjects,
            tasks="noise",
            sessions=self.cfg.sessions,
            datatypes="meg",
            extensions=".fif",
            ignore_nosub=True,
        )
        if not paths_noise:
            raise FileNotFoundError(
                "No noise recording found. ZCA requires task='noise' data "
                "for computing the noise covariance matrix."
            )
        data["noise"] = read_raw_bids_with_retry(paths_noise[0], extra_params={"preload": True})
        self.log("Loaded noise recording")

        # Load BEM solution
        bem_path = self._find_bem_solution(fs_subject)
        data["bem"] = mne.read_bem_solution(bem_path)
        self.log(f"Loaded BEM solution: {bem_path.name}")

        # Load source space
        src_path = self._find_source_space(fs_subject)
        data["src"] = mne.read_source_spaces(src_path)
        self.log(f"Loaded source space: {src_path.name}")

        # Load head-MRI transform
        data["trans"] = mne_bids.get_head_mri_trans(
            bids_path=paths[0],
            fs_subjects_dir=self.cfg.subjects_dir,
            fs_subject=fs_subject,
        )
        self.log("Loaded head-MRI transform")

        return data

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute and apply ZCA projections.

        Parameters
        ----------
        data : dict
            Dictionary from load_data() with raw, noise, bem, src, trans.

        Returns
        -------
        results : dict
            Dictionary with ZCA-filtered raw data per task.
        """
        results: Dict[str, Any] = {}

        # Extract data
        raw = data["raw"]
        noise = data["noise"]
        bem = data["bem"]
        src = data["src"]
        trans = data["trans"]

        # Dispatch to ZCA or GEDAI based on config
        method = getattr(self.cfg, "_zca_method", "zca").lower()
        if method == "gedai":
            raw, noise, projs = self._apply_gedai(raw, noise, bem, src, trans)
        else:
            raw, noise, projs = self._apply_zca(raw, noise, bem, src, trans)

        results[self.cfg.task] = raw
        results["noise"] = noise
        results["projs"] = projs

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save ZCA-filtered data back to BIDS structure.

        Parameters
        ----------
        results : dict
            Dictionary with ZCA-filtered raw data per task.
        """
        self.log("Saving results...")

        # Separate task data from metadata
        tasks = {k: v for k, v in results.items() if k not in {"projs"}}

        # Find empty room path for association
        er_bids_path = None
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

    def _get_fs_subject(self) -> str:
        """Get FreeSurfer subject name.

        Returns
        -------
        fs_subject : str
            FreeSurfer subject name in format 'sub-{subject}_ses-{session}'.
        """
        subject = self.cfg.subjects[0]
        session = self.cfg.sessions[0] if self.cfg.sessions else None

        if session:
            return f"sub-{subject}_ses-{session}"
        else:
            return f"sub-{subject}"

    def _find_bem_solution(self, fs_subject: str) -> Path:
        """Find BEM solution file in FreeSurfer subject's bem folder.

        Parameters
        ----------
        fs_subject : str
            FreeSurfer subject name.

        Returns
        -------
        bem_path : Path
            Path to BEM solution file.

        Raises
        ------
        FileNotFoundError
            If no BEM solution is found.
        """
        bem_dir = Path(self.cfg.subjects_dir) / fs_subject / "bem"

        if not bem_dir.exists():
            raise FileNotFoundError(
                f"BEM directory not found: {bem_dir}\n"
                f"Run FreeSurfer and BEM computation first."
            )

        # Glob for BEM solution files
        bem_files = list(bem_dir.glob("*bem-sol.fif"))

        if not bem_files:
            raise FileNotFoundError(
                f"No BEM solution (*bem-sol.fif) found in {bem_dir}\n"
                f"Run mne_bids_pipeline source/make_bem_solution step first."
            )

        # If multiple, prefer the one with subject name prefix
        for bem_file in bem_files:
            if bem_file.name.startswith(fs_subject):
                return bem_file

        # Otherwise return first match
        return bem_files[0]

    def _find_source_space(self, fs_subject: str) -> Path:
        """Find source space file in FreeSurfer subject's bem folder.

        Parameters
        ----------
        fs_subject : str
            FreeSurfer subject name.

        Returns
        -------
        src_path : Path
            Path to source space file.

        Raises
        ------
        FileNotFoundError
            If no source space is found.
        """
        bem_dir = Path(self.cfg.subjects_dir) / fs_subject / "bem"

        if not bem_dir.exists():
            raise FileNotFoundError(
                f"BEM directory not found: {bem_dir}\n"
                f"Run FreeSurfer and source space computation first."
            )

        # Glob for source space files
        src_files = list(bem_dir.glob("*-src.fif"))

        if not src_files:
            raise FileNotFoundError(
                f"No source space (*-src.fif) found in {bem_dir}\n"
                f"Run mne_bids_pipeline source/setup_source_space step first."
            )

        # If multiple, prefer the one with subject name prefix
        for src_file in src_files:
            if src_file.name.startswith(fs_subject):
                return src_file

        # Otherwise return first match
        return src_files[0]

    def _apply_zca(
        self,
        raw: mne.io.BaseRaw,
        noise: mne.io.BaseRaw,
        bem: mne.bem.ConductorModel,
        src: mne.SourceSpaces,
        trans: mne.transforms.Transform,
    ) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, list]:
        """Compute and apply ZCA projections.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw MEG data to apply ZCA to.
        noise : mne.io.BaseRaw
            Noise recording for covariance estimation.
        bem : mne.bem.ConductorModel
            BEM solution.
        src : mne.SourceSpaces
            Source space.
        trans : mne.transforms.Transform
            Head-MRI transform.

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw data with ZCA projections applied.
        noise : mne.io.BaseRaw
            Noise data with ZCA projections applied.
        projs : list
            List of ZCA projectors.
        """
        ext_order = getattr(self.cfg, "_zca_ext_order", 3)
        threshold = getattr(self.cfg, "_zca_threshold", 0.50)
        cov_reg = 1e-8  # Regularization for numerical stability

        self.log(f"Computing ZCA filter (ext_order={ext_order}, threshold={threshold})")

        info = raw.info

        # Step 1: Compute covariances
        self.log("Computing data covariance...")
        data_cov = mne.compute_raw_covariance(
            raw, method="shrunk", rank="info", n_jobs=self.cfg.n_jobs
        )

        self.log("Computing noise covariance...")
        noise_cov = mne.compute_raw_covariance(
            noise, method="shrunk", rank="info", n_jobs=self.cfg.n_jobs
        )

        # Step 2: Compute forward solution
        self.log("Computing forward solution...")
        fwd = mne.make_forward_solution(
            info=info,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=self.cfg.n_jobs,
        )

        # Step 3: Compute inverse operator (for eigendecomposition)
        self.log("Computing inverse operator...")
        inv = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            noise_cov,
            loose="auto",
            depth=0.8,
            fixed="auto",
            rank="info",
            use_cps=True,
        )

        fwd_field = inv["eigen_fields"]["data"]
        fwd_sing = inv["sing"]

        # Step 4: Compute external SSS basis
        self.log(f"Computing external SSS basis (order={ext_order})...")
        sss_info = pick_info(
            info, mne.pick_types(info, meg=True, eeg=False, exclude="bads")
        )
        exp = dict(origin=(0.0, 0.0, 0.0), int_order=0, ext_order=ext_order)
        coils = _prep_mf_coils(sss_info, ignore_ref=True, accuracy="accurate")

        ext_basis = _sss_basis(exp, coils)
        ext_basis /= (np.linalg.norm(ext_basis, axis=0) ** 0.8)

        # Step 5: Build signal and noise transforms
        self.log("Building signal and noise transforms...")

        signal_trans = fwd_field @ np.diag(fwd_sing**2) @ fwd_field.T
        signal_cov_mat = signal_trans @ data_cov["data"] @ signal_trans.T
        noise_trans = ext_basis @ ext_basis.T
        noise_cov_mat = noise_trans @ data_cov["data"] @ noise_trans.T

        # signal_cov_mat = fwd_field @ np.diag(fwd_sing**2) @ fwd_field.T
        # noise_cov_mat = ext_basis @ ext_basis.T

        # Step 6: Compute GED
        self.log("Computing generalized eigendecomposition...")
        

        # Symmetrize for numerical stability
        signal_cov_mat = (signal_cov_mat + signal_cov_mat.T) / 2
        signal_cov_mat = (
            cov_reg * np.eye(signal_cov_mat.shape[0])
            * (np.trace(signal_cov_mat) / signal_cov_mat.shape[0])
            + (1 - cov_reg) * signal_cov_mat
        )

        noise_cov_mat = (noise_cov_mat + noise_cov_mat.T) / 2
        noise_cov_mat = (
            cov_reg * np.eye(noise_cov_mat.shape[0])
            * (np.trace(noise_cov_mat) / noise_cov_mat.shape[0])
            + (1 - cov_reg) * noise_cov_mat
        )

        denom = signal_cov_mat + noise_cov_mat
        denom = (denom + denom.T) / 2

        eigenvalues_noise, eigenvectors_noise = eigh(noise_cov_mat, denom)

        # Select noise components (high eigenvalues = most noise-like)
        if threshold <= 1.0:
            # threshold on signal → noise eigenvalue cutoff is (1 - threshold)
            noise_mask = eigenvalues_noise > threshold
            U_noise = eigenvectors_noise[:, noise_mask]
            U_signal = eigenvectors_noise[:, ~noise_mask]
        else:
            n_signal = int(threshold)
            n_noise = eigenvectors_noise.shape[1] - n_signal
            idx = np.argsort(eigenvalues_noise)[::-1][:n_noise]
            U_noise = eigenvectors_noise[:, idx]
            U_signal = eigenvectors_noise[:, ~np.isin(np.arange(eigenvectors_noise.shape[1]), idx)]

        n_channels = U_signal.shape[0]
        n_signal = U_signal.shape[1]
        n_noise =  U_noise.shape[1]

        self.log(f"Signal subspace: {n_signal} dimensions")
        self.log(f"Noise subspace: {n_noise} dimensions")

        if n_noise <= 0:
            self.log("WARNING: No noise components to remove. Skipping ZCA.")
            return raw, noise, []

        # Step 8: Compute noise subspace (orthogonal complement)
        self.log(f"Created {U_noise.shape[1]} noise projectors")

        # Step 9: Create SSP projectors
        desc_prefix = f"ZCA_ext{ext_order}_thresh{threshold:.2f}"
        ch_names = [
            ch for ch in sss_info["ch_names"]
            if ch in raw.ch_names
        ]

        projs = []
        for k in range(U_noise.shape[1]):
            proj_data = np.zeros(len(raw.ch_names))
            for i, ch_name in enumerate(ch_names):
                if ch_name in raw.ch_names:
                    idx = raw.ch_names.index(ch_name)
                    proj_data[idx] = U_noise[i, k]

            # Normalize
            norm = np.linalg.norm(proj_data)
            if norm > 0:
                proj_data /= norm

            proj = mne.Projection(
                data=dict(
                    data=proj_data.reshape(1, -1),
                    col_names=raw.ch_names,
                    row_names=None,
                    nrow=1,
                    ncol=len(raw.ch_names),
                ),
                desc=f"{desc_prefix}_{k + 1:02d}",
                kind=1,  # MEG projection
                active=False,
            )
            projs.append(proj)

        # Step 10: Apply projectors
        self.log(f"Applying {len(projs)} ZCA projections to task data...")
        raw.add_proj(projs=projs).apply_proj()

        self.log("Applying ZCA projections to noise data...")
        noise.add_proj(projs=projs).apply_proj()

        self.log("ZCA filtering complete!")

        return raw, noise, projs

    def _apply_gedai(
        self,
        raw: mne.io.BaseRaw,
        noise: mne.io.BaseRaw,
        bem: mne.bem.ConductorModel,
        src: mne.SourceSpaces,
        trans: mne.transforms.Transform,
    ) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, list]:
        """Compute and apply GEDAI projections.

        GEDAI (Generalized Eigenvalue Decomposition for Artifact
        Identification) builds a brain reference covariance (refCOV) from
        the noise-whitened, depth-weighted forward model via the inverse
        operator:

            G_wd = C_n^{-1/2} G D   (whitened + depth-weighted gain)
            G_wd = U S V^T           (SVD)
            refCOV = U S^2 U^T       (sensor-space Gramian)

        Depth weighting (default 0.8) corrects the leadfield's bias toward
        superficial cortex; noise whitening normalizes sensor noise patterns.

        The task data covariance (dataCOV) is then decomposed relative to the
        sum (refCOV + dataCOV) via GEVD:

            eigh(dataCOV, refCOV_reg + dataCOV_reg)

        Eigenvalues are in [0, 1] and represent the fraction of total energy
        attributable to observed data variance. Components with eigenvalue
        above the threshold (default 0.5, i.e., data exceeds brain reference)
        are treated as artifacts. Regularization follows Cohen (2022):
        Tikhonov regularization scaled by the trace of each matrix.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Task MEG data (source of dataCOV).
        noise : mne.io.BaseRaw
            Noise recording (used for refCOV Tikhonov regularization scale).
        bem : mne.bem.ConductorModel
            BEM solution.
        src : mne.SourceSpaces
            Source space.
        trans : mne.transforms.Transform
            Head-MRI transform.

        Returns
        -------
        raw : mne.io.BaseRaw
            Raw data with GEDAI projections applied.
        noise : mne.io.BaseRaw
            Noise data with GEDAI projections applied.
        projs : list
            List of GEDAI projectors.
        """
        threshold = getattr(self.cfg, "_gedai_threshold", 0.50)
        cov_reg = 1e-8  # Tikhonov regularization (Cohen 2022 style)

        self.log(f"Computing GEDAI filter (threshold={threshold})")

        info = raw.info

        # Step 1: Compute task data covariance (dataCOV)
        self.log("Computing task data covariance (dataCOV)...")
        data_cov = mne.compute_raw_covariance(
            raw, method="shrunk", rank="info", n_jobs=self.cfg.n_jobs
        )

        # Step 2: Compute noise covariance (used to scale refCOV regularization)
        self.log("Computing noise covariance...")
        noise_cov = mne.compute_raw_covariance(
            noise, method="shrunk", rank="info", n_jobs=self.cfg.n_jobs
        )

        # Step 3: Compute forward solution
        self.log("Computing forward solution...")
        fwd = mne.make_forward_solution(
            info=info,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=self.cfg.n_jobs,
        )

        # Step 4: Compute inverse operator (for whitened, depth-weighted Gramian)
        self.log("Computing inverse operator...")
        inv = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            noise_cov,
            loose="auto",
            depth=0.8,
            fixed="auto",
            rank="info",
            use_cps=True,
        )

        



        # Step 5: Build refCOV = U S² Uᵀ (whitened + depth-weighted Gramian)
        # This is G_wd @ G_wd.T where G_wd is the noise-whitened,
        # depth-weighted gain matrix from the inverse operator.
        self.log("Building refCOV from whitened/depth-weighted Gramian...")

        fwd_field = inv["eigen_fields"]["data"]
        fwd_sing = inv["sing"]
        refCOV = fwd_field @ np.diag(fwd_sing**2) @ fwd_field.T
        refCOV = refCOV @ data_cov["data"] @ refCOV.T

        # L = fwd["sol"]["data"]                  # [n_ch × n_sources]
        # refCOV = L @ L.T

        # Get MEG channel names aligned to eigenvectors
        meg_picks = mne.pick_types(info, meg=True, eeg=False, exclude="bads")
        meg_ch_names = [info["ch_names"][p] for p in meg_picks]

        # Step 6: Extract MEG-channel submatrix of dataCOV aligned to refCOV
        cov_ch_names = list(data_cov["names"])
        shared_ch_names = [ch for ch in meg_ch_names if ch in cov_ch_names]
        cov_idx = [cov_ch_names.index(ch) for ch in shared_ch_names]
        dataCOV = data_cov["data"][np.ix_(cov_idx, cov_idx)]

        # Trim refCOV rows/cols to the shared channel set if necessary
        if len(shared_ch_names) < len(meg_ch_names):
            meg_idx = [meg_ch_names.index(ch) for ch in shared_ch_names]
            refCOV = refCOV[np.ix_(meg_idx, meg_idx)]
            self.log(
                f"Channel mismatch: using {len(shared_ch_names)}/{len(meg_ch_names)} "
                "channels shared between refCOV and dataCOV"
            )

        # Step 7: Regularize (Tikhonov, Cohen 2022 style)
        refCOV = (refCOV + refCOV.T) / 2
        refCOV_reg = (
            (1 - cov_reg) * refCOV
            + cov_reg * (np.trace(refCOV) / refCOV.shape[0]) * np.eye(refCOV.shape[0])
        )

        dataCOV = (dataCOV + dataCOV.T) / 2
        dataCOV_reg = (
            (1 - cov_reg) * dataCOV
            + cov_reg * (np.trace(dataCOV) / dataCOV.shape[0]) * np.eye(dataCOV.shape[0])
        )

        # Step 7: GEDAI generalized eigendecomposition
        # eigh(dataCOV, refCOV + dataCOV): eigenvalues in [0, 1]
        # high eigenvalue = data energy dominates → artifact
        self.log("Computing GEDAI generalized eigendecomposition...")
        denom = refCOV_reg + dataCOV_reg
        denom = (denom + denom.T) / 2
        eigenvalues, eigenvectors = eigh(dataCOV_reg, denom)

        self.log(
            f"Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]"
        )

        # Step 8: Select artifact components (high eigenvalue = artifact)
        if threshold <= 1.0:
            # Fractional threshold: eigenvalue > threshold → artifact
            artifact_mask = eigenvalues > threshold
            U_artifact = eigenvectors[:, artifact_mask]
            U_signal = eigenvectors[:, ~artifact_mask]
        else:
            # Integer threshold: keep this many signal dimensions
            n_keep = int(threshold)
            n_remove = eigenvectors.shape[1] - n_keep
            idx = np.argsort(eigenvalues)[::-1][:n_remove]
            artifact_mask = np.isin(np.arange(eigenvectors.shape[1]), idx)
            U_artifact = eigenvectors[:, artifact_mask]
            U_signal = eigenvectors[:, ~artifact_mask]

        n_artifacts = U_artifact.shape[1]
        n_signal = U_signal.shape[1]

        self.log(f"Signal components: {n_signal}")
        self.log(f"Artifact components: {n_artifacts}")

        if n_artifacts <= 0:
            self.log("WARNING: No artifact components found. Skipping GEDAI.")
            return raw, noise, []

        # Step 9: Create SSP projectors from artifact eigenvectors
        desc_prefix = f"GEDAI_thresh{threshold:.2f}"

        projs = []
        for k in range(n_artifacts):
            proj_data = np.zeros(len(raw.ch_names))
            for i, ch_name in enumerate(shared_ch_names):
                if ch_name in raw.ch_names:
                    idx = raw.ch_names.index(ch_name)
                    proj_data[idx] = U_artifact[i, k]

            norm = np.linalg.norm(proj_data)
            if norm > 0:
                proj_data /= norm

            proj = mne.Projection(
                data=dict(
                    data=proj_data.reshape(1, -1),
                    col_names=raw.ch_names,
                    row_names=None,
                    nrow=1,
                    ncol=len(raw.ch_names),
                ),
                desc=f"{desc_prefix}_{k + 1:02d}",
                kind=1,  # MEG projection
                active=False,
            )
            projs.append(proj)

        # Step 10: Apply projectors to task and noise data
        self.log(f"Applying {len(projs)} GEDAI projections to task data...")
        raw.add_proj(projs=projs).apply_proj()

        self.log("Applying GEDAI projections to noise data...")
        noise.add_proj(projs=projs).apply_proj()

        self.log("GEDAI filtering complete!")

        return raw, noise, projs


def run(cfg: SimpleNamespace) -> None:
    """Module entry point for CLI dispatcher.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration object.
    """
    analysis = ZCAFilterAnalysis(cfg)

    if not analysis.is_enabled():
        print(f"\n[{analysis.ANALYSIS_NAME}] Disabled in configuration; exiting")
        return

    analysis.execute()
