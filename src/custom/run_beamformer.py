"""LCMV Beamformer source reconstruction for OPM-MEG data.

This script performs LCMV (Linearly Constrained Minimum Variance) beamformer
analysis on preprocessed MEG data. It supports two types of analyses:

1. Time-locked analysis: Apply beamformer to evoked responses (PRIMARY)
2. Power analysis: Apply beamformer to covariance matrices (SECONDARY)

The beamformer provides spatial filtering to reconstruct source activity,
particularly well-suited for OPM-MEG data.

Usage:
    python run_beamformer.py --config=/path/to/config.py --output-type=both

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import mne
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv
from mne_bids import BIDSPath, get_head_mri_trans

# Add mne-bids-pipeline to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mne-bids-pipeline"))

from mne_bids_pipeline._config_import import _update_config_from_path, _import_config
from mne_bids_pipeline._config_utils import (
    _get_bem_conductivity,
    get_fs_subject,
    get_fs_subjects_dir,
    get_noise_cov_bids_path,
    sanitize_cond_name,
)
from mne_bids_pipeline._report import _all_conditions, _open_report, _sanitize_cond_tag


# --------------------------------------------------------------------------------------
# Configuration and Environment
# --------------------------------------------------------------------------------------


def load_config(config_path: str) -> SimpleNamespace:
    """Load configuration from file and environment variables.

    Parameters
    ----------
    config_path : str
        Path to the configuration Python file.

    Returns
    -------
    cfg : SimpleNamespace
        Configuration object with all settings.
    """
    print(f"\n[load_config] Loading configuration from: {config_path}")

    # Load config file (matching custom_preproc.py pattern)
    # config = SimpleNamespace()
    # _update_config_from_path(config=config, config_path=config_path)

    cfg = SimpleNamespace()
    _update_config_from_path(config=cfg, config_path=config_path)

    # Extract environment variables (matching custom_preproc.py pattern)
    subject = os.environ.get("SUBJECT", cfg.subjects[0])
    session = os.environ.get("SESSION", "01")

    print(f"[load_config] Subject: {cfg.subjects[0]}, Session: {cfg.sessions[0]}")
    print(f"[load_config] Task: {cfg.task}")
    print(f"[load_config] Beamformer enabled: {cfg._run_beamformer}")

    return cfg


def resolve_source_spaces(cfg: SimpleNamespace) -> list[str]:
    """Return the beamformer source spaces to run, as an ordered unique list.

    ``_beamformer_source_space`` may be a single string (``"surface"`` |
    ``"volume"``) or a list/tuple of them (e.g. ``["volume", "surface"]``) to run
    both reconstructions in one invocation.  Duplicates are collapsed and the
    given order is preserved.
    """
    val = getattr(cfg, "_beamformer_source_space", "surface")
    spaces = [val] if isinstance(val, str) else list(val)
    resolved: list[str] = []
    for s in spaces:
        if s not in ("surface", "volume"):
            raise ValueError(
                f"Invalid _beamformer_source_space entry: {s!r}. "
                f"Must be 'surface' or 'volume'."
            )
        if s not in resolved:
            resolved.append(s)
    if not resolved:
        raise ValueError("_beamformer_source_space must not be empty.")
    return resolved


# --------------------------------------------------------------------------------------
# Volume Forward Solution
# --------------------------------------------------------------------------------------


def _find_bem_solution(fs_subjects_dir: str, fs_subject: str, tag: str) -> Path | None:
    """Locate an existing BEM solution in the FreeSurfer subject ``bem/`` directory.

    Prefers the conductivity-tagged file the pipeline writes
    (``{fs_subject}-{tag}-bem-sol.fif``), then falls back to any ``*bem-sol.fif``.
    Mirrors ``coreg_diagnostics._find_bem_solution`` but honours the pipeline's
    ``_get_bem_conductivity`` tag so the OPM single-layer solution is found first.
    """
    bem_dir = Path(fs_subjects_dir) / fs_subject / "bem"
    if not bem_dir.exists():
        return None
    tagged = bem_dir / f"{fs_subject}-{tag}-bem-sol.fif"
    if tagged.exists():
        return tagged
    matches = sorted(bem_dir.glob("*bem-sol.fif"))
    if not matches:
        return None
    for m in matches:
        if m.name.startswith(fs_subject):
            return m
    return matches[0]


def build_volume_forward(cfg: SimpleNamespace, info: mne.Info) -> mne.Forward:
    """Build (or load a cached) volume-source-space forward solution.

    mne-bids-pipeline only ever builds *surface* forward solutions, so a volume
    beamformer needs its own forward.  This adapts the on-the-fly forward pattern
    from ``coreg_diagnostics._compute_forward``, swapping ``setup_source_space``
    for :func:`mne.setup_volume_source_space` (a regular 3D grid bounded by the
    BEM inner-skull surface).

    The result is cached to ``*_acq-vol_fwd.fif`` (the forward file embeds its
    volume ``src``) and reused on subsequent runs when
    ``_beamformer_volume_cache`` is set.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.  Reads ``_beamformer_volume_pos`` (grid spacing,
        mm), ``_beamformer_volume_mindist`` (mm from inner skull),
        ``_beamformer_volume_bem_conductivity``, ``_beamformer_volume_bem_ico``
        and ``_beamformer_volume_cache``.
    info : mne.Info
        Measurement info (sensor geometry) the forward is computed for.

    Returns
    -------
    fwd : mne.Forward
        Volume-source-space forward solution.
    """
    print("\n[build_volume_forward] Building volume-source-space forward...")

    subject = cfg.subjects[0]
    session = cfg.sessions[0]

    fs_subject = get_fs_subject(config=cfg, subject=subject, session=session)
    fs_subjects_dir = get_fs_subjects_dir(config=cfg)
    print(
        f"[build_volume_forward] FreeSurfer subject={fs_subject}, "
        f"subjects_dir={fs_subjects_dir}"
    )

    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    pos = float(getattr(cfg, "_beamformer_volume_pos", 5.0))
    mindist = float(getattr(cfg, "_beamformer_volume_mindist", getattr(cfg, "mindist", 5.0)))
    cache = getattr(cfg, "_beamformer_volume_cache", True)

    # Cache -----------------------------------------------------------------
    vol_fwd_path = bids_path.copy().update(
        suffix="fwd", acquisition="vol", extension=".fif"
    )
    if cache and vol_fwd_path.fpath.exists():
        print(f"[build_volume_forward] Loading cached volume forward: {vol_fwd_path.fpath}")
        return mne.read_forward_solution(vol_fwd_path.fpath)

    # BEM solution ----------------------------------------------------------
    # _get_bem_conductivity reads per-step fields (fs_subject / use_template_mri
    # / ch_types) that the raw imported config does not carry; supply them via a
    # lightweight shim so we still reuse the pipeline's conductivity-tag
    # convention (rather than re-implementing it) without mutating cfg.
    bem_cfg = SimpleNamespace(
        fs_subject=fs_subject,
        use_template_mri=getattr(cfg, "use_template_mri", None),
        ch_types=cfg.ch_types,
    )
    conductivity_default, tag = _get_bem_conductivity(bem_cfg)
    bem_path = _find_bem_solution(fs_subjects_dir, fs_subject, tag)
    if bem_path is not None:
        print(f"[build_volume_forward] Loading BEM solution: {bem_path}")
        bem = mne.read_bem_solution(bem_path)
    else:
        conductivity = tuple(
            getattr(cfg, "_beamformer_volume_bem_conductivity", (0.3,))
        )
        ico = int(getattr(cfg, "_beamformer_volume_bem_ico", 4))
        print(
            f"[build_volume_forward] No BEM on disk; building model "
            f"(conductivity={conductivity}, ico={ico})"
        )
        model = mne.make_bem_model(
            subject=fs_subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=fs_subjects_dir,
        )
        bem = mne.make_bem_solution(model)

    # Head <-> MRI transform ------------------------------------------------
    # Prefer the trans the pipeline already wrote next to the surface forward;
    # fall back to recomputing it from the BIDS anatomical landmarks.
    trans = None
    trans_path = bids_path.copy().update(suffix="trans", extension=".fif")
    if trans_path.fpath.exists():
        print(f"[build_volume_forward] Loading head-MRI trans: {trans_path.fpath}")
        trans = mne.read_trans(trans_path.fpath)
    else:
        print("[build_volume_forward] No trans on disk; deriving from BIDS landmarks")
        t1_bids_path = BIDSPath(
            subject=subject,
            session=session,
            root=cfg.bids_root,
            datatype="anat",
            suffix="T1w",
            extension=".nii.gz",
            check=False,
        )
        trans = get_head_mri_trans(
            bids_path,
            fs_subject=fs_subject,
            fs_subjects_dir=fs_subjects_dir,
            t1_bids_path=t1_bids_path,
        )

    # Volume source space (grid bounded by the BEM inner skull) -------------
    print(f"[build_volume_forward] Setting up volume source space (pos={pos} mm)")
    src = mne.setup_volume_source_space(
        subject=fs_subject,
        pos=pos,
        bem=bem,
        subjects_dir=fs_subjects_dir,
        add_interpolator=True,
        n_jobs=getattr(cfg, "n_jobs", -1),
    )
    print(f"[build_volume_forward] Volume source space: {sum(s['nuse'] for s in src)} sources")

    # Forward solution ------------------------------------------------------
    print("[build_volume_forward] Computing forward solution...")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        ignore_ref=True,
        mindist=mindist,
        n_jobs=getattr(cfg, "n_jobs", -1),
    )

    # Persist for reuse -----------------------------------------------------
    if cache:
        try:
            vol_fwd_path.fpath.parent.mkdir(parents=True, exist_ok=True)
            mne.write_forward_solution(vol_fwd_path.fpath, fwd, overwrite=True)
            print(f"[build_volume_forward] Cached volume forward to {vol_fwd_path.fpath}")
        except Exception as e:
            print(f"[build_volume_forward] WARNING: could not cache forward: {e}")

    return fwd


# --------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------


def load_beamformer_data(cfg: SimpleNamespace) -> Dict[str, Any]:
    """Load all required input files for beamformer analysis.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'forward': mne.Forward
        - 'epochs': mne.Epochs
        - 'noise_cov': mne.Covariance or None
        - 'info': mne.Info
    """
    print("\n[load_beamformer_data] Loading data files...")

    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    data = {}

    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    # Source space(s) to reconstruct: 'surface' (mne-bids-pipeline forward),
    # 'volume' (built on the fly from the measurement info), or a list of both.
    source_spaces = resolve_source_spaces(cfg)
    print(f"[load_beamformer_data] Source space(s): {source_spaces}")
    data["forwards"] = {}

    # Load surface forward up front so a missing one errors clearly (volume
    # forwards are built after epochs/info, which they require).
    if "surface" in source_spaces:
        fwd_path = bids_path.copy().update(suffix="fwd", extension=".fif")
        if not fwd_path.fpath.exists():
            raise FileNotFoundError(
                f"Forward solution not found at {fwd_path.fpath}\n"
                f"Run forward modeling first with:\n"
                f"  mne_bids_pipeline --steps=source/make_forward --config=<config>"
            )
        print(f"[load_beamformer_data] Loading forward solution: {fwd_path.fpath}")
        data["forwards"]["surface"] = mne.read_forward_solution(fwd_path)

    # Load clean epochs
    epochs_path = bids_path.copy().update(
        suffix="epo", processing="clean", extension=".fif"
    )
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(
            f"Clean epochs not found at {epochs_path.fpath}\nRun preprocessing first."
        )
    print(f"[load_beamformer_data] Loading epochs: {epochs_path.fpath}")
    data["epochs"] = mne.read_epochs(epochs_path, preload=True)
    data["info"] = mne.io.read_info(epochs_path)

    # Build (or load cached) volume forward from the measurement info.
    if "volume" in source_spaces:
        data["forwards"]["volume"] = build_volume_forward(cfg, data["info"])

    # Back-compat: expose the first requested space's forward as data["forward"].
    data["forward"] = data["forwards"][source_spaces[0]]

    # Load noise covariance
    if cfg.noise_cov == "ad-hoc":
        print("[load_beamformer_data] Using ad-hoc noise covariance")
        data["noise_path"] = None
    else:
        noise_path = bids_path.copy().update(
            task="noise", processing="clean", suffix="raw", extension=".fif"
        )
        print(f"[load_beamformer_data] Loading noise data: {noise_path.fpath}")
        data["noise_path"] = noise_path

    print(f"[load_beamformer_data] Data loading complete")
    print(f"  - Forward: {len(data['forward']['src'])} source spaces")
    print(
        f"  - Epochs: {len(data['epochs'])} epochs, {len(data['epochs'].ch_names)} channels"
    )
    print(
        f"  - Noise cov: {'ad-hoc' if data['noise_path'] is None else 'loaded from file'}"
    )

    return data


# --------------------------------------------------------------------------------------
# Beamformer Computation
# --------------------------------------------------------------------------------------


def compute_lcmv_filters(
    forward: mne.Forward,
    data_cov: mne.Covariance,
    noise_cov: mne.Covariance | None,
    rank: int | str,
    info: mne.Info,
    cfg: SimpleNamespace,
) -> dict:
    """Compute LCMV spatial filters.

    Parameters
    ----------
    forward : mne.Forward
        Forward solution.
    data_cov : mne.Covariance
        Data covariance matrix.
    noise_cov : mne.Covariance or None
        Noise covariance matrix. If None, uses ad-hoc.
    rank : int or str
        Rank of the covariance matrix (int or 'info' for MNE default).
    info : mne.Info
        Measurement info.
    cfg : SimpleNamespace
        Configuration with beamformer parameters.

    Returns
    -------
    filters : dict
        LCMV filters object.
    """
    print("\n[compute_lcmv_filters] Computing LCMV spatial filters...")
    print(f"  - Regularization: {cfg._beamformer_reg}")
    print(f"  - Pick orientation: {cfg._beamformer_pick_ori}")
    print(f"  - Weight normalization: {cfg._beamformer_weight_norm}")
    print(f"  - Depth weighting: {cfg._beamformer_depth}")
    print(f"  - Rank: {rank}")

    # Validate parameters
    valid_ori = ["max-power", "vector", None]
    if cfg._beamformer_pick_ori not in valid_ori:
        raise ValueError(
            f"Invalid _beamformer_pick_ori: {cfg._beamformer_pick_ori}. "
            f"Must be one of {valid_ori}"
        )

    valid_norm = ["unit-noise-gain", "nai", "unit-noise-gain-invariant", None]
    if cfg._beamformer_weight_norm not in valid_norm:
        raise ValueError(
            f"Invalid _beamformer_weight_norm: {cfg._beamformer_weight_norm}. "
            f"Must be one of {valid_norm}"
        )

    # Warn about suboptimal combinations
    if (
        cfg._beamformer_pick_ori == "vector"
        and cfg._beamformer_weight_norm == "unit-noise-gain"
    ):
        print("  [WARNING] Using 'unit-noise-gain' with vector beamformer.")
        print("  Consider using 'unit-noise-gain-invariant' instead.")

    # Use ad-hoc noise covariance if none provided
    if noise_cov is None:
        print("  [INFO] Creating ad-hoc noise covariance")
        noise_cov = mne.make_ad_hoc_cov(info)

    # Compute filters
    filters = make_lcmv(
        info,
        forward,
        data_cov,
        reg=cfg._beamformer_reg,
        noise_cov=noise_cov,
        pick_ori=cfg._beamformer_pick_ori,
        weight_norm=cfg._beamformer_weight_norm,
        depth=cfg._beamformer_depth,
        rank=rank,
        reduce_rank=cfg._reduce_rank,  # Always reduce rank for stability
        verbose=True,
    )

    print("[compute_lcmv_filters] Filters computed successfully")

    return filters


def run_beamformer_timecourse(
    epochs: mne.Epochs,
    filters: dict,
    cfg: SimpleNamespace,
) -> Dict[str, mne.SourceEstimate]:
    """Apply beamformer to evoked responses (time-locked analysis).

    This is the PRIMARY beamformer analysis, following the same pattern
    as the MNE inverse solution in _05_make_inverse.py.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    filters : dict
        LCMV filters from make_lcmv.
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    stcs : dict
        Dictionary mapping condition names to source estimates.
    """
    print("\n[run_beamformer_timecourse] Running time-locked beamformer analysis...")

    stcs = {}
    conditions = _all_conditions(cfg=cfg)

    print(f"[run_beamformer_timecourse] Processing {len(conditions)} conditions")

    for condition in conditions:
        print(f"  - Processing condition: {condition}")

        # Check if this is a contrast
        is_contrast = condition not in cfg.conditions

        if is_contrast:
            # Find the contrast definition
            contrast_def = None
            for contrast in cfg.contrasts:
                if contrast["name"] == condition:
                    contrast_def = contrast
                    break

            if contrast_def is None:
                print(
                    f"    [WARNING] Could not find contrast definition for '{condition}'. Skipping."
                )
                continue

            # Average epochs for each condition in the contrast
            evokeds = []
            for cond_name in contrast_def["conditions"]:
                try:
                    epochs_subset = epochs[cond_name].copy()
                    if len(epochs_subset) == 0:
                        print(
                            f"    [WARNING] No epochs for condition '{cond_name}'. Skipping contrast."
                        )
                        continue
                    evokeds.append(epochs_subset.average())
                except KeyError:
                    print(
                        f"    [WARNING] Condition '{cond_name}' not found in epochs. Skipping contrast."
                    )
                    continue

            if len(evokeds) != len(contrast_def["conditions"]):
                print(
                    f"    [WARNING] Could not load all conditions for contrast. Skipping."
                )
                continue

            # Combine evoked responses with weights
            evoked = mne.combine_evoked(evokeds, weights=contrast_def["weights"])
            print(f"    - Created contrast from {len(evokeds)} conditions")

        else:
            # Simple condition - just average
            try:
                epochs_subset = epochs[condition].copy()
                if len(epochs_subset) == 0:
                    print(
                        f"    [WARNING] No epochs for condition '{condition}'. Skipping."
                    )
                    continue
                evoked = epochs_subset.average()
                print(f"    - Averaged {len(epochs_subset)} epochs")
            except KeyError:
                print(
                    f"    [WARNING] Condition '{condition}' not found in epochs. Skipping."
                )
                continue

        # Set EEG reference if needed
        if "eeg" in cfg.ch_types:
            evoked.set_eeg_reference("average", projection=True)

        # Apply beamformer
        stc = apply_lcmv(evoked, filters)
        stcs[condition] = stc

        print(f"    - STC shape: {stc.data.shape}")

    print(
        f"[run_beamformer_timecourse] Completed. Generated {len(stcs)} source estimates."
    )

    return stcs


def run_beamformer_power(
    epochs: mne.Epochs,
    filters: dict,
    cfg: SimpleNamespace,
) -> Dict[str, mne.SourceEstimate]:
    """Apply beamformer to covariance matrices (power analysis).

    This is the SECONDARY beamformer analysis, following the pattern
    from fit_beamformer.py.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    filters : dict
        LCMV filters from make_lcmv.
    cfg : SimpleNamespace
        Configuration object.

    Returns
    -------
    stcs : dict
        Dictionary mapping condition names to source estimates.
    """
    print("\n[run_beamformer_power] Running power beamformer analysis...")
    print(
        f"  - Time window: {cfg._beamformer_power_tmin} to {cfg._beamformer_power_tmax} s"
    )

    stcs = {}
    conditions = cfg.conditions  # Only run on base conditions, not contrasts for power

    print(f"\n\n[run_beamformer_power] Processing {len(conditions)} conditions")

    # Compute covariance for each condition
    covs = {}
    for condition in conditions:
        print(f"\n  - Computing covariance for condition: {condition}")

        try:
            epochs_subset = epochs[condition].copy()
            if len(epochs_subset) == 0:
                print(f"    [WARNING] No epochs for condition '{condition}'. Skipping.")
                continue

            # Compute covariance in specified time window
            cov = mne.compute_covariance(
                epochs_subset,
                method="shrunk",
                tmin=cfg._beamformer_power_tmin,
                tmax=cfg._beamformer_power_tmax,
                n_jobs=cfg.n_jobs,
            )
            covs[condition] = cov
            print(f"    - Computed from {len(epochs_subset)} epochs")

        except KeyError:
            print(
                f"    [WARNING] Condition '{condition}' not found in epochs. Skipping."
            )
            continue

    # Apply beamformer to each covariance
    for condition, cov in covs.items():
        print(f"  - Applying beamformer to {condition} covariance")
        stc = apply_lcmv_cov(cov, filters)
        stcs[condition] = stc
        print(f"    - STC shape: {stc.data.shape}")

    # CONTRASTS -------------------------------------
    print(f"\n\n[run_beamformer_power] Processing {len(cfg.contrasts)} contrasts")
    for contrast in cfg.contrasts:
        print("-" * 10)
        contrast_name = contrast["name"]
        contrast_conditions = contrast["conditions"]

        print(f"  - Computing contrast: {contrast_name}")
        print(f"    [{contrast}]")

        stc_list = []
        for condition in contrast_conditions:
            # try:
            print(f"    contrast_condition: {condition}")
            epochs_subset = epochs[f"{condition}"].copy()

            if len(epochs_subset) == 0:
                print(f"    [WARNING] No epochs for condition '{condition}'. Skipping.")
                continue

            # if stcs[condition] is not None:
            #     continue

            # Compute covariance in specified time window
            cov = mne.compute_covariance(
                epochs_subset,
                method="shrunk",
                tmin=cfg._beamformer_power_tmin,
                tmax=cfg._beamformer_power_tmax,
                n_jobs=cfg.n_jobs,
            )
            stc_list.append(apply_lcmv_cov(cov, filters))
            print(f"    - Computed from {len(epochs_subset)} epochs")

        # For power analysis, use normalized difference: W*stc / |W|*stc
        # This is more appropriate for power than weighted sums
        # stc_list = [stcs[cond] for cond in contrast_conditions if cond in stcs]
        if not stc_list:
            print(
                f"    [WARNING] No valid STCs found for contrast '{contrast_name}'. Skipping."
            )
            continue

        stc_contrast = stc_list[0].copy()
        stc_norm = stc_list[0].copy()

        # sum each item in stc_list, weigthed by contrast weights
        for i, stc in enumerate(stc_list):
            weight = contrast["weights"][i]
            stc_contrast.data += weight * stc.data
            stc_norm.data += abs(weight) * stc.data
        stc_contrast.data /= stc_norm.data
        stcs[contrast_name] = stc_contrast
        print(f"    - Normalized difference contrast created")

    print(f"[run_beamformer_power] Completed. Generated {len(stcs)} source estimates.")

    return stcs


# --------------------------------------------------------------------------------------
# Result Saving
# --------------------------------------------------------------------------------------


def save_beamformer_results(
    cfg: SimpleNamespace,
    filters: dict,
    stcs: Dict[str, mne.SourceEstimate],
    analysis_type: str,
    source_space: str | None = None,
) -> Dict[str, Path]:
    """Save beamformer results to BIDS derivatives.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    filters : dict
        LCMV filters.
    stcs : dict
        Source estimates for each condition.
    analysis_type : str
        'time' or 'power' to distinguish analysis types.
    source_space : str or None
        'surface' or 'volume'.  Controls the STC/filter naming so both
        reconstructions can coexist.  When ``None``, the first entry of
        ``_beamformer_source_space`` is used (single-space back-compat).

    Returns
    -------
    out_files : dict
        Dictionary mapping condition names to output file paths.
    """
    if source_space is None:
        source_space = resolve_source_spaces(cfg)[0]
    print(
        f"\n[save_beamformer_results] Saving {analysis_type} beamformer results "
        f"({source_space})..."
    )

    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    out_files = {}

    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        root=cfg.deriv_root,
        datatype=cfg.datatype,
        check=False,
    )

    # Save filters (once per analysis; volume filters get an acq-vol tag so a
    # combined surface+volume run does not overwrite one with the other).
    if cfg._beamformer_save_filters and analysis_type == "time":
        filter_path = bids_path.copy().update(suffix="lcmv", extension=".h5")
        if source_space == "volume":
            filter_path = filter_path.update(acquisition="vol")
        filters.save(filter_path.fpath, overwrite=True)
        out_files["filters"] = filter_path.fpath

    # Volume STCs are stored under a distinct "+vol" token (and save as "-vl.h5")
    # so downstream tooling can tell them apart from surface "+hemi" ("-stc.h5").
    space_tag = "vol" if source_space == "volume" else "hemi"

    # Save source estimates
    for condition, stc in stcs.items():
        # Create suffix based on analysis type
        cond_sanitized = sanitize_cond_name(condition)
        if analysis_type == "time":
            suffix = f"{cond_sanitized}+lcmv+{space_tag}"
        else:  # power
            suffix = f"{cond_sanitized}+lcmv-power+{space_tag}"

        stc_path = bids_path.copy().update(suffix=suffix)
        print(f"  - Saving {condition} to: {stc_path.fpath}")

        stc.save(stc_path.fpath, ftype="h5", overwrite=True)
        out_files[condition] = stc_path.fpath

    print(f"[save_beamformer_results] Saved {len(stcs)} source estimates")

    return out_files


# --------------------------------------------------------------------------------------
# Report Generation
# --------------------------------------------------------------------------------------


def add_to_report(
    cfg: SimpleNamespace,
    stcs: Dict[str, Path],
    analysis_type: str,
    src: "mne.SourceSpaces | None" = None,
    source_space: str | None = None,
) -> None:
    """Add beamformer results to MNE-BIDS-Pipeline HTML report.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object.
    stcs : dict
        Dictionary mapping condition names to STC file paths.
    analysis_type : str
        'time' or 'power' to distinguish analysis types.
    src : mne.SourceSpaces or None
        Volume source space.  Required when ``source_space == 'volume'``:
        ``report.add_stc`` cannot render volume estimates (it has no ``src``
        parameter), so each volume STC is plotted with
        ``stc.plot(src=..., mode='stat_map')`` and added as a figure instead.
    source_space : str or None
        'surface' or 'volume'.  Selects the rendering path and namespaces the
        report titles/tags so a combined run keeps both.  When ``None``, the
        first entry of ``_beamformer_source_space`` is used.
    """
    if not cfg._beamformer_add_to_report:
        print(f"\n[add_to_report] Report generation disabled. Skipping.")
        return

    if source_space is None:
        source_space = resolve_source_spaces(cfg)[0]
    is_volume = source_space == "volume"

    print(
        f"\n[add_to_report] Adding {analysis_type} beamformer results "
        f"({source_space}) to report..."
    )

    subject = cfg.subjects[0]
    session = cfg.sessions[0]

    # Strip BIDS prefixes if present (report system adds them back)
    subject_clean = (
        subject.replace("sub-", "") if subject.startswith("sub-") else subject
    )
    session_clean = (
        session.replace("ses-", "") if session.startswith("ses-") else session
    )
    print(f"[add_to_report] Clean subject: {subject_clean}, session: {session_clean}")

    # fs subject and subjects_dir
    fs_subject = get_fs_subject(config=cfg, subject=subject, session=session)
    fs_subjects_dir = get_fs_subjects_dir(config=cfg)
    print(
        f"[add_to_report] FreeSurfer subject: {fs_subject}, subjects_dir: {fs_subjects_dir}"
    )

    try:
        with _open_report(
            cfg=cfg,
            exec_params=cfg.exec_params,
            subject=subject_clean,
            session=session_clean,
        ) as report:
            print(f"[add_to_report] Report opened successfully")

            for condition, stc_path in stcs.items():
                if condition == "filters":
                    continue  # Skip the filters entry

                print(f"  - Adding {condition} to report")

                # Determine tags (namespaced by source space so a combined
                # surface+volume run keeps both entries in the report).
                tag_prefix = f"beamformer-{analysis_type}-{source_space}"
                tags = (tag_prefix, _sanitize_cond_tag(condition))
                report_title = (
                    f"Beamformer {source_space} ({analysis_type}): {condition}"
                )

                # Add 'contrast' tag if this is a contrast
                if condition not in cfg.conditions:
                    tags = tags + ("contrast",)

                if is_volume:
                    # report.add_stc has no `src` argument and cannot render a
                    # VolSourceEstimate, so plot it with nilearn and add the
                    # resulting figure instead.
                    if src is None:
                        print(
                            f"    [WARNING] Volume STC but no source space provided; "
                            f"skipping report figure for {condition}"
                        )
                        continue
                    import matplotlib.pyplot as plt

                    stc = mne.read_source_estimate(str(stc_path))
                    # Representative slice at the peak of the mean-over-sources signal.
                    peak_time = stc.times[
                        int(np.argmax(np.abs(stc.data).mean(axis=0)))
                    ]
                    fig = stc.plot(
                        src=src,
                        subject=fs_subject,
                        subjects_dir=fs_subjects_dir,
                        mode="stat_map",
                        initial_time=peak_time,
                        show=False,
                    )
                    report.add_figure(
                        fig=fig,
                        title=report_title,
                        tags=tags,
                        replace=True,
                    )
                    plt.close(fig)
                else:
                    report.add_stc(
                        stc=stc_path,
                        title=report_title,
                        subject=fs_subject,
                        subjects_dir=fs_subjects_dir,
                        n_time_points=cfg.report_stc_n_time_points,
                        tags=tags,
                        replace=True,
                    )
                print(f"    - tags: {tags}")

            print(
                f"[add_to_report] Successfully added {len(stcs) - 1} source estimates to report"
            )

    except Exception as e:
        print(f"[add_to_report] Warning: Could not add to report: {e}")
        print(f"[add_to_report] Continuing without report update...")
        exit(1)


# --------------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="LCMV Beamformer source reconstruction for OPM-MEG data"
    )

    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    return p.parse_args()


def main():
    """Main entry point for beamformer analysis."""

    # Parse command-line arguments
    args = parse_args()

    # load configuration
    cfg = _import_config(config_path=args.config)
    _update_config_from_path(config=cfg, config_path=args.config)
    cfg.data_type = "meg"
    cfg.datatype = "meg"

    # Check if beamformer is enabled
    if not cfg._run_beamformer:
        print("\n[main] Beamformer disabled in configuration (_run_beamformer=False)")
        print("[main] Exiting without running analysis.")
        return

    # Load data
    data = load_beamformer_data(cfg)

    # Compute data covariance (shared by both analyses)
    print("\n[main] Computing data covariance matrix...")
    rank = mne.compute_rank(data["epochs"], info=data["info"], tol="auto")
    data_cov = mne.compute_covariance(
        data["epochs"],
        method="shrunk",
        rank=rank,
        n_jobs=cfg.n_jobs,
    )
    print(f"[main] Data covariance computed from {len(data['epochs'])} epochs")

    if data["noise_path"] is None:
        rank = "info"
        noise_cov = None
    else:
        print(f"\n[main] Loading noise covariance from: {data['noise_path']}")
        noise_raw = mne.io.read_raw_fif(data["noise_path"], preload=True)
        rank = mne.compute_rank(noise_raw, info=noise_raw.info, tol="auto")
        noise_cov = mne.compute_raw_covariance(
            noise_raw,
            method="shrunk",
            rank=rank,
            n_jobs=cfg.n_jobs,
        )
        print(f"[main] Noise covariance computed from raw data: {data['noise_path']}")

    if getattr(cfg, "_beamformer_rank", "info") == "empty_room":
        print(f"[main] Using empty-room noise covariance rank: {rank}")
    else:
        rank = cfg._beamformer_rank
        print(f"[main] Using specified beamformer rank: {rank}")

    # Reconstruct each requested source space (surface, volume, or both).  The
    # data / noise covariance and rank above are shared; the forward, filters,
    # STCs, saved filenames and report entries are per source space.
    source_spaces = resolve_source_spaces(cfg)
    output_type = getattr(cfg, "_beamformer_output_type", "both")

    for space in source_spaces:
        print("\n" + "#" * 80)
        print(f"SOURCE SPACE: {space.upper()}")
        print("#" * 80)

        forward = data["forwards"][space]

        # Compute LCMV filters (shared by this space's time and power analyses)
        filters = compute_lcmv_filters(
            forward=forward,
            data_cov=data_cov,
            noise_cov=noise_cov,
            rank=rank,
            info=data["info"],
            cfg=cfg,
        )

        # Run Time-locked beamformer --------------------------------
        if output_type in ["time", "both"]:
            print("\n" + "=" * 80)
            print(f"TIME-LOCKED BEAMFORMER ({space})")
            print("=" * 80)
            stcs_time = run_beamformer_timecourse(
                epochs=data["epochs"],
                filters=filters,
                cfg=cfg,
            )

            out_files_time = save_beamformer_results(
                cfg=cfg,
                filters=filters,
                stcs=stcs_time,
                analysis_type="time",
                source_space=space,
            )

            add_to_report(
                cfg=cfg,
                stcs=out_files_time,
                analysis_type="time",
                src=forward["src"],
                source_space=space,
            )

        # Run Power beamformer --------------------------------
        if output_type in ["power", "both"]:
            print("\n" + "=" * 80)
            print(f"POWER BEAMFORMER ({space})")
            print("=" * 80)

            stcs_power = run_beamformer_power(
                epochs=data["epochs"],
                filters=filters,
                cfg=cfg,
            )

            out_files_power = save_beamformer_results(
                cfg=cfg,
                filters=filters,
                stcs=stcs_power,
                analysis_type="power",
                source_space=space,
            )

            add_to_report(
                cfg=cfg,
                stcs=out_files_power,
                analysis_type="power",
                src=forward["src"],
                source_space=space,
            )

    print("\n" + "=" * 80)
    print("BEAMFORMER ANALYSIS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
