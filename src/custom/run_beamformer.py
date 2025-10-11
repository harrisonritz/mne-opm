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
import mne_bids
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv
from mne_bids import BIDSPath

# Add mne-bids-pipeline to path for importing utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mne-bids-pipeline"))

from mne_bids_pipeline._config_import import _update_config_from_path
from mne_bids_pipeline._config_utils import (
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
    
    # Load config file
    config = _update_config_from_path(SimpleNamespace(), config_path)
    
    # Extract environment variables (matching custom_preproc.py pattern)
    subject = os.environ.get("SUBJECT", config.subjects[0])
    session = os.environ.get("SESSION", "01")
    
    # Create simplified config namespace
    cfg = SimpleNamespace(
        # Core BIDS settings
        bids_root=Path(config.bids_root),
        deriv_root=Path(config.deriv_root),
        subjects=[subject],
        sessions=[session],
        task=config.task,
        datatype=config.datatype,
        
        # Beamformer-specific settings
        _run_beamformer=getattr(config, "_run_beamformer", True),
        _beamformer_reg=getattr(config, "_beamformer_reg", 0.05),
        _beamformer_pick_ori=getattr(config, "_beamformer_pick_ori", "max-power"),
        _beamformer_weight_norm=getattr(config, "_beamformer_weight_norm", "unit-noise-gain"),
        _beamformer_depth=getattr(config, "_beamformer_depth", None),
        _beamformer_rank=getattr(config, "_beamformer_rank", "info"),
        _beamformer_save_filters=getattr(config, "_beamformer_save_filters", True),
        _beamformer_add_to_report=getattr(config, "_beamformer_add_to_report", True),
        _beamformer_report_n_time_points=getattr(config, "_beamformer_report_n_time_points", 51),
        _beamformer_power_tmin=getattr(config, "_beamformer_power_tmin", 0.0),
        _beamformer_power_tmax=getattr(config, "_beamformer_power_tmax", 0.5),
        
        # Analysis settings
        conditions=config.conditions,
        contrasts=config.contrasts,
        ch_types=config.ch_types,
        noise_cov=config.noise_cov,
        source_info_path_update=config.source_info_path_update,
        
        # FreeSurfer settings
        fs_subject=get_fs_subject(config=config, subject=subject, session=session),
        fs_subjects_dir=get_fs_subjects_dir(config),
        
        # For report generation
        exec_params=config.exec_params,
        report_stc_n_time_points=config.report_stc_n_time_points,
    )
    
    print(f"[load_config] Subject: {cfg.subjects[0]}, Session: {cfg.sessions[0]}")
    print(f"[load_config] Task: {cfg.task}")
    print(f"[load_config] Beamformer enabled: {cfg._run_beamformer}")
    
    return cfg


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
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    
    # Load forward solution
    fwd_path = bids_path.copy().update(suffix="fwd", extension=".fif")
    if not fwd_path.fpath.exists():
        raise FileNotFoundError(
            f"Forward solution not found at {fwd_path.fpath}\n"
            f"Run forward modeling first with:\n"
            f"  mne_bids_pipeline --steps=source/make_forward --config=<config>"
        )
    print(f"[load_beamformer_data] Loading forward solution: {fwd_path.fpath}")
    data["forward"] = mne.read_forward_solution(fwd_path)
    
    # Load clean epochs
    epochs_path = bids_path.copy().update(
        suffix="epo", processing="clean", extension=".fif"
    )
    if not epochs_path.fpath.exists():
        raise FileNotFoundError(
            f"Clean epochs not found at {epochs_path.fpath}\n"
            f"Run preprocessing first."
        )
    print(f"[load_beamformer_data] Loading epochs: {epochs_path.fpath}")
    data["epochs"] = mne.read_epochs(epochs_path, preload=True)
    
    # Load noise covariance
    if cfg.noise_cov == "ad-hoc":
        print("[load_beamformer_data] Using ad-hoc noise covariance")
        data["noise_cov"] = None
    else:
        try:
            noise_cov_path = get_noise_cov_bids_path(
                cfg=cfg, subject=subject, session=session
            )
            print(f"[load_beamformer_data] Loading noise covariance: {noise_cov_path.fpath}")
            data["noise_cov"] = mne.read_cov(noise_cov_path)
        except Exception as e:
            print(f"[load_beamformer_data] Warning: Could not load noise covariance: {e}")
            print("[load_beamformer_data] Using ad-hoc noise covariance instead")
            data["noise_cov"] = None
    
    # Load info from appropriate source
    if cfg.source_info_path_update is None:
        if cfg.noise_cov in ("rest", "noise"):
            source_info_path_update = dict(
                processing="clean", suffix="raw", task=cfg.noise_cov
            )
        else:
            source_info_path_update = dict(suffix="ave")
    else:
        source_info_path_update = cfg.source_info_path_update
    
    info_path = bids_path.copy().update(**source_info_path_update, extension=".fif")
    print(f"[load_beamformer_data] Loading info from: {info_path.fpath}")
    data["info"] = mne.io.read_info(info_path)
    
    print(f"[load_beamformer_data] Data loading complete")
    print(f"  - Forward: {len(data['forward']['src'])} source spaces")
    print(f"  - Epochs: {len(data['epochs'])} epochs, {len(data['epochs'].ch_names)} channels")
    print(f"  - Noise cov: {'ad-hoc' if data['noise_cov'] is None else 'loaded from file'}")
    
    return data


# --------------------------------------------------------------------------------------
# Beamformer Computation
# --------------------------------------------------------------------------------------


def compute_lcmv_filters(
    forward: mne.Forward,
    data_cov: mne.Covariance,
    noise_cov: mne.Covariance | None,
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
    print(f"  - Rank: {cfg._beamformer_rank}")
    
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
    if (cfg._beamformer_pick_ori == "vector" and 
        cfg._beamformer_weight_norm == "unit-noise-gain"):
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
        rank=cfg._beamformer_rank,
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
                print(f"    [WARNING] Could not find contrast definition for '{condition}'. Skipping.")
                continue
            
            # Average epochs for each condition in the contrast
            evokeds = []
            for cond_name in contrast_def["conditions"]:
                try:
                    epochs_subset = epochs[cond_name]
                    if len(epochs_subset) == 0:
                        print(f"    [WARNING] No epochs for condition '{cond_name}'. Skipping contrast.")
                        continue
                    evokeds.append(epochs_subset.average())
                except KeyError:
                    print(f"    [WARNING] Condition '{cond_name}' not found in epochs. Skipping contrast.")
                    continue
            
            if len(evokeds) != len(contrast_def["conditions"]):
                print(f"    [WARNING] Could not load all conditions for contrast. Skipping.")
                continue
            
            # Combine evoked responses with weights
            evoked = mne.combine_evoked(evokeds, weights=contrast_def["weights"])
            print(f"    - Created contrast from {len(evokeds)} conditions")
            
        else:
            # Simple condition - just average
            try:
                epochs_subset = epochs[condition]
                if len(epochs_subset) == 0:
                    print(f"    [WARNING] No epochs for condition '{condition}'. Skipping.")
                    continue
                evoked = epochs_subset.average()
                print(f"    - Averaged {len(epochs_subset)} epochs")
            except KeyError:
                print(f"    [WARNING] Condition '{condition}' not found in epochs. Skipping.")
                continue
        
        # Set EEG reference if needed
        if "eeg" in cfg.ch_types:
            evoked.set_eeg_reference("average", projection=True)
        
        # Apply beamformer
        stc = apply_lcmv(evoked, filters)
        stcs[condition] = stc
        
        print(f"    - STC shape: {stc.data.shape}")
    
    print(f"[run_beamformer_timecourse] Completed. Generated {len(stcs)} source estimates.")
    
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
    print(f"  - Time window: {cfg._beamformer_power_tmin} to {cfg._beamformer_power_tmax} s")
    
    stcs = {}
    conditions = cfg.conditions  # Only run on base conditions, not contrasts for power
    
    print(f"[run_beamformer_power] Processing {len(conditions)} conditions")
    
    # Compute covariance for each condition
    covs = {}
    for condition in conditions:
        print(f"  - Computing covariance for condition: {condition}")
        
        try:
            epochs_subset = epochs[condition]
            if len(epochs_subset) == 0:
                print(f"    [WARNING] No epochs for condition '{condition}'. Skipping.")
                continue
            
            # Compute covariance in specified time window
            cov = mne.compute_covariance(
                epochs_subset,
                method="shrunk",
                tmin=cfg._beamformer_power_tmin,
                tmax=cfg._beamformer_power_tmax,
            )
            covs[condition] = cov
            print(f"    - Computed from {len(epochs_subset)} epochs")
            
        except KeyError:
            print(f"    [WARNING] Condition '{condition}' not found in epochs. Skipping.")
            continue
    
    # Apply beamformer to each covariance
    for condition, cov in covs.items():
        print(f"  - Applying beamformer to {condition} covariance")
        stc = apply_lcmv_cov(cov, filters)
        stcs[condition] = stc
        print(f"    - STC shape: {stc.data.shape}")
    
    # Compute contrasts if we have sufficient conditions
    print(f"[run_beamformer_power] Computing power contrasts...")
    for contrast in cfg.contrasts:
        contrast_name = contrast["name"]
        contrast_conditions = contrast["conditions"]
        
        # Check if all conditions needed for contrast exist
        if not all(cond in stcs for cond in contrast_conditions):
            print(f"  - Skipping contrast '{contrast_name}': not all conditions available")
            continue
        
        print(f"  - Computing contrast: {contrast_name}")
        
        # For power analysis, use normalized difference: (A - B) / (A + B)
        # This is more appropriate for power than weighted sums
        stc_list = [stcs[cond] for cond in contrast_conditions]
        
        if len(stc_list) == 2:
            # Simple two-condition contrast
            stc_A, stc_B = stc_list
            stc_contrast = stc_A.copy()
            stc_contrast.data = (stc_A.data - stc_B.data) / (stc_A.data + stc_B.data + 1e-10)
            stcs[contrast_name] = stc_contrast
            print(f"    - Normalized difference contrast created")
        else:
            print(f"    [WARNING] Power contrasts only supported for 2 conditions. Skipping.")
    
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
        
    Returns
    -------
    out_files : dict
        Dictionary mapping condition names to output file paths.
    """
    print(f"\n[save_beamformer_results] Saving {analysis_type} beamformer results...")
    
    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    out_files = {}
    
    # Construct base BIDS path
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    
    # Save filters (only once, shared by both analyses)
    if cfg._beamformer_save_filters and analysis_type == "time":
        filter_path = bids_path.copy().update(suffix="lcmv", extension=".h5")
        print(f"  - Saving filters to: {filter_path.fpath}")
        filters.save(filter_path.fpath, overwrite=True)
        out_files["filters"] = filter_path.fpath
    
    # Save source estimates
    for condition, stc in stcs.items():
        # Create suffix based on analysis type
        cond_sanitized = sanitize_cond_name(condition)
        if analysis_type == "time":
            suffix = f"{cond_sanitized}+lcmv+hemi"
        else:  # power
            suffix = f"{cond_sanitized}+lcmv-power+hemi"
        
        stc_path = bids_path.copy().update(suffix=suffix, extension=".h5")
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
    """
    if not cfg._beamformer_add_to_report:
        print(f"\n[add_to_report] Report generation disabled. Skipping.")
        return
    
    print(f"\n[add_to_report] Adding {analysis_type} beamformer results to report...")
    
    subject = cfg.subjects[0]
    session = cfg.sessions[0]
    
    try:
        with _open_report(
            cfg=cfg,
            exec_params=cfg.exec_params,
            subject=subject,
            session=session,
        ) as report:
            print(f"[add_to_report] Report opened successfully")
            
            for condition, stc_path in stcs.items():
                if condition == "filters":
                    continue  # Skip the filters entry
                
                print(f"  - Adding {condition} to report")
                
                # Determine tags
                tag_prefix = f"beamformer-{analysis_type}"
                tags = (tag_prefix, _sanitize_cond_tag(condition))
                
                # Add 'contrast' tag if this is a contrast
                if condition not in cfg.conditions:
                    tags = tags + ("contrast",)
                
                # Add to report
                report.add_stc(
                    stc=stc_path,
                    title=f"Beamformer ({analysis_type}): {condition}",
                    subject=cfg.fs_subject,
                    subjects_dir=cfg.fs_subjects_dir,
                    n_time_points=cfg._beamformer_report_n_time_points,
                    tags=tags,
                    replace=True,
                )
                
                print(f"    - Added with tags: {tags}")
            
            print(f"[add_to_report] Successfully added {len(stcs)-1} source estimates to report")
    
    except Exception as e:
        print(f"[add_to_report] Warning: Could not add to report: {e}")
        print(f"[add_to_report] Continuing without report update...")


# --------------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------------


def main():
    """Main entry point for beamformer analysis."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="LCMV Beamformer source reconstruction for OPM-MEG data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="both",
        choices=["time", "power", "both"],
        help="Type of beamformer analysis: 'time' (evoked), 'power' (covariance), or 'both'",
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("LCMV BEAMFORMER ANALYSIS")
    print("=" * 80)
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Check if beamformer is enabled
    if not cfg._run_beamformer:
        print("\n[main] Beamformer disabled in configuration (_run_beamformer=False)")
        print("[main] Exiting without running analysis.")
        return
    
    # Load data
    data = load_beamformer_data(cfg)
    
    # Compute data covariance (shared by both analyses)
    print("\n[main] Computing data covariance matrix...")
    data_cov = mne.compute_covariance(data["epochs"], method="shrunk")
    print(f"[main] Data covariance computed from {len(data['epochs'])} epochs")
    
    # Compute LCMV filters (shared by both analyses)
    filters = compute_lcmv_filters(
        forward=data["forward"],
        data_cov=data_cov,
        noise_cov=data["noise_cov"],
        info=data["info"],
        cfg=cfg,
    )
    
    # Run PRIMARY analysis: Time-locked beamformer
    if args.output_type in ("time", "both"):
        print("\n" + "=" * 80)
        print("PRIMARY ANALYSIS: TIME-LOCKED BEAMFORMER")
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
        )
        
        add_to_report(
            cfg=cfg,
            stcs=out_files_time,
            analysis_type="time",
        )
    
    # Run SECONDARY analysis: Power beamformer
    if args.output_type in ("power", "both"):
        print("\n" + "=" * 80)
        print("SECONDARY ANALYSIS: POWER BEAMFORMER")
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
        )
        
        add_to_report(
            cfg=cfg,
            stcs=out_files_power,
            analysis_type="power",
        )
    
    print("\n" + "=" * 80)
    print("BEAMFORMER ANALYSIS COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
