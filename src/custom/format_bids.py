"""Convert Cerca OPM data to BIDS format.

This module converts raw OPM-MEG data (with optional empty-room recordings,
anatomical images, and eye-tracking) into a BIDS-compliant directory structure.

Functions
---------
set_bids_params
    Load BIDS conversion configuration from environment + config file.
validate_raw_folder
    Print file tree and validate the participant's raw folder structure.
convert_triggers
    Convert 8-bit trigger channels into combined annotations.
process_eyetracking
    Full eye-tracking processing pipeline (load, interpolate, align, merge).
bids_conversion
    Main BIDS conversion pipeline.

Author: Harrison Ritz (2025)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mne
import mne_bids
import numpy as np
from mne_bids_pipeline._config_import import _update_config_from_path


# ===========================================================================
# Configuration
# ===========================================================================

# Default screen parameters for eye-tracking calibration
_DEFAULT_SCREEN_RESOLUTION = (1920, 1080)
_DEFAULT_SCREEN_SIZE = (0.606, 0.341)  # metres
_DEFAULT_SCREEN_DISTANCE = 0.895  # metres

# Head position channel names recorded by the eye-tracker
_HEAD_POS_CHANNELS = ("x_head", "y_head", "distance")


def set_bids_params(config_path: str = "") -> SimpleNamespace:
    """Load BIDS conversion configuration.

    Reads directory paths from environment variables ``RAW_DIR`` and
    ``BIDS_DIR``, then merges settings from an optional Python config file.

    Parameters
    ----------
    config_path : str
        Path to a Python configuration file.  If empty, only environment
        variables and defaults are used.

    Returns
    -------
    config : SimpleNamespace
        Flat configuration namespace.
    """
    print("\n\n\n[loading configuration]\n")

    config = SimpleNamespace(
        # Directory paths
        raw_dir=os.environ.get("RAW_DIR", ""),
        bids_dir=os.environ.get("BIDS_DIR", ""),
        # Session information
        ids=0,
        task="",
        session="",
        # Trigger information
        rename_annot=True,
        trigger_desc={},
        response_desc={},
        # Recording information
        line_freq=60.0,
        bads=[],
        crop=0,
        # Eye-tracking screen parameters
        screen_resolution=_DEFAULT_SCREEN_RESOLUTION,
        screen_size=_DEFAULT_SCREEN_SIZE,
        screen_distance=_DEFAULT_SCREEN_DISTANCE,
    )

    if config_path:
        print(f"\n\nloading config from Python file: {config_path}\n")
        try:
            _update_config_from_path(config=config, config_path=config_path)
        except Exception as e:
            print(
                f"error loading config from Python file: {e},\n"
                "creating new config from template"
            )
            template_path = os.path.join(
                os.path.dirname(config_path), "TEMPLATE_config-bids.py"
            )
            with open(template_path, "r") as f:
                template_content = f.read()
            with open(config_path, "w") as f:
                f.write(template_content)
            _update_config_from_path(config=config, config_path=config_path)

    print('\nconfig:"\n', config)
    return config


# ===========================================================================
# Folder validation
# ===========================================================================


def _build_file_tree(directory: str, prefix: str = "", max_depth: int = 3) -> str:
    """Build a visual file-tree string for *directory*.

    Parameters
    ----------
    directory : str
        Root directory to display.
    prefix : str
        Line prefix for recursive indentation (internal use).
    max_depth : int
        Maximum depth of recursion.

    Returns
    -------
    tree : str
        Multi-line string showing the directory structure.
    """
    lines: list[str] = []
    dir_path = Path(directory)

    if not dir_path.is_dir():
        return f"  {prefix}{dir_path.name}/ [NOT FOUND]"

    entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        if entry.is_dir():
            lines.append(f"  {prefix}{connector}{entry.name}/")
            if max_depth > 1:
                extension = "    " if i == len(entries) - 1 else "│   "
                subtree = _build_file_tree(
                    str(entry), prefix=prefix + extension, max_depth=max_depth - 1
                )
                if subtree:
                    lines.append(subtree)
        else:
            lines.append(f"  {prefix}{connector}{entry.name}")

    return "\n".join(lines)


def validate_raw_folder(raw_dir: str, subj: int) -> dict[str, object]:
    """Validate the participant's raw folder and print its file tree.

    Checks naming conventions (``*_NNN``, ``*_task``, ``*_noise``) and warns
    about unexpected file counts.  Raises on critical errors (missing subject
    folder or task files).

    Parameters
    ----------
    raw_dir : str
        Root raw-data directory.
    subj : int
        Participant numeric ID (zero-padded to 3 digits for matching).

    Returns
    -------
    paths : dict
        Dictionary with keys ``"emptyroom"``, ``"task"``, ``"t1w"``,
        ``"t2w"``, ``"eye"`` mapping to discovered file paths (or ``None``).

    Raises
    ------
    FileNotFoundError
        If the subject folder or required task files are missing.
    """
    subj_pattern = f"*_{subj:03}"
    subj_dirs = glob.glob(os.path.join(raw_dir, subj_pattern))

    if not subj_dirs:
        raise FileNotFoundError(
            f"No subject folder matching '{subj_pattern}' found in {raw_dir}.\n"
            f"Expected a folder ending with '_{subj:03}'."
        )

    subj_dir = subj_dirs[0]
    if len(subj_dirs) > 1:
        print(
            f"  WARNING: Multiple folders match '{subj_pattern}': {subj_dirs}. "
            f"Using first: {subj_dir}"
        )

    # Print file tree
    print(f"\n{'=' * 60}")
    print(f"  Raw folder structure for subject {subj:03}")
    print(f"{'=' * 60}")
    print(f"  {os.path.basename(subj_dir)}/")
    print(_build_file_tree(subj_dir))
    print(f"{'=' * 60}\n")

    # --- Discover files ---
    errors: list[str] = []
    warnings: list[str] = []

    # Task files (required)
    task_files = glob.glob(os.path.join(subj_dir, "*_task", "*_meg.fif"))
    if not task_files:
        errors.append(
            "No task files found (expected *_task/*_meg.fif). "
            "Ensure task subfolders end with '_task'."
        )

    # Empty-room (optional)
    noise_files = glob.glob(os.path.join(subj_dir, "*_noise", "*_meg.fif"))
    emptyroom = noise_files[0] if noise_files else None
    if len(noise_files) > 1:
        warnings.append(
            f"Multiple noise recordings found ({len(noise_files)}); using first."
        )

    # Anatomical images (optional)
    t1w_files = glob.glob(os.path.join(subj_dir, "*", "*_t1w.nii*"))
    t1w = t1w_files[0] if t1w_files else None
    if len(t1w_files) > 1:
        warnings.append(f"Multiple T1w images found ({len(t1w_files)}); using first.")

    t2w_files = glob.glob(os.path.join(subj_dir, "*", "*_t2w.nii*"))
    t2w = t2w_files[0] if t2w_files else None
    if len(t2w_files) > 1:
        warnings.append(f"Multiple T2w images found ({len(t2w_files)}); using first.")

    # Eye-tracking (optional)
    eye_files = glob.glob(os.path.join(subj_dir, "*", "*.asc"))
    eye = eye_files[0] if eye_files else None
    if len(eye_files) > 1:
        warnings.append(
            f"Multiple .asc eye-tracking files found ({len(eye_files)}); using first."
        )

    # --- Check subfolder naming conventions ---
    for entry in Path(subj_dir).iterdir():
        if entry.is_dir():
            name = entry.name
            # Subfolders should end with _task, _noise, or be a known name
            has_convention = (
                name.endswith("_task")
                or name.endswith("_noise")
                or name.lower() in {"anat", "anatomy", "eyetrack", "eyetracking"}
            )
            if not has_convention:
                warnings.append(
                    f"Subfolder '{name}' does not follow naming convention "
                    f"(*_task, *_noise, anat). It will be ignored."
                )

    # Check subject folder naming
    subj_folder_name = os.path.basename(subj_dir)
    if not re.search(r"_\d{3}$", subj_folder_name):
        warnings.append(
            f"Subject folder '{subj_folder_name}' does not end with "
            f"3-digit zero-padded ID (expected *_{subj:03})."
        )

    # --- Report ---
    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
        print()

    if errors:
        print("  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        raise FileNotFoundError(
            f"Critical validation errors for subject {subj}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    print("  Validation passed.\n")

    return {
        "emptyroom": emptyroom,
        "task": task_files,
        "t1w": t1w,
        "t2w": t2w,
        "eye": eye,
    }


# ===========================================================================
# Trigger conversion
# ===========================================================================


def convert_triggers(raw: mne.io.Raw, cfg: SimpleNamespace) -> mne.io.Raw:
    """Convert 8-bit trigger channels to a combined channel and annotations.

    Reads ``Trigger 1`` through ``Trigger 8``, converts their values to
    binary, packs them into an integer, and maps event IDs to descriptions
    using ``cfg.trigger_desc``.  Response annotations are renamed via
    ``cfg.response_desc``.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data containing individual trigger channels.
    cfg : SimpleNamespace
        Configuration with ``trigger_desc`` and ``response_desc`` dicts.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with updated annotations.
    """
    print("\n\n\nconverting triggers ----------------------\n")

    trigger_channels = [f"Trigger {i}" for i in range(1, 9)]

    # Stack trigger channels and binarise
    stacked = np.vstack([raw.get_data(ch) for ch in trigger_channels])
    stacked = (stacked > 2).astype(float)

    # Pack bits into integer
    powers = 2 ** np.arange(len(trigger_channels))[:, np.newaxis]
    combined = np.sum(stacked * powers, axis=0).astype(int)

    # Add combined trigger channel
    combined_info = mne.create_info(
        ["Trigger Combined"], raw.info["sfreq"], ["stim"]
    )
    combined_raw = mne.io.RawArray(combined.reshape(1, -1), combined_info)
    raw.add_channels([combined_raw], force_update_info=True)

    # Extract events and convert to annotations
    events = mne.find_events(
        raw, stim_channel="Trigger Combined", min_duration=0.001, consecutive=True
    )
    new_annotations = mne.annotations_from_events(
        events,
        event_desc=cfg.trigger_desc,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )

    # Remove old trigger-channel annotations
    old_ann = raw.copy().annotations
    keep_mask = np.array(
        [
            not any(ch in desc for ch in trigger_channels)
            for desc in old_ann.description
        ]
    )
    filtered_ann = mne.Annotations(
        onset=old_ann.onset[keep_mask],
        duration=old_ann.duration[keep_mask],
        description=old_ann.description[keep_mask],
        orig_time=old_ann.orig_time,
    )

    raw.set_annotations(filtered_ann + new_annotations)

    # Rename response annotations
    if getattr(cfg, "response_desc", None):
        raw.annotations.rename(cfg.response_desc)

    # Drop trigger stim channels now that all event information is captured
    # in annotations.  Keeping them would cause mne_bids.write_raw_bids()
    # to re-extract events with different find_events parameters whenever
    # the data is re-saved later in the pipeline (e.g. by bad_segments),
    # which can produce a different event count and break metadata alignment.
    stim_channels_to_drop = [ch for ch in trigger_channels + ["Trigger Combined"]
                             if ch in raw.ch_names]
    if stim_channels_to_drop:
        raw.drop_channels(stim_channels_to_drop)
        print(f"Dropped {len(stim_channels_to_drop)} trigger stim channels: "
              f"{stim_channels_to_drop}")

    print("Trigger & Response conversion completed.\n----------\n")
    return raw


# ===========================================================================
# Eye-tracking: modular helpers
# ===========================================================================


def _load_eyetracking(
    eye_path: str, cfg: SimpleNamespace
) -> tuple[mne.io.Raw, dict]:
    """Load eye-tracking data and apply calibration.

    Parameters
    ----------
    eye_path : str
        Path to the Eyelink ``.asc`` file.
    cfg : SimpleNamespace
        Configuration with screen parameters.

    Returns
    -------
    eye : mne.io.Raw
        Loaded and calibrated eye-tracking data.
    cal : dict
        Calibration object used.
    """
    print("\nLoading eye-tracking data from:", eye_path, "...")
    eye = mne.io.read_raw_eyelink(
        eye_path, create_annotations=True, apply_offsets=True, find_overlaps=True
    )

    # Read or create calibration
    print("\nCalibrating recording...")
    try:
        cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eye_path)
        print(f"found {len(cals)}, using first one")
        cal = cals[0]
    except Exception as e:
        print(f"***** error reading eyelink calibration: {e}")
        print("warning: assuming zero calibration error")
        cal = mne.preprocessing.eyetracking.Calibration(
            onset=0,
            model="HV13",
            eye="right",
            avg_error=0.0,
            max_error=0.0,
            positions=None,
            offsets=None,
            gaze=None,
        )

    cal["screen_resolution"] = getattr(
        cfg, "screen_resolution", _DEFAULT_SCREEN_RESOLUTION
    )
    cal["screen_size"] = getattr(cfg, "screen_size", _DEFAULT_SCREEN_SIZE)
    cal["screen_distance"] = getattr(cfg, "screen_distance", _DEFAULT_SCREEN_DISTANCE)
    print("calibration:", cal)

    mne.preprocessing.eyetracking.convert_units(eye, calibration=cal, to="radians")

    # print the channel names for eye
    print("Eye-tracking channels:", eye.ch_names)

    return eye, cal


def _get_head_pos_channels(eye: mne.io.Raw) -> list[str]:
    """Return the subset of head position channel names present in *eye*."""
    return [ch for ch in _HEAD_POS_CHANNELS if ch in eye.ch_names]


def _interpolate_nans(
    eye: mne.io.Raw,
    buffer_sec: float = 0.1,
    exclude_from_mask: list[str] | None = None,
) -> np.ndarray:
    """Interpolate NaN values in eye-tracking data with a buffer region.

    Marks samples within ``buffer_sec`` of any NaN as also needing
    interpolation, then applies linear interpolation from valid neighbours.

    Parameters
    ----------
    eye : mne.io.Raw
        Eye-tracking data (modified in-place).
    buffer_sec : float
        Buffer in seconds around NaN regions.
    exclude_from_mask : list of str or None
        Channel names to exclude from the returned ``orig_nan_mask``
        (they are still interpolated).  Useful for head-position channels
        whose NaN pattern should not feed into downstream feature
        extraction (e.g. NMF decomposition).

    Returns
    -------
    orig_nan_mask : np.ndarray
        Boolean mask of shape ``(n_times,)`` indicating original NaN
        positions (before interpolation), collapsed across non-excluded
        channels.
    """
    buffer_samp = int(buffer_sec * eye.info["sfreq"])
    print(f"\nInterpolating remaining NaNs (buffer = {buffer_sec} sec)...")

    data = eye.get_data()

    # Compute NaN mask only from channels that are *not* excluded
    if exclude_from_mask:
        mask_picks = [
            i for i, ch in enumerate(eye.ch_names) if ch not in exclude_from_mask
        ]
    else:
        mask_picks = list(range(len(eye.ch_names)))
    orig_nan_mask = np.isnan(data[mask_picks]).any(axis=0)

    # Report NaN statistics for excluded (head-position) channels
    if exclude_from_mask:
        for ch_name in exclude_from_mask:
            if ch_name in eye.ch_names:
                ch_idx = eye.ch_names.index(ch_name)
                n_nan = np.isnan(data[ch_idx]).sum()
                pct = n_nan / data.shape[1] * 100
                print(
                    f"  Head position channel '{ch_name}': "
                    f"{n_nan} NaN samples ({pct:.1f}%)"
                )

    for ch_idx, ch_name in enumerate(eye.ch_names):
        ch_data = data[ch_idx, :]
        nan_mask = np.isnan(ch_data)

        # Expand mask by buffer in both directions
        expanded = nan_mask.copy()
        for offset in range(1, buffer_samp + 1):
            if offset < len(ch_data):
                expanded[:-offset] |= nan_mask[offset:]  # future NaN
                expanded[offset:] |= nan_mask[:-offset]  # past NaN

        if not np.any(expanded):
            continue
        if np.all(expanded):
            print(f"Warning: All values NaN for '{ch_name}', skipping interpolation")
            continue

        nan_idx = np.where(expanded)[0]
        valid_idx = np.where(~expanded)[0]

        if len(valid_idx) > 1:
            data[ch_idx, nan_idx] = np.interp(
                nan_idx, valid_idx, ch_data[valid_idx]
            )
        elif len(valid_idx) == 1:
            data[ch_idx, nan_idx] = ch_data[valid_idx[0]]
            print(
                f"Warning: Only one valid point for '{ch_name}', "
                "using constant interpolation"
            )
        else:
            print(f"Warning: No valid data for interpolation in '{ch_name}'")

    eye._data = data
    return orig_nan_mask


def _annotation_to_timeseries(
    eye: mne.io.Raw, description: str, smooth_sec: float = 0.05
) -> np.ndarray:
    """Convert annotations matching *description* to a smoothed binary channel.

    Parameters
    ----------
    eye : mne.io.Raw
        Eye-tracking raw object.
    description : str
        Annotation description to match.
    smooth_sec : float
        Hanning window width for smoothing (seconds).

    Returns
    -------
    channel : np.ndarray
        Shape ``(1, n_times)`` smoothed binary indicator, scaled by ``1e-5``.
    """
    ts = np.zeros(eye._data.shape[1])
    for ann in eye.annotations:
        if ann["description"] == description:
            onset_samp = int((ann["onset"] - eye.first_time) * eye.info["sfreq"])
            dur_samp = int(np.ceil(ann["duration"] * eye.info["sfreq"]))
            ts[onset_samp : onset_samp + dur_samp] = 1.0

    window = np.hanning(int(smooth_sec * eye.info["sfreq"]))
    return np.convolve(ts, window, "same")[np.newaxis, :] * 1e-5


def _create_eye_feature_channels(
    eye: mne.io.Raw, orig_nan_mask: np.ndarray
) -> None:
    """Create NMF/SVD decomposition channels from blink/saccade/NaN signals.

    Adds ``eye_nmf1`` … ``eye_nmf3`` (or ``eye_pc1`` … ``eye_pc3`` if NMF
    fails) as EOG channels to *eye* in-place.

    Parameters
    ----------
    eye : mne.io.Raw
        Eye-tracking data (modified in-place).
    orig_nan_mask : np.ndarray
        Boolean mask of original NaN positions (from ``_interpolate_nans``).
    """
    print("\nRenaming 'BAD_blink' -> 'blink'...")
    eye.annotations.rename({"BAD_blink": "blink"})

    # Build feature channels
    print("Adding NaN / blink / saccade feature channels...")
    nan_channel = (
        np.convolve(
            orig_nan_mask, np.hanning(int(0.05 * eye.info["sfreq"])), "same"
        )[np.newaxis, :]
        * 1e-5
    )
    blink_channel = _annotation_to_timeseries(eye, "blink")
    saccade_channel = _annotation_to_timeseries(eye, "saccade")

    # NMF / SVD decomposition
    print("Computing NMF decomposition of eye feature channels...")
    feature_data = np.clip(
        np.vstack([nan_channel, blink_channel, saccade_channel]), 0.0, None
    )
    n_comp = min(3, feature_data.shape[0])

    try:
        from sklearn.decomposition import NMF

        model = NMF(
            n_components=n_comp, init="nndsvda", random_state=99, max_iter=500
        )
        model.fit_transform(feature_data)
        components = model.components_
        label = "eye_nmf"
        del model
    except Exception as e:
        print(f"NMF unavailable or failed ({e}); falling back to SVD.")
        u, s, vh = np.linalg.svd(feature_data, full_matrices=False)
        components = np.array([s[k] * vh[k, :] for k in range(n_comp)])
        label = "eye_pc"
        del u, s, vh

    for k in range(n_comp):
        comp_info = mne.create_info(
            [f"{label}{k + 1}"], eye.info["sfreq"], ch_types="eog"
        )
        comp_raw = mne.io.RawArray(
            components[k][np.newaxis, :],
            comp_info,
            first_samp=eye.first_samp,
            copy="auto",
        )
        eye.add_channels([comp_raw], force_update_info=True)

    print("Done adding eye-tracking channels. New info:")
    print(eye.info)


def _align_eyetracking(
    raw: mne.io.Raw, eye: mne.io.Raw
) -> tuple[mne.io.Raw, mne.io.Raw, float, float, float]:
    """Temporally align eye-tracking data to MEG raw data.

    Uses ``stim_onset`` (eye) and ``trial`` (raw) events to compute a
    polynomial mapping, then applies bilateral zero-padding so that
    ``mne.preprocessing.realign_raw`` crops eye data (never raw).

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data.
    eye : mne.io.Raw
        Eye-tracking data (modified in-place via padding & realignment).

    Returns
    -------
    raw : mne.io.Raw
        MEG data (may be modified by ``realign_raw``).
    eye : mne.io.Raw
        Realigned eye-tracking data.
    zero_ord_est : float
        Polynomial intercept (pre-pad).
    first_ord_est : float
        Polynomial slope (pre-pad).
    eye_original_duration : float
        Original eye recording duration in seconds.
    """
    from numpy.polynomial.polynomial import Polynomial

    eye_events, _ = mne.events_from_annotations(eye, regexp="stim_onset")
    raw_events, _ = mne.events_from_annotations(raw, regexp="trial")
    eye_shape, raw_shape = eye_events.shape[0], raw_events.shape[0]

    eye_duration = eye.times[-1] - eye.times[0]
    raw_duration = raw.times[-1] - raw.times[0]
    eye_shorter = eye_duration < raw_duration

    # Match events from beginning or end depending on which is shorter
    if eye_shorter:
        n_match = min(eye_shape, raw_shape)
        raw_times = (raw_events[:n_match, 0] / raw.info["sfreq"]) - raw.first_time
        eye_times = (eye_events[:n_match, 0] / eye.info["sfreq"]) - eye.first_time
        print(
            f"\nEye-tracker shorter than MEG — aligning from start "
            f"(matching first {n_match} events)."
        )
    else:
        raw_onset = max(0, raw_shape - eye_shape)
        eye_onset = max(0, eye_shape - raw_shape)
        raw_times = (raw_events[raw_onset:, 0] / raw.info["sfreq"]) - raw.first_time
        eye_times = (eye_events[eye_onset:, 0] / eye.info["sfreq"]) - eye.first_time

    eye_original_duration = eye_duration

    # Polynomial fit: t_raw = zero_ord + first_ord * t_eye
    poly = Polynomial.fit(x=eye_times, y=raw_times, deg=1)
    coefs = poly.convert(domain=(-1, 1)).coef
    zero_ord_est, first_ord_est = float(coefs[0]), float(coefs[1])

    print(f"\n--- Eye-tracking alignment diagnostics ---")
    print(f"  First matched event: raw={raw_times[0]:.3f}s, eye={eye_times[0]:.3f}s")
    print(f"  Last matched event:  raw={raw_times[-1]:.3f}s, eye={eye_times[-1]:.3f}s")
    print(f"  Durations: raw={raw_duration:.1f}s, eye={eye_duration:.1f}s")
    print(f"  Event counts: raw={raw_shape}, eye={eye_shape}")
    print(f"  Polynomial: zero_ord={zero_ord_est:.3f}s, first_ord={first_ord_est:.6f}")

    # --- Bilateral zero-padding ---
    pad_seconds = np.abs(raw_duration - eye_duration) + 60.0
    pad_n = int(np.ceil(pad_seconds * eye.info["sfreq"]))

    # Start padding
    pad_start = np.zeros((eye._data.shape[0], pad_n))
    eye._data = np.concatenate([pad_start, eye._data], axis=1)
    eye._first_samps = np.array([eye._first_samps[0] - pad_n])
    eye._last_samps = np.array([eye._first_samps[0] + eye._data.shape[1] - 1])
    eye_times += pad_seconds

    # End padding
    pad_end = np.zeros((eye._data.shape[0], pad_n))
    eye._data = np.concatenate([eye._data, pad_end], axis=1)
    eye._last_samps = np.array([eye._first_samps[0] + eye._data.shape[1] - 1])

    padded_dur = eye._data.shape[1] / eye.info["sfreq"]
    print(
        f"\n*** Bilateral eye padding applied ***\n"
        f"  Original eye duration: {eye_original_duration:.1f}s\n"
        f"  Padded eye duration:   {padded_dur:.1f}s  "
        f"(+{pad_seconds:.0f}s each side)\n"
        f"  Raw duration:          {raw_duration:.1f}s"
    )

    # Count trial events before alignment for verification
    n_trial_before = len(mne.events_from_annotations(raw, regexp="trial")[0])

    # Realign
    print("\nRealigning eye-tracking data to OPM...")
    mne.preprocessing.realign_raw(raw, eye, raw_times, eye_times, verbose=True)

    n_trial_after = len(mne.events_from_annotations(raw, regexp="trial")[0])
    if n_trial_after < n_trial_before:
        print(
            f"\n*** WARNING: realign_raw removed {n_trial_before - n_trial_after} "
            f"trial events ({n_trial_before} -> {n_trial_after}). ***\n"
        )
    else:
        print(f"  Trial events preserved: {n_trial_before} -> {n_trial_after}")

    return raw, eye, zero_ord_est, first_ord_est, eye_original_duration


def _reset_first_samp(raw_obj: mne.io.Raw) -> mne.io.Raw:
    """Rebuild a Raw with ``first_samp=0``, preserving annotations."""
    ann = raw_obj.annotations
    ann.onset -= raw_obj.first_time
    new_raw = mne.io.RawArray(raw_obj._data, raw_obj.info, first_samp=0, copy="both")
    new_raw.set_annotations(ann)
    return new_raw


def _match_lengths(raw: mne.io.Raw, eye: mne.io.Raw) -> mne.io.Raw:
    """Pad or trim *eye* so it has the same number of samples as *raw*."""
    n_raw = raw._data.shape[1]
    n_eye = eye._data.shape[1]
    if n_eye < n_raw:
        pad_n = n_raw - n_eye
        print(f"Padding eye with {pad_n} zero samples to match raw length.")
        eye._data = np.concatenate(
            [eye._data, np.zeros((eye._data.shape[0], pad_n))], axis=1
        )
        eye._last_samps = np.array([eye._first_samps[0] + eye._data.shape[1] - 1])
    elif n_eye > n_raw:
        print(f"Trimming {n_eye - n_raw} samples from eye to match raw length.")
        eye._data = eye._data[:, :n_raw]
        eye._last_samps = np.array([eye._first_samps[0] + n_raw - 1])
    return eye


def _add_no_eyetrack_annotations(
    raw: mne.io.Raw,
    zero_ord: float,
    first_ord: float,
    eye_original_duration: float,
) -> None:
    """Mark raw regions that lack real eye-tracking data.

    Uses the polynomial mapping to determine where the original eye
    recording maps in raw time, and annotates the complement.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data (annotations modified in-place).
    zero_ord : float
        Polynomial intercept.
    first_ord : float
        Polynomial slope.
    eye_original_duration : float
        Duration of the original (unpadded) eye recording in seconds.
    """
    eye_start_raw = max(0.0, zero_ord)
    eye_end_raw = min(raw.times[-1], zero_ord + first_ord * eye_original_duration)

    print(f"\n--- No-eyetrack annotation boundaries ---")
    print(f"  Real eye data covers {eye_start_raw:.1f}s to {eye_end_raw:.1f}s")
    print(f"  Raw duration: {raw.times[-1]:.1f}s")

    if eye_start_raw > 1.0:
        raw.annotations.append(
            onset=0.0, duration=eye_start_raw, description="no_eyetrack"
        )
        print(f"  Added no_eyetrack: 0.0s to {eye_start_raw:.1f}s (start)")

    if eye_end_raw < raw.times[-1] - 1.0:
        dur = raw.times[-1] - eye_end_raw
        raw.annotations.append(
            onset=eye_end_raw, duration=dur, description="no_eyetrack"
        )
        print(f"  Added no_eyetrack: {eye_end_raw:.1f}s to {raw.times[-1]:.1f}s (end)")


def _set_eyetrack_channel_types(raw: mne.io.Raw) -> None:
    """Drop DIN channel and set proper eyetrack channel types on *raw*."""
    if "DIN" in raw.ch_names:
        raw.drop_channels("DIN")

    eye_channels = {
        "right": {
            "xpos_right": ("eyegaze", "rad", "right", "x"),
            "ypos_right": ("eyegaze", "rad", "right", "y"),
            "pupil_right": ("pupil", "rad", "right"),
        },
        "left": {
            "xpos_left": ("eyegaze", "rad", "left", "x"),
            "ypos_left": ("eyegaze", "rad", "left", "y"),
            "pupil_left": ("pupil", "rad", "left"),
        },
    }
    for side, mapping in eye_channels.items():
        first_ch = list(mapping.keys())[0]
        if first_ch in raw.ch_names:
            mne.preprocessing.eyetracking.set_channel_types_eyetrack(raw, mapping)

    # Head position channels → misc
    for ch_name in _HEAD_POS_CHANNELS:
        if ch_name in raw.ch_names:
            raw.set_channel_types({ch_name: "misc"})


def process_eyetracking(raw: mne.io.Raw, eye_path: str, cfg: SimpleNamespace) -> mne.io.Raw:
    """Full eye-tracking processing pipeline.

    Loads the eye-tracking file, interpolates NaNs, creates feature channels,
    aligns to MEG, and merges into *raw*.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data.
    eye_path : str
        Path to the ``.asc`` eye-tracking file.
    cfg : SimpleNamespace
        Configuration (used for screen calibration parameters).

    Returns
    -------
    raw : mne.io.Raw
        MEG data with eye-tracking channels added.
    """
    print("\n\n\nformatting eyetracker ----------------------\n")

    # 1. Load & calibrate
    eye, cal = _load_eyetracking(eye_path, cfg)

    # 1b. Detect head position channels
    head_pos_chs = _get_head_pos_channels(eye)
    if head_pos_chs:
        print(f"  Head position channels found: {head_pos_chs}")
    else:
        print("  No head position channels found in eye data.")

    # 2. Interpolate NaNs (exclude head-pos from the NaN mask used by NMF)
    orig_nan_mask = _interpolate_nans(eye, exclude_from_mask=head_pos_chs)

    # 3. Create feature channels (NMF/SVD of blink, saccade, NaN)
    _create_eye_feature_channels(eye, orig_nan_mask)

    # 4. Temporal alignment
    raw, eye, zero_ord, first_ord, eye_dur = _align_eyetracking(raw, eye)

    # 5. Reset first_samp to zero
    raw = _reset_first_samp(raw)
    eye = _reset_first_samp(eye)

    # 6. Match sample counts
    eye = _match_lengths(raw, eye)

    # 7. Annotate regions without real eye data
    _add_no_eyetrack_annotations(raw, zero_ord, first_ord, eye_dur)

    # 8. Merge eye channels into raw
    raw.add_channels([eye], force_update_info=True)
    _set_eyetrack_channel_types(raw)

    if head_pos_chs:
        print(f"\nHead position channels added: {head_pos_chs} (type=misc)")

    print(
        "\nupdated raw ----------------------\n",
        raw,
        "\nupdated info ----------------------\n",
        raw.info,
        "\n----------------------\n",
    )

    del eye
    return raw


# ===========================================================================
# BIDS writing helpers
# ===========================================================================


def _write_empty_room(
    cfg: SimpleNamespace, subj: int, emptyroom_path: str
) -> Optional[mne_bids.BIDSPath]:
    """Write empty-room recording to BIDS and return its BIDSPath."""
    raw_er = mne.io.read_raw_fif(emptyroom_path)
    raw_er.info["line_freq"] = getattr(cfg, "line_freq", 60.0)

    bads = getattr(cfg, "bads", [])
    if bads:
        raw_er.info["bads"] = bads

    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=getattr(cfg, "session", "01"),
        task="noise",
        root=cfg.bids_dir,
    )

    mne_bids.write_raw_bids(
        raw_er,
        bids_path,
        allow_preload=True,
        overwrite=True,
        events=None,
        format="FIF",
    )
    return bids_path


def _write_anatomical(
    cfg: SimpleNamespace,
    subj: int,
    t1w_path: Optional[str],
    t2w_path: Optional[str],
) -> None:
    """Write T1w and/or T2w anatomical images to BIDS."""
    session = getattr(cfg, "session", "01")

    for image_path, suffix in [(t1w_path, "T1w"), (t2w_path, "T2w")]:
        if not image_path:
            continue

        bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=session,
            suffix=suffix,
            root=cfg.bids_dir,
        )
        mne_bids.write_anat(
            image=image_path,
            bids_path=bids_path,
            overwrite=True,
            verbose=True,
        )
        print(f"\n  saved {suffix}: {image_path}")


# ===========================================================================
# Main conversion pipeline
# ===========================================================================


def bids_conversion(cfg: SimpleNamespace) -> None:
    """Convert raw OPM-MEG data to BIDS format.

    Orchestrates the full pipeline: validation, empty-room processing,
    task concatenation, trigger conversion, optional eye-tracking integration,
    and BIDS file writing.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration namespace (from ``set_bids_params``).
    """
    subj = cfg.ids
    task = cfg.task

    # --- Validate folder structure ---
    paths = validate_raw_folder(cfg.raw_dir, subj)

    emptyroom_path = paths["emptyroom"]
    task_paths = paths["task"]
    t1w_path = paths["t1w"]
    t2w_path = paths["t2w"]
    eye_path = paths["eye"]

    print(
        f"\nparticipant: {subj}"
        f"\ntask: {task}"
        f"\ndata dir: {cfg.raw_dir}"
        f"\nbids dir: {cfg.bids_dir}"
        f"\ntask paths: {task_paths}"
        f"\nemptyroom path: {emptyroom_path}"
        f"\nT1w path: {t1w_path}"
        f"\nT2w path: {t2w_path}"
        f"\nEye-tracking path: {eye_path}"
        f"\n--------\n\n"
    )

    # --- Empty room ---
    emptyroom_bids_path = None
    if emptyroom_path:
        emptyroom_bids_path = _write_empty_room(cfg, subj, emptyroom_path)

    # --- Read & concatenate task runs ---
    raw_list = []
    for fn in task_paths:
        raw_run = mne.io.read_raw_fif(fn)
        raw_run.info["line_freq"] = getattr(cfg, "line_freq", 60.0)
        raw_run.info["subject_info"] = {
            "id": int(subj),
            "his_id": f"{subj:03}",
        }
        raw_list.append(raw_run)

    raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")
    del raw_list

    # Optional crop
    crop = getattr(cfg, "crop", 0)
    if crop > 0:
        print(f"*****Cropping first {crop} seconds of raw data")
        raw.crop(tmin=crop, tmax=None)
        print(f"raw after cropping: {raw.first_time}")

    duration_min = (raw.times[-1] - raw.times[0]) / 60
    print(
        f"\n\n*************\n"
        f"Recording duration for subject {subj}: {duration_min:.2f} minutes\n"
        f"*************\n\n"
    )

    # --- Annotations ---
    if getattr(cfg, "rename_annot", False):
        raw = convert_triggers(raw, cfg)

    bads = getattr(cfg, "bads", [])
    if bads:
        raw.info["bads"] = bads

    # --- Eye-tracking ---
    if eye_path:
        raw = process_eyetracking(raw, eye_path, cfg)

    # --- Write task data to BIDS ---
    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=getattr(cfg, "session", "01"),
        task=task,
        run="01",
        root=cfg.bids_dir,
    )

    write_kwargs = dict(
        raw=raw,
        bids_path=bids_path,
        allow_preload=True,
        overwrite=True,
        format="FIF",
    )
    if emptyroom_bids_path is not None:
        write_kwargs["empty_room"] = emptyroom_bids_path

    mne_bids.write_raw_bids(**write_kwargs)

    # --- Anatomical images ---
    _write_anatomical(cfg, subj, t1w_path, t2w_path)

    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OPM data to BIDS format")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        help="Path to the Python configuration file",
        default="",
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        type=str,
        default="",
        help="Path to the configuration file (positional, backward compat)",
    )

    args = parser.parse_args()
    config_path = args.config_path or args.config_pos

    print("config path: ", config_path)
    cfg = set_bids_params(config_path)
    bids_conversion(cfg)

    print("\n\n\nDONE!\n\n\n")
