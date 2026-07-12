# %% ============================================================
# CONFIGURATION FILE — mne-opm pipeline (ANNOTATED SAMPLE)
#
# This file is read by every run_*.sh script via:
#   export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"
#
# It is the canonical, fully-annotated reference for EVERY option the
# mne-opm pipeline understands.  Copy it to create a real analysis config,
# then edit the experiment-specific bits (conditions, contrasts, metadata).
#
# Conventions
# -----------
#   * Variables prefixed with _ are custom (read by the mne-opm custom steps).
#   * Variables without _ are standard mne_bids_pipeline parameters.
#
# IMPORTANT (for maintainers)
# ---------------------------
# Whenever a NEW config option is added anywhere in the pipeline, it MUST be
# documented here as well, with a default value and an explanatory comment.
# This sample is the single source of truth for the available options — see
# CLAUDE.md.
# ================================================================


# %% IMPORTS -----------------------------------------------------
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


# %% ============================================================
# SECTION 1: ENVIRONMENT — used by ALL pipeline stages
# These are injected as shell environment variables before this
# config is loaded; they locate your data on disk.
# ================================================================

ROOT_DIR = f"{os.environ.get('ROOT_DIR')}"  # repo root
EXPERIMENT = f"{os.environ.get('EXPERIMENT')}"  # task name (= BIDS task label)
BIDS_DIR = f"{os.environ.get('BIDS_DIR')}"  # BIDS root directory
RAW_DIR = f"{os.environ.get('RAW_DIR')}"  # raw / unprocessed data
SUBJECTS_DIR = f"{os.environ.get('SUBJECTS_DIR')}"  # FreeSurfer SUBJECTS_DIR
SUBJECT = f"{os.environ.get('SUBJECT')}"  # current subject ID


# %% ============================================================
# SECTION 2: ANALYSIS IDENTITY — used by ALL pipeline stages
# Controls derivative folder naming and pipeline caching.
# Bump _version when you want a clean derivative output folder
# without overwriting prior results.
# ================================================================

# The version tag is appended to the analysis name to create a unique
# derivatives folder.  Increment it when changing any parameter that would
# produce different output files (filter settings, epoch window, ICA, etc.)
# so that old results are not silently overwritten.
_version = "sample"  # ← bump this to start a fresh derivatives folder

# Full analysis name: used as the derivatives sub-directory name.
# Format: <ANALYSIS env var>__<version tag>
_ANALYSIS_NAME = f"{os.environ.get('ANALYSIS')}__{_version}"
print(f"\n[loading configuration]: {_ANALYSIS_NAME}")

# Primary spatial filter method.  Exactly one of {"maxwell", "HFC", "ZCA", "none"}.
# Controls which spatial filter is applied during preprocessing.
#   "maxwell" : Maxwell / Signal Space Separation (SSS)
#   "HFC"     : Homogeneous Field Correction
#   "ZCA"     : Zero-phase Component Analysis / GEDAI-based artifact removal
#   "none"    : No spatial filtering (raw sensor data)
_spatial_filter = "HFC"

# Use a precomputed head<->MRI transform (-trans.fif) instead of running
# coregistration.  Set True once you have saved a trans from manual_coreg.
_use_precomputed_trans = False

# Route custom preprocessing outputs through deriv_root using a proc-<label>
# suffix instead of overwriting the raw BIDS files.  When set (e.g. 'init'),
# every custom step (bad_channels, bad_segments, regress, apply_hfc,
# manual_channel, zca_filter) reads from / writes to ``deriv_root`` with
# ``proc-<custom_proc>``.  Leave as None to overwrite the raw BIDS data files.
custom_proc = "init"

# Generate the mne-bids-pipeline HTML reports.  Disable to speed up runs.
generate_reports = False

# Load trial-level behavioral metadata from CSV?
#   True  : load metadata, validate the row count against the pipeline events,
#           expose it as `epochs_custom_metadata`, and enable metadata-dependent
#           contrasts / decoder contrasts.  Required for trial-locked analyses
#           where each epoch corresponds 1:1 with a CSV row.
#   False : skip metadata entirely.  `epochs_custom_metadata` is set to None,
#           and only simple condition-label contrasts are used.  Useful for
#           response-locked analyses where the number of responses does not
#           match the number of metadata rows.
_load_metadata = False

# Opt-in trial/response alignment check (select_trial_response step).
# Set to the name of the metadata column holding the trial-wise response side
# (values like 'left'/'right', case-insensitive; blank / 'n/a' = no response).
# When set (and _load_metadata = True), the step epochs on each trial with
# `keep_first='response'` and asserts the first-response side per trial matches
# this column row-for-row, raising on any misalignment.  None disables the check.
# _response_metadata_column = 'response'


# %% ============================================================
# SECTION 3: BAD CHANNEL DETECTION (bad_channels step)
# Per-channel metrics are combined via PCA -> Šidák GESD.
# ================================================================

# Which metrics to use (None = all available, [] = disable detection).
# Available: "log_std", "logit_outlier_frac", "kurtosis", "lof", "psd".
_channel_metrics = [
    "log_std",
    "logit_outlier_frac",
    "kurtosis",
    "lof",
    "psd",
]
_bad_channel_significance_level = 0.05  # family-wise GESD alpha (+ per-window alpha)
_bad_channel_window_sec = 2.0  # window length (s) for logit_outlier_frac
_bad_channel_psd_fmin = 1.0  # PSD metric frequency band (Hz)
_bad_channel_psd_fmax = 80.0
_bad_channel_psd_nfft = 2000  # PSD metric FFT length
_bad_channel_lof_neighbors = 16  # LOF neighbours


# %% ============================================================
# SECTION 4: EXPERIMENT DESIGN — run_preproc.sh / run_sensor.sh
# Defines the events, epoch timing, and trial metadata.
#
# NOTE: conditions / contrasts below are EXAMPLES (an oddball MMN paradigm).
# Replace them with the events and contrasts for your own experiment.
# ================================================================

# Task name; must match the BIDS task entity in the file names.
task = EXPERIMENT

# Epoch conditions to include.  Matched against the 'trial_type' column in the
# BIDS events.tsv file.
conditions = [
    "standard_onset",  # EXAMPLE — trigger 1
    "deviant_onset",  # EXAMPLE — trigger 2
]

# Contrasts define linear combinations of epoch sub-populations for evoked and
# sensor-space decoding analyses.  Each contrast specifies:
#   name       : unique identifier (used in file names and reports)
#   conditions : list of condition labels (or metadata query strings)
#   weights    : linear weights summing to 0 (positive = "signal")
contrasts = [
    {
        "name": "MMN",
        "conditions": ["deviant_onset", "standard_onset"],
        "weights": [1.0, -1.0],
    },
]


# %% ============================================================
# SECTION 5: EPOCHS — run_preproc.sh / run_sensor.sh
# ================================================================

# Time window around the locking event (seconds).
epochs_tmin = -0.5  # pre-stimulus baseline start
epochs_tmax = 0.8  # post-stimulus end

# Pre-stimulus baseline correction applied to evoked responses.
#   None              : no baseline correction
#   (tmin, 0)         : use the full pre-stimulus period
#   (-0.2, 0)         : use a restricted pre-stimulus window
baseline = (-0.2, 0)

# Optional pandas query applied after epochs are created to restrict which
# epochs enter the evoked / decoding analyses (operates on the metadata
# DataFrame).  None keeps all epochs.
# epochs_metadata_query = None

# regexp matched against eye-tracking annotations to find sync events
_eye_sync_regex = "stim_onset"
# regexp matched against MEG annotations to find sync events
_raw_sync_regex = "trial"


# %% ============================================================
# SECTION 6: METADATA (optional) — gated by _load_metadata
# Trial-level behavioral metadata is loaded from CSV and attached to the
# epochs as `epochs_custom_metadata`, enabling pandas-query contrasts.
# This is a SIMPLE loader; real analyses may derive extra columns here.
# ================================================================

if not _load_metadata:
    epochs_custom_metadata = None
    print("    - metadata: skipped (_load_metadata=False)")
else:
    _metadata_dir = os.path.join(
        RAW_DIR, f"*_{SUBJECT}", "metadata", f"sub-{SUBJECT}_*.csv"
    )
    _metadata_path = glob(_metadata_dir)
    assert len(_metadata_path) > 0, f"no metadata found at {_metadata_dir}"
    _meta_df = pd.read_csv(sorted(_metadata_path)[0])
    _meta_df = _meta_df.replace({np.nan: None})
    epochs_custom_metadata = _meta_df
    print(
        f"    - metadata ready: {len(_meta_df)} rows, {len(_meta_df.columns)} columns"
    )
    del _meta_df


# %% ============================================================
# SECTION 7: PIPELINE CONTROL & BIDS PATHS — used by ALL stages
# ================================================================

# Run pipeline in non-interactive (batch) mode.
# Set True (+ %matplotlib qt) only when running interactively.
interactive = False

# Number of parallel workers.  Reads from MAX_WORKERS env var; defaults to 1.
n_jobs = int(os.environ.get("MAX_WORKERS", 1))

# Process the empty-room recording through the same preprocessing steps as
# the experimental data.  Required when noise_cov = 'emptyroom'.
process_empty_room = True

# mne_bids_pipeline output goes here:
bids_root = BIDS_DIR
deriv_root = f"{bids_root}/derivatives/{_ANALYSIS_NAME}"
subjects_dir = SUBJECTS_DIR

# Which subjects / sessions / channel types to process
subjects = [SUBJECT]
sessions = ["01"]
ch_types = ["mag"]  # OPM sensors are magnetometers


# %% ============================================================
# SECTION 8: SPATIAL FILTER — run_preproc.sh
# Only ONE method is active at a time (enforced by the assert below).
#
#   HFC     — recommended; removes low-order external fields without a
#             calibration file or reference array.
#   ZCA     — data-driven noise suppression via GED; good for high channel count.
#   maxwell — designed for SQUID MEG; generally NOT recommended for OPM.
#   none    — skip spatial filtering (useful for debugging).
# ================================================================

# Flags derived from _spatial_filter — do not edit directly.
use_maxwell_filter, _do_HFC, _do_ZCA = False, False, False
if _spatial_filter.lower() == "maxwell":
    use_maxwell_filter = True
elif _spatial_filter.upper() == "HFC":
    _do_HFC = True
elif _spatial_filter.upper() == "ZCA":
    _do_ZCA = True
elif _spatial_filter.lower() != "none":
    raise ValueError(
        f'Unknown _spatial_filter value: "{_spatial_filter}". '
        f'Must be one of: "maxwell", "HFC", "ZCA", "none".'
    )
print(f"    - spatial filter: {_spatial_filter}")

# -- HFC parameters --
# Spherical harmonic order for homogeneous field correction.
# Order 1 = uniform field; order 3 also removes first-order gradients.
_hfc_order = 3

# -- ZCA parameters (only used when _spatial_filter == "ZCA") --
_zca_method = "zca"  # toggle between 'zca' and 'gedai'
_zca_ext_order = 3  # external SSS order used to form the GED noise basis
_zca_threshold = 0.50  # GED eigenvalue threshold for ZCA component selection
_gedai_threshold = 0.50  # absolute eigenvalue cutoff for GEDAI artifact identification

# -- Maxwell filter parameters (only used when _spatial_filter == "maxwell") --
mf_int_order = 8  # internal multipole order
mf_ext_order = 3  # external multipole order
mf_st_duration = 60.0  # tSSS sliding-window duration (s); None = SSS only
mf_st_correlation = 0.95  # tSSS subspace correlation threshold
mf_reference_run = "01"  # reference run for multi-run head-position alignment
mf_extra_kws = {"ignore_ref": True, "st_overlap": True}
mf_esss = 0  # extended SSS basis projectors (0 = disabled)
mf_esss_reject = None
mf_cal_missing = "warn"  # how to handle a missing calibration file
mf_ctc_missing = "warn"  # how to handle a missing cross-talk file

# Built-in flat / noisy channel detection is only meaningful with Maxwell
# filtering (SSS can interpolate over bad channels).  Disabled for HFC/ZCA.
find_flat_channels_meg = use_maxwell_filter
find_noisy_channels_meg = use_maxwell_filter
find_bad_channels_extra_kws = {"ignore_ref": True}


# %% ============================================================
# SECTION 9: CUSTOM STEP FLAGS — run_preproc.sh / run_channel.sh
# Read by the custom pipeline hooks to control interactive review
# and caching behaviour.
# ================================================================

_manual_channels = False  # pause to inspect bad channels interactively
_auto_ica = True  # run automated ICA component classification
_manual_ica = False  # pause for manual inspection / override of ICA labels

# Skip a pipeline step if its derivatives folder already exists (resume runs).
_skip_on_deriv = True


# %% ============================================================
# SECTION 10: BAD SEGMENT DETECTION (STAGED) — bad_segments step
# Per-stage parameters for bad_segments_1 and bad_segments_2.
# Detection runs on a bandpass-filtered copy (l_freq / h_freq) but the
# annotations are written back to the unfiltered raw data.
#
#   Stage 1 (before spatial filter): gross artifacts — lenient threshold,
#           longer window.
#   Stage 2 (after spatial filter): finer transients — stricter threshold,
#           shorter window.
#
#   channel_threshold       : fraction of channels that must be outliers before
#                             a segment is marked bad (higher = more lenient).
#   noise_channel_threshold : same, applied to the empty-room recording.
#   segment_len_sec         : sliding-window length (s).
# ================================================================

_bad_segments_params = {
    "1": {
        "channel_threshold": 0.20,  # lenient — only gross artifacts
        "noise_channel_threshold": 0.50,  # very lenient for noise
        "segment_len_sec": 1.0,  # 1 s window (coarse)
    },
    "2": {
        "channel_threshold": 0.05,  # strict — fine cleanup
        "noise_channel_threshold": 0.30,  # moderate for noise
        "segment_len_sec": 0.5,  # 0.5 s window (fine)
    },
}


# %% ============================================================
# SECTION 11: REFERENCE / NUISANCE REGRESSION (regress step)
# Regress out nuisance predictors from the data before decoding.
# ================================================================

_regress = False  # master switch
_regress_preds = ["x_head", "y_head", "distance"]  # predictor channels / signals
_regress_lags = 1  # delay-embedding lags (0 = none)
# The regress step also supports (optional, defaults shown):
# _regress_timevarying = False   # sliding-window (time-varying) regression weights
# _regress_window      = 100.0   # window size (ms) for time-varying regression
# _regress_freqs       = None    # frequency band(s) to filter predictors, e.g. [(None, 5.0)]
# _regress_plot        = False   # save before/after diagnostic plots


# %% ============================================================
# SECTION 12: PREPROCESSING — BREAKS, FILTERING, RESAMPLING
# ================================================================

# -- Rest / break detection --
find_breaks = True
min_break_duration = 6  # minimum break length (s)
t_break_annot_start_after_previous_event = 1.5  # buffer after last event (s)
t_break_annot_stop_before_next_event = 1.5  # buffer before next event (s)

# -- Bandpass filter (applied to raw before epoching) --
l_freq = 1.0  # high-pass cut-off (Hz)
h_freq = 40.0  # low-pass cut-off (Hz)
# Blackman window — excellent stopband attenuation, minimal edge artefacts.
bandpass_extra_kws = {"fir_window": "blackman"}

# -- Notch filter --
# Set to a frequency (or list) to remove line noise; None = skip (preferred for
# OPM when line noise is low).
notch_freq = None
notch_extra_kws = {"method": "spectrum_fit", "fir_window": "blackman"}

# Zapline (notch via DSS) — alternative to the MNE notch filter. Uncomment to use.
# zapline_fline = 60.0
# zapline_iter  = True

# -- Downsampling --
# Decimate epochs by this factor after epoching (e.g. 1200 Hz / 3 = 400 Hz).
epochs_decim = 3


# %% ============================================================
# SECTION 13: ARTIFACT REJECTION — run_preproc.sh
# ================================================================

# How to handle trials with duplicate events ("drop" or "merge").
event_repeated = "drop"

# Peak-to-peak amplitude threshold for magnetometers (Tesla); epochs exceeding
# this value are rejected.
reject = dict(mag=5e-12)


# %% ============================================================
# SECTION 14: ICA — run_preproc.sh
# Components correlated with ECG/EOG (and other diagnostic metrics) are
# automatically labelled as artifacts via a unified PCA -> GESD step.
# ================================================================

# Apply ICA as the primary artifact-rejection step (rather than SSP).
spatial_filter = "ica"

# ICA decomposition algorithm.
#   'picard-extended_infomax' : fast, robust; recommended for MEG.
#   'picard' / 'fastica' / 'infomax' : alternatives.
ica_algorithm = "picard-extended_infomax"

# High-pass filter applied before ICA fitting (Hz). Must be >= l_freq.
ica_l_freq = np.max([l_freq, 1.0])

ica_max_iterations = 1024  # maximum ICA fitting iterations
ica_n_components = 64  # number of components (<= data rank)
ica_decim = epochs_decim  # decimation during ICA fitting
ica_reject = dict(mag=5e-12)  # peak-to-peak threshold for ICA-fit data
ica_ecg_threshold = 0.10  # correlation threshold for cardiac components
ica_eog_threshold = 3.0  # z-score threshold for ocular components

# Disable mne-bids-pipeline's built-in EOG/ECG detection — ALL ICA component
# selection is performed by the unified GESD in the custom auto_ica step, where
# the EOG/ECG correlations are included as scores.
ica_use_eog_detection = False
ica_use_ecg_detection = False

# Which ICA components have their properties plotted as PNGs.
#   "all"      : plot every component.
#   "excluded" : only plot excluded components.
ica_plot_component_properties = "all"

# Corrmap template parameters (used when "corrmap_eog"/"corrmap_ecg" appear in
# _ica_metrics): template directory and number of EOG/ECG template columns each
# component topography is correlated against.
_corrmap_template_dir = str(Path(__file__).resolve().parent / "ICA")
_n_eog_templates = 3
_n_ecg_templates = 0

# Per-IC scores fed into the unified PCA -> GESD ICA component detection.
# All are z-scored, projected onto principal components, and GESD-tested per
# eigenscore under one Šidák family-wise error rate.
# None = use ALL available scores; [] = disable the GESD entirely.
# Diagnostic metrics: "log_hf_ratio", "log_line_ratio", "temporal_kurtosis_sqrt",
#   "autocorr_fisher_z", "spectral_slope", "spatial_kurtosis_sqrt",
#   "spectral_deriv_kurtosis_sqrt", "spectral_resid_kurtosis_sqrt",
#   "log_mean_abs_gradient".
# Targeted scores: "eog", "ecg", "reference", "corrmap_eog", "corrmap_ecg".
_ica_metrics = [
    "log_hf_ratio",
    "log_line_ratio",
    "temporal_kurtosis_sqrt",
    "autocorr_fisher_z",
    "spectral_slope",
    "spatial_kurtosis_sqrt",
    "spectral_deriv_kurtosis_sqrt",
    "log_mean_abs_gradient",
    "eog",
    "ecg",
    "reference",
    "corrmap_eog",
]


# %% ============================================================
# SECTION 15: SENSOR-SPACE ANALYSIS — DECODING (mne-bids-pipeline)
# ================================================================

# Enable multivariate pattern analysis (MVPA / decoding).
decode = True

# Time-generalisation matrix (train at each time, test at all times).
decoding_time_generalization = False
decoding_time_generalization_decim = 5  # extra temporal decimation for the TGM

# Leave-One-Group-Out (LOGO) cross-validation — holds out one group at a time,
# preventing temporal autocorrelation from inflating decoding accuracy.
decoding_LOGO = True
decoding_LOGO_group = "run"  # metadata column defining the CV group

# Baseline period applied to epochs before decoding ((None, 0.0) = full
# pre-stimulus window; None = no baseline).
_decoding_baseline = None

# Equalise the number of trials across conditions before decoding.
_decoding_equalize = True

# Regularisation parameter grid for the sensor-space decoder (SVM).
_decoder_epoch_C_grid = [0.001, 0.01, 0.1, 1, 10]

# Time points (s) at which to extract decoder patterns.
_decoder_pattern_times = np.arange(epochs_tmin, epochs_tmax, 0.1)

# Common Spatial Patterns (CSP) frequency-band decoding.
decoding_csp = True
# Uncomment to define custom CSP frequency bands:
# decoding_csp_freqs = {
#     'delta': [1,  4],
#     'theta': [4,  8],
#     'alpha': [8,  12],
#     'beta':  [12, 30],
# }

# Uncomment to enable TFR sensor analysis:
# time_frequency_conditions = conditions
# time_frequency_freq_min = 1
# time_frequency_freq_max = 40


# %% ============================================================
# SECTION 16: NOISE COVARIANCE
# Used by source estimation and the LCMV beamformer to whiten / regularise.
#   'emptyroom' : estimate from the empty-room recording (recommended).
#   'rest'      : estimate from a resting-state recording.
#   (tmin,tmax) : estimate from a pre-stimulus baseline window.
#   'ad-hoc'    : diagonal ad-hoc covariance (no recording needed).
# ================================================================

noise_cov = "emptyroom"


# %% ============================================================
# SECTION 17: SOURCE ESTIMATION (dSPM family) — run_source.sh
# ================================================================

# Master switch: whether mne_bids_pipeline runs source estimation steps.
run_source_estimation = True

# Print verbose FreeSurfer output during BEM / source-space construction.
freesurfer_verbose = True

# Source-space type and resolution.
#   'oct5' ~8 mm (coarse) | 'oct6' ~4 mm (standard) | 'oct7' ~2 mm (fine, slow)
spacing = "oct6"

# Minimum distance (mm) between sources and the inner skull surface.
mindist = 5

# Automatically adjust the head<->MRI coregistration (use with template MRIs).
adjust_coreg = False

# Distributed inverse method: 'dSPM' | 'sLORETA' | 'eLORETA' | 'MNE'.
inverse_method = "dSPM"


# %% ============================================================
# SECTION 18: BEAMFORMER (LCMV) — run_beamformer.sh
# Reconstructs source activity with a spatial filter that passes a target
# location while suppressing all others.  Well-suited to OPM-MEG.
# ================================================================

_run_beamformer = True  # master switch

# Source space for the beamformer forward model.
#   'surface' : cortical-surface source space built by mne-bids-pipeline
#               (uses SECTION 17 `spacing`/`mindist`; per-hemisphere STCs saved
#               as `*+lcmv+hemi-stc.h5`).  DEFAULT — backward-compatible.
#   'volume'  : regular 3D grid inside the inner skull, built on the fly by
#               run_beamformer.py via mne.setup_volume_source_space; the volume
#               forward is cached as `*_acq-vol_fwd.fif` and STCs are saved as
#               `*+lcmv+vol-vl.h5` (rendered with nilearn, not the surface Brain).
_beamformer_source_space = "surface"

# --- Volume-source-space options (only used when source_space == 'volume') ---
# Grid spacing (mm) of the volume source space.  Smaller = finer and slower.
_beamformer_volume_pos = 5.0
# Minimum distance (mm) of grid points from the inner skull (defaults to `mindist`).
_beamformer_volume_mindist = mindist
# BEM used to build the volume forward when none is cached on disk.  For MEG-only
# OPM data a single-shell conductivity `(0.3,)` is appropriate; `ico` sets the
# surface tessellation when the BEM model has to be built from FreeSurfer surfaces.
_beamformer_volume_bem_conductivity = (0.3,)
_beamformer_volume_bem_ico = 4
# Cache the built volume forward to `*_acq-vol_fwd.fif` for reuse across reruns.
_beamformer_volume_cache = True

# Regularisation added to the data covariance before inversion (0.01–0.10).
_beamformer_reg = 0.05

# Dipole orientation selection.
#   'max-power' : optimise orientation for maximum power (scalar beamformer)
#   'vector'    : return all three dipole orientations (vector beamformer)
#   None        : fixed orientation from the forward model
_beamformer_pick_ori = "max-power"

# Weight normalisation.
#   'unit-noise-gain'           : corrects depth bias (scalar)
#   'nai'                       : Neural Activity Index (scalar)
#   'unit-noise-gain-invariant' : orientation-invariant (vector only)
#   None                        : no normalisation
_beamformer_weight_norm = "nai"

# Depth-bias compensation via forward-model weighting (0.0 none | 0.8 standard
# | None when weight_norm is set).  Cancels out for two-condition contrasts.
_beamformer_depth = 0.8

# Data rank for covariance estimation/regularisation.
#   'info' : infer from the Info object (recommended)
#   dict   : explicit per-channel-type rank, e.g. {'mag': 64}
#   None   : auto-detect via SVD
_beamformer_rank = "info"

# What the beamformer operates on.
#   'time'  : evoked response (time-domain output)
#   'power' : data covariance (power / envelope output)
#   'both'  : both
_beamformer_output_type = "time"

# Time window for power-mode covariance estimation (s, relative to epoch onset).
_beamformer_power_tmin = 0.0
_beamformer_power_tmax = epochs_tmax

# Bookkeeping / reporting.
_beamformer_save_filters = True  # persist filter weights for re-use
_beamformer_add_to_report = True  # include source maps in the HTML report
_beamformer_report_n_time_points = 51  # frames in the report source-map animation


# %% ============================================================
# SECTION 19: COREG DIAGNOSTICS — run_coreg_diagnostics.sh
# Source: src/custom/coreg_diagnostics.py
# Produces BEM, alignment, dig->scalp distance and sensitivity-map figures
# under {deriv_root}/sub-XX/ses-YY/meg/coreg_diagnostics/.  No GUI required.
# ================================================================

_run_coreg_diagnostics = True  # master switch

# Per-section toggles.
_coreg_diag_run_alignment = True
_coreg_diag_run_bem = True
_coreg_diag_run_headpoint = True
_coreg_diag_run_sensitivity = True

# Output formats.  3D figures are always saved as PNG regardless of this list.
_coreg_diag_output_formats = ["png"]  # any subset of {'png','pdf','svg'}
_coreg_diag_dpi = 200
_coreg_diag_figsize = (10, 10)

# Alignment views to render — keys of the _VIEWS dict in coreg_diagnostics.py.
# Available: 'frontal', 'posterior', 'lateral_left', 'lateral_right',
# 'superior', 'oblique'.
_coreg_diag_alignment_views = [
    "frontal",
    "posterior",
    "lateral_left",
    "lateral_right",
    "superior",
    "oblique",
]

# 360° rotating-azimuth GIF of the alignment scene.  Off by default — slow.
_coreg_diag_make_gif = False

# mne.sensitivity_map(...) modes to compute, per ch_type in cfg.ch_types.
_coreg_diag_sensitivity_modes = ["free", "radiality"]

# Use nilearn for richer BEM-on-T1 contour overlays.
_coreg_diag_use_nilearn = True

# On-the-fly forward-solution fallback (only used if no *-fwd.fif is found).
_coreg_diag_bem_conductivity = (0.3,)  # single-shell, appropriate for MEG
_coreg_diag_bem_ico = 4
_coreg_diag_src_spacing = "oct6"


# %% ============================================================
# SECTION 20: MULTIVARIATE DECODING (CUSTOM) — run_decoding.sh
# SEPARATE from the built-in mne-bids-pipeline decoding above.
# Uses its own contrasts (metadata query strings) for binary classification
# with Leave-One-Group-Out cross-validation.
# ================================================================

_run_decoding = True  # master switch

# Scoring metric ('roc_auc' or 'accuracy').
_decoder_scoring = "roc_auc"

# Parallel jobs inside the SlidingEstimator (per time-step).
_decoder_n_jobs_inner = n_jobs

# Temporal decimation for time-resolved decoding (1 = none).
_decoder_decim = 4

# Run temporal generalization (train-time x test-time matrix)?
_decoder_run_temporal_gen = True

# Baseline correction applied before decoding (None or (tmin, tmax)).
_decoder_baseline = None

# Chance level for the scoring metric (0.5 for ROC-AUC binary classification).
_decoder_chance = 0.5

# File formats for per-subject diagnostic plots ('png','tiff','pdf','eps').
_decoder_save_formats = ["png"]

# Metadata column used for Leave-One-Group-Out CV grouping (typically 'run').
_decoder_group_column = "run"

# PCA n_components for each decoding pipeline.
#   "rank" -> data rank from mne.compute_rank (recommended for per-time)
#   int    -> fixed number of components
#   float  -> fraction of variance to retain (0 < x <= 1.0)
_decoder_time_n_components = 0.99  # per-time-step pipeline (SlidingEstimator / TG)
_decoder_epoch_n_components = 0.99  # full-epoch pipeline (Vectorizer -> PCA -> SVC)

# Binary classification contrasts for the custom decoder.  Each entry needs:
#   name       : unique identifier (used in BIDS file names)
#   conditions : list of exactly 2 metadata query strings
# EXAMPLE (replace with your own); requires _load_metadata = True.
_decoder_contrasts = [
    {"name": "MMN", "conditions": ["deviant_onset", "standard_onset"]},
]

# Cross-decoding contrasts (train on one condition set, test on another).
# Each entry needs 'name', 'train', 'test', and 'analyses' (subset of
# ['time', 'tg', 'epoch']).  Empty by default.
_decoder_cross_contrasts = []

# Without metadata the decoder has nothing to query, so clear both lists.
if not _load_metadata:
    _decoder_contrasts = []
    _decoder_cross_contrasts = []


# %% ============================================================
# SECTION 21: VALIDATION
# Collected here so misconfigured runs fail early with a clear message.
# ================================================================

# -- Required environment variables --
_required_env = {
    "EXPERIMENT": EXPERIMENT,
    "BIDS_DIR": BIDS_DIR,
    "RAW_DIR": RAW_DIR,
    "SUBJECTS_DIR": SUBJECTS_DIR,
    "SUBJECT": SUBJECT,
}
_missing_env = [k for k, v in _required_env.items() if not v or v == "None"]
assert not _missing_env, (
    f"The following required environment variables are not set: {_missing_env}\n"
    f"Export them before calling mne_bids_pipeline."
)

# -- Epoch time window --
assert epochs_tmin < epochs_tmax, (
    f"epochs_tmin ({epochs_tmin}) must be < epochs_tmax ({epochs_tmax})."
)

# -- Bandpass filter --
if l_freq is not None and h_freq is not None:
    assert l_freq < h_freq, f"l_freq ({l_freq}) must be < h_freq ({h_freq})."

# -- ICA high-pass must be >= bandpass high-pass --
if l_freq is not None:
    assert ica_l_freq >= l_freq, (
        f"ica_l_freq ({ica_l_freq}) should be >= l_freq ({l_freq})."
    )

# -- Spatial filter exclusivity --
assert sum([use_maxwell_filter, _do_HFC, _do_ZCA]) <= 1, (
    "More than one spatial filter is active.  Check _spatial_filter."
)

# -- Decimation --
assert epochs_decim >= 1 and isinstance(epochs_decim, int), (
    f"epochs_decim must be a positive integer, got {epochs_decim}."
)

# -- ICA parameters --
assert ica_n_components > 0, (
    f"ica_n_components must be positive, got {ica_n_components}."
)
assert ica_max_iterations > 0, (
    f"ica_max_iterations must be positive, got {ica_max_iterations}."
)
assert 0 < ica_ecg_threshold <= 1, (
    f"ica_ecg_threshold should be in (0, 1], got {ica_ecg_threshold}."
)
assert ica_eog_threshold > 0, (
    f"ica_eog_threshold must be positive, got {ica_eog_threshold}."
)

# -- Contrasts (weights must sum to 0) --
for _c in contrasts:
    assert len(_c["weights"]) == len(_c["conditions"]), (
        f"Contrast '{_c['name']}': number of weights must equal number of conditions."
    )
    assert abs(sum(_c["weights"])) < 1e-10, (
        f"Contrast '{_c['name']}': weights must sum to 0 (got {sum(_c['weights']):.6f})."
    )

# -- Beamformer --
assert 0.0 <= _beamformer_reg <= 1.0, (
    f"_beamformer_reg must be in [0.0, 1.0], got {_beamformer_reg}."
)
assert _beamformer_pick_ori in {"max-power", "vector", None}, (
    f"_beamformer_pick_ori must be 'max-power', 'vector', or None; got '{_beamformer_pick_ori}'."
)
_valid_weight_norms = {"unit-noise-gain", "nai", "unit-noise-gain-invariant", None}
assert _beamformer_weight_norm in _valid_weight_norms, (
    f"_beamformer_weight_norm must be one of {_valid_weight_norms}; got '{_beamformer_weight_norm}'."
)
if _beamformer_pick_ori == "vector":
    assert _beamformer_weight_norm in {"unit-noise-gain-invariant", None}, (
        "For vector beamformers, use weight_norm='unit-noise-gain-invariant' (or None)."
    )
assert _beamformer_power_tmin < _beamformer_power_tmax, (
    f"_beamformer_power_tmin ({_beamformer_power_tmin}) must be < "
    f"_beamformer_power_tmax ({_beamformer_power_tmax})."
)
assert _beamformer_source_space in {"surface", "volume"}, (
    f"_beamformer_source_space must be 'surface' or 'volume'; got '{_beamformer_source_space}'."
)
assert _beamformer_volume_pos > 0, (
    f"_beamformer_volume_pos must be > 0 (mm), got {_beamformer_volume_pos}."
)
assert _beamformer_volume_mindist >= 0, (
    f"_beamformer_volume_mindist must be >= 0 (mm), got {_beamformer_volume_mindist}."
)

# -- Source estimation --
_valid_inverse_methods = {"dSPM", "sLORETA", "eLORETA", "MNE"}
assert inverse_method in _valid_inverse_methods, (
    f"inverse_method must be one of {_valid_inverse_methods}; got '{inverse_method}'."
)

# -- Custom decoder --
if _run_decoding:
    assert isinstance(_decoder_contrasts, list), (
        "_decoder_contrasts must be a list of dicts."
    )
    for _dc in _decoder_contrasts:
        assert "name" in _dc and "conditions" in _dc, (
            f"Each _decoder_contrasts entry must have 'name' and 'conditions'. Got: {list(_dc.keys())}"
        )
        assert len(_dc["conditions"]) == 2, (
            f"Decoder contrast '{_dc['name']}': must have exactly 2 conditions, got {len(_dc['conditions'])}."
        )
    assert isinstance(_decoder_cross_contrasts, list), (
        "_decoder_cross_contrasts must be a list of dicts."
    )
    for _dcc in _decoder_cross_contrasts:
        assert all(k in _dcc for k in ("name", "train", "test", "analyses")), (
            f"Each _decoder_cross_contrasts entry must have 'name', 'train', 'test', 'analyses'. Got: {list(_dcc.keys())}"
        )
        assert all(a in ("time", "tg", "epoch") for a in _dcc["analyses"]), (
            f"Cross-contrast '{_dcc['name']}': analyses must be subset of ['time','tg','epoch']. Got: {_dcc['analyses']}"
        )
    assert _decoder_decim >= 1 and isinstance(_decoder_decim, int), (
        f"_decoder_decim must be a positive integer, got {_decoder_decim}."
    )
    assert _decoder_scoring in ("roc_auc", "accuracy"), (
        f"_decoder_scoring must be 'roc_auc' or 'accuracy', got '{_decoder_scoring}'."
    )

print(f"    - configuration loaded and validated: {_ANALYSIS_NAME}")
