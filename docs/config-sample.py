# %% ============================================================
# CONFIGURATION FILE — mne-opm pipeline
# Annotated for NEU502B
#
# This file is read by every run_*.sh script via:
#   export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"
#
# Variables prefixed with _ are custom (pipeline-internal).
# Variables without _ are standard mne_bids_pipeline parameters.
# ================================================================


# %% IMPORTS -----------------------------------------------------
import os
from glob import glob
import pandas as pd
import numpy as np


# %% ============================================================
# SECTION 1: ENVIRONMENT — used by ALL pipeline stages
# Source: run_all.sh / run_anat.sh / run_func.sh / run_preproc.sh etc.
# These are injected as shell environment variables before this
# config is loaded; they locate your data on disk.
# ================================================================

ROOT_DIR    = f"{os.environ.get('ROOT_DIR')}"    # repo root
EXPERIMENT  = f"{os.environ.get('EXPERIMENT')}"  # task name (= BIDS task label)
BIDS_DIR    = f"{os.environ.get('BIDS_DIR')}"    # BIDS root directory
RAW_DIR     = f"{os.environ.get('RAW_DIR')}"     # raw DICOM / unprocessed data
SUBJECTS_DIR = f"{os.environ.get('SUBJECTS_DIR')}"  # FreeSurfer SUBJECTS_DIR
SUBJECT     = f"{os.environ.get('SUBJECT')}"     # current subject ID


# %% ============================================================
# SECTION 2: ANALYSIS IDENTITY — used by ALL pipeline stages
# Controls derivative folder naming and pipeline caching.
# Change _version when you want a clean derivative output folder
# without overwriting prior results.
# ================================================================

_version = "test-online"   # ← bump this to start a fresh derivatives folder
_ANALYSIS_NAME = f"{os.environ.get('ANALYSIS')}__{_version}"

task = EXPERIMENT   # passed to mne_bids_pipeline as the BIDS task name

# mne_bids_pipeline output goes here:
bids_root  = BIDS_DIR
deriv_root = f'{bids_root}/derivatives/{_ANALYSIS_NAME}'
subjects_dir = SUBJECTS_DIR

# Which subjects / sessions / channel types to process
subjects = [SUBJECT]
sessions = ['01']
ch_types = ['mag']   # OPM sensors are magnetometers

# Parallelism: read from environment (set MAX_WORKERS in your launch script)
n_jobs = int(os.environ.get('MAX_WORKERS', 1))
interactive = False         # set True (+ %matplotlib qt) for interactive plots
process_empty_room = True   # preprocess empty-room run alongside task data


# %% ============================================================
# SECTION 3: EXPERIMENT DESIGN — run_preproc.sh / run_sensor.sh
# Defines the events, epoch timing, and trial metadata.
# These are used when epoching the data and computing evoked responses.
# ================================================================

conditions = [
    'standard_onset',   # trigger 1 — 1000 Hz tone
    'deviant_onset',    # trigger 2 — 1200 Hz tone
]

# Contrasts computed at the sensor and source level
# Format: {'name': str, 'conditions': [condition_A, condition_B], 'weights': [1.0, -1.0]}  →  A minus B
contrasts = [
    {'name': 'MMN', 'conditions': ['deviant_onset', 'standard_onset'], 'weights': [1.0, -1.0]}
]

# Epoch window (seconds relative to trigger)
epochs_tmin = -0.5   # pre-stimulus period starts 500 ms before trigger
epochs_tmax =  0.8   # epoch ends 800 ms after trigger

# Baseline correction window (applied to evoked responses)
baseline = (-0.2, 0)   # 200 ms pre-stimulus → trigger onset

# Per-trial metadata CSV (loaded from raw directory)
# Used to attach behavioural variables to each epoch
_metadata_dir = os.path.join(RAW_DIR, f'*_{SUBJECT}', 'metadata', f'sub-{SUBJECT}_*.csv')
_metadata_path = glob(_metadata_dir)
assert len(_metadata_path) > 0, f"no metadata found at {_metadata_dir}"
_meta_df = pd.read_csv(_metadata_path[0])
_meta_df = _meta_df.replace({np.nan: None})
epochs_custom_metadata = _meta_df
del _meta_df

# Optional pandas query to subset epochs after metadata is attached
# e.g. "accuracy == 'correct'" — None keeps all epochs
epochs_metadata_query = None


# %% ============================================================
# SECTION 4: SPATIAL FILTER — run_preproc.sh
# OPM data requires noise suppression before standard preprocessing.
# Only ONE of these methods is active at a time (enforced by assert below).
#
#   maxwell  — Signal Space Separation (SSS); designed for SQUID MEG,
#              requires calibration/cross-talk files; generally NOT used for OPM.
#   HFC      — Homogeneous Field Correction; removes spatially uniform
#              interference (e.g. lab magnetic noise) using reference channels.
#              Recommended for OPM.
#   ZCA      — Zero-phase Component Analysis; data-driven spatial whitening
#              via generalised eigendecomposition. Optional add-on after HFC.
#   none     — No spatial filter (not recommended).
# ================================================================

_spatial_filter = "HFC"   # ← change this to switch spatial filter method

# Flags derived from _spatial_filter — do not edit directly
use_maxwell_filter = False
_do_HFC = False
_do_ZCA = False
if _spatial_filter == "maxwell":
    use_maxwell_filter = True
elif _spatial_filter == "HFC":
    _do_HFC = True
elif _spatial_filter == "ZCA":
    _do_ZCA = True
assert sum([use_maxwell_filter, _do_HFC, _do_ZCA]) <= 1

# HFC settings
_hfc_order = 3           # spherical harmonic order for HFC (default 3)

# ZCA settings (only used if _spatial_filter == "ZCA")
_zca_ext_order  = 3      # external SSS order
_zca_threshold  = 0.99   # GED eigenvalue threshold for component selection

# Maxwell filter settings (only used if _spatial_filter == "maxwell")
mf_int_order = 8         # internal multipole order
mf_ext_order = 3         # external multipole order
mf_st_duration = 30.0    # tSSS buffer duration (sec); None = SSS only
mf_reference_run = '01'
mf_extra_kws = {'ignore_ref': True, 'st_overlap': True}
mf_esss = 0              # extended SSS basis projectors (0 = disabled)
mf_esss_reject = None
mf_cal_missing = "warn"
mf_ctc_missing = "warn"


# %% ============================================================
# SECTION 5: PREPROCESSING — run_preproc.sh
# Standard MNE-BIDS-Pipeline preprocessing parameters.
# Applied after spatial filtering.
# ================================================================

# --- Bad channel / segment detection ---
# Automated flat/noisy channel detection via Maxwell-style algorithm.
# Set to True only if using Maxwell filter (it relies on SSS geometry).
find_flat_channels_meg  = use_maxwell_filter
find_noisy_channels_meg = use_maxwell_filter
find_bad_channels_extra_kws = {'ignore_ref': True}

# Manual bad channel review step (run_channel.sh / run_preproc.sh)
_manual_channels = True   # pause pipeline to inspect bad channels interactively

# Break annotation (mark long inter-trial gaps as BAD_break)
find_breaks = False               # disabled — continuous oddball paradigm
min_break_duration = 6            # min gap (sec) to annotate as a break
t_break_annot_start_after_previous_event = 1.5
t_break_annot_stop_before_next_event     = 1.5

# --- Filtering ---
# Bandpass: 0.1–100 Hz
# A relatively wide band — low end preserves slow drifts without distorting
# ERPs; high end captures gamma and is safely below Nyquist after decimation.
l_freq = 0.1
h_freq = 100.0
bandpass_extra_kws = {'fir_window': 'blackman'}

# Notch filter: removes 60 Hz line noise (US electrical grid)
notch_freq = 60
notch_extra_kws = {'method': 'spectrum_fit', 'fir_window': 'blackman'}

# --- Decimation ---
# Original sample rate is 1200 Hz. Decimate by 3 → 400 Hz.
# Nyquist at 400 Hz = 200 Hz, safely above h_freq = 100 Hz.
epochs_decim = 3

# --- Epoching ---
event_repeated = 'drop'         # if two triggers land on same sample, drop one
reject = dict(mag=5e-12)        # peak-to-peak amplitude rejection threshold (T)

# --- Reference channel regression ---
# Regresses out interference captured by OPM reference channels.
# Applied as a custom step in custom_preproc.py.
_regress_ref             = True
_regress_ref_timevarying = False          # True = time-varying regression weights
_regress_ref_method      = "window"       # sliding-window regression
_regress_ref_window      = 100.0          # window size (ms)
_regress_ref_freqs       = [(None, 5.0)]  # frequency band(s) for regression
_regress_ref_plot        = True

# --- ICA artifact rejection ---
_auto_ica   = True   # run automated ICA component classification
_manual_ica = True   # pause for manual inspection / override of ICA labels

# Skip preprocessing if derivatives already exist (useful for re-running
# downstream steps without redoing slow preprocessing)
_skip_on_deriv = False


# %% ============================================================
# SECTION 6: COREGISTRATION & SOURCE SPACE — run_coreg.sh
# These settings control the MEG–MRI alignment and the construction
# of the cortical surface source space used for forward modeling.
# ================================================================

# Whether to apply a fine-tuning adjustment after automated coregistration.
# Set True if using a template MRI (e.g. fsaverage) rather than individual MRI.
adjust_coreg = False

# Cortical source space density.
# 'oct6' = recursively subdivided octahedron at level 6 → ~4 mm spacing,
# ~8000 source locations per hemisphere. Standard choice for most analyses.
spacing = 'oct6'

# Minimum distance (mm) a source point must be from the inner skull surface.
# Prevents sources from being placed in white matter or outside the cortex.
mindist = 5

# Print full FreeSurfer log output (useful for debugging recon-all issues)
freesurfer_verbose = True

# Uncomment to force BEM surfaces to be recomputed even if they already exist:
# recreate_bem = True


# %% ============================================================
# SECTION 7: SENSOR-LEVEL ANALYSIS — run_sensor.sh
# Controls evoked response, noise covariance, and decoding analyses.
# ================================================================

# Noise covariance source: use the empty-room recording.
# Required for both sensor-level (whitening evoked) and source-level (inverse).
# Also set again in SOURCE SETTINGS below (same value, different context).
noise_cov = 'emptyroom'

# --- MVPA / Decoding ---
decode = True          # run time-by-time and full-epoch decoding (LDA classifier)

# CSP (Common Spatial Patterns) decoding: time–frequency resolved decoding
decoding_csp = True
# Uncomment to define custom frequency bands for CSP:
# decoding_csp_freqs = {
#     'delta': [1, 4],
#     'theta': [4, 8],
#     'alpha': [8, 12],
#     'beta':  [12, 30],
# }

# Uncomment to enable temporal generalisation (train-test across all time pairs):
# decoding_time_generalization = True
# decoding_time_generalization_decim = 6

# Uncomment to enable TFR sensor analysis:
# time_frequency_conditions = conditions
# time_frequency_freq_min = 1
# time_frequency_freq_max = 40


# %% ============================================================
# SECTION 8: BEAMFORMER (LCMV) — run_beamformer.sh
# Controls the LCMV spatial filter source reconstruction.
# The beamformer is the primary source analysis method for this pipeline
# (run_source.sh with dSPM is currently commented out in run_all.sh).
#
# LCMV computes an optimal spatial filter (set of sensor weights) for
# each source location that passes activity from that location while
# suppressing noise and interference from other locations.
# ================================================================

_run_beamformer = True   # master switch; set False to skip entirely

# Regularization applied to the data covariance matrix before inversion.
# Prevents instability when the covariance is rank-deficient.
# Typical range: 0.01–0.10. Higher = more stable but potentially less precise.
_beamformer_reg = 0.01

# Dipole orientation selection:
#   'max-power' — scalar beamformer; finds the dipole orientation with
#                 maximum output power at each location (most common).
#   'vector'    — retains all three orientation components.
#   None        — uses the fixed orientation from the forward model.
_beamformer_pick_ori = 'max-power'

# Weight normalisation:
#   'unit-noise-gain' — normalises by noise, corrects depth bias.
#   'nai'             — Neural Activity Index; good for power comparisons.
#   None              — no normalisation (biased toward superficial sources).
#   Note: use 'unit-noise-gain-invariant' for vector beamformers.
_beamformer_weight_norm = 'nai'

# Depth weighting applied to the forward model (0 = none, 0.8 = standard).
# Recommended: None when using weight_norm, or when computing contrasts
# (depth bias cancels in subtraction). Set 0.8 for single-condition images.
_beamformer_depth = 0.8

# Rank of the data/noise covariance matrices.
# 'info' = infer from channel info (recommended for OPM data).
_beamformer_rank = 'info'

# What to compute:
#   'time'  — time-locked beamformer applied to evoked data (ERP-like STC)
#   'power' — beamformer applied to data covariance (oscillatory power STC)
#   'both'  — run both
_beamformer_output_type = 'time'

# Time window for power beamformer covariance estimation (relative to epoch)
_beamformer_power_tmin = 0.0
_beamformer_power_tmax = epochs_tmax

# Bookkeeping / reporting
_beamformer_save_filters        = True   # save filter weights for reuse
_beamformer_add_to_report       = True   # include STC images in HTML report
_beamformer_report_n_time_points = 51    # number of time frames shown in report


# %% ============================================================
# SECTION 9: SOURCE ANALYSIS (dSPM) — run_source.sh
# Currently commented out in run_all.sh; beamformer is used instead.
# These settings would apply if run_source.sh is re-enabled.
# ================================================================

# Master switch: whether mne_bids_pipeline runs source estimation steps
run_source_estimation = True

# Inverse method: minimum norm family
#   'dSPM'    — dynamic Statistical Parametric Mapping (noise-normalised MNE)
#   'MNE'     — plain minimum norm
#   'sLORETA' — standardised LORETA
#   'eLORETA' — exact LORETA
inverse_method = 'dSPM'

# noise_cov is shared with sensor analysis (defined in Section 7 above)
# Re-stated here for clarity; 'emptyroom' uses the empty-room recording.
noise_cov = 'emptyroom'
