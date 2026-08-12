"""config-trial.py — pipeline configuration for the synthetic OPM subject.

This is a working, end-to-end configuration for the dataset under
``synthetic/datasets/synth``.  It is deliberately shaped like the real
analysis configs in the companion analysis repository (same env-var contract,
same section layout, same custom ``_``-prefixed settings) so that it is a
useful starting point to copy from — but the experimental design is generic:
two stimulus conditions, ``A`` and ``B``, and a left/right button press.

Usage
-----
    sh mne-opm.sh preproc --exp synth --sub 001 --session 01 \\
        --data synthetic/datasets --config synthetic/config --analysis trial

or, driving mne-bids-pipeline directly::

    mne_bids_pipeline --steps=preprocessing --config=synthetic/config/synth/config-trial.py

Environment variables
---------------------
    EXPERIMENT    BIDS task name (``synth``)
    BIDS_DIR      BIDS dataset root
    RAW_DIR       pre-BIDS directory; only the behavioural CSVs are read from it
    SUBJECT       subject ID, e.g. ``001``
    SESSION       session ID, e.g. ``01``
    ANALYSIS      analysis label prefix (``trial``)
    SUBJECTS_DIR  FreeSurfer subjects directory; defaults to the one inside
                  the synthetic BIDS derivatives, so it can be left unset
    MAX_WORKERS   parallel workers (default 1)

Author: Harrison Ritz (2025)
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# ENVIRONMENT
# ============================================================================

EXPERIMENT = os.environ.get("EXPERIMENT", "synth")
SUBJECT = os.environ.get("SUBJECT", "001")
SESSION = os.environ.get("SESSION", "01")

_here = Path(__file__).resolve()
_default_root = _here.parents[2] / "datasets" / EXPERIMENT

BIDS_DIR = os.environ.get("BIDS_DIR") or str(_default_root / "bids")
RAW_DIR = os.environ.get("RAW_DIR") or str(_default_root / "raw")

# The synthetic dataset carries its own FreeSurfer subjects (a phantom "recon"
# plus an `fsaverage` group template), so SUBJECTS_DIR need not be set.
SUBJECTS_DIR = os.environ.get("SUBJECTS_DIR") or str(
    Path(BIDS_DIR) / "derivatives" / "freesurfer" / "subjects"
)


# ============================================================================
# ANALYSIS IDENTITY
# ============================================================================

_version = (
    f"{os.environ.get('SYNTH_SPATIAL_FILTER', 'HFC').lower()}"
    "_filt-p1-40Hz_ica-48_cov-emptyroom"
)
_ANALYSIS_NAME = f"{os.environ.get('ANALYSIS', 'trial')}__{_version}"
print(f"\n[loading configuration]: {_ANALYSIS_NAME}")

# Spatial filter: one of {"maxwell", "HFC", "ZCA", "none"}.
# HFC is the default here because it is the appropriate choice for OPM and it
# is fast.  "maxwell" also works on this dataset -- the array has 144
# magnetometers, comfortably above the 128 basis vectors that
# mf_int_order=10 + mf_ext_order=2 requires -- and so does "ZCA".  Comparing
# them is a common enough thing to want that it is switchable from the
# environment:
#     SYNTH_SPATIAL_FILTER=maxwell bash mne-opm.sh preproc --exp synth ...
_spatial_filter = os.environ.get("SYNTH_SPATIAL_FILTER", "HFC")

custom_proc = "init"
_use_precomputed_trans = False

# Off by default. The report's 3D panels (sensor alignment in
# source/make_forward, source maps) render through VTK and need a display;
# without one the pipeline aborts at make_forward with "Cannot connect to a
# valid display", which is the normal situation in a container or on a cluster
# node. If you have a display -- or Xvfb -- turn this on:
#     xvfb-run -a bash mne-opm.sh preproc ... # with generate_reports = True
generate_reports = False

_trial_conditions = ("trial",)
_response_conditions = ("response/left", "response/right")

# Load the per-trial behavioural CSV shipped with the dataset.
_load_metadata = True

# Bad-channel detection metrics (PCA -> Sidak GESD).
_channel_metrics = ["log_std", "kurtosis", "psd"]
_bad_channel_significance_level = 0.05
_bad_channel_window_sec = 5.0
_bad_channel_lfreq = 1.0
_bad_channel_hfreq = 60.0
_bad_channel_psd_nfft = 512
_bad_channel_lof_neighbors = 16
_bad_channel_emptyroom = False


# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

task = EXPERIMENT

# The generator emits `trial/cond_a` and `trial/cond_b`; the parent label
# `trial` matches both hierarchically, and the sub-conditions are separated
# through metadata queries in the contrasts below.
conditions = ["trial"]

contrasts = [
    {
        # Stimulus condition. The ground-truth simulation puts a stronger left
        # temporal source in condition A and a stronger right parietal source
        # in condition B, so this contrast has a known source-space answer.
        "name": "condAB",
        "conditions": ['condition == "A"', 'condition == "B"'],
        "weights": [0.5, -0.5],
    },
    {
        # Response hand. Correlated with condition but not identical, so it is
        # a genuinely different contrast.
        "name": "responseLR",
        "conditions": ['response == "left"', 'response == "right"'],
        "weights": [0.5, -0.5],
    },
]


# ============================================================================
# EPOCHS
# ============================================================================

epochs_tmin = -0.200
epochs_tmax = 0.600
baseline = (None, 0.0)

# Keep only trials the participant responded to.
epochs_metadata_query = "responded == 1"


# ============================================================================
# METADATA
# ============================================================================
# One row per trial, written by the generator next to the pre-BIDS recording.
# The row count must equal the number of `trial` annotations the pipeline will
# epoch, otherwise mne-bids-pipeline fails late with an opaque
# "Event metadata has N rows, but custom metadata has M rows".

if not _load_metadata:
    epochs_custom_metadata = None
else:
    _pattern = os.path.join(RAW_DIR, f"synth_{SUBJECT}", "metadata", f"sub-{SUBJECT}_*.csv")
    _paths = sorted(glob(_pattern))
    assert _paths, (
        f"No behavioural metadata found.\n"
        f"  Search pattern : {_pattern}\n"
        f"  Check RAW_DIR and SUBJECT, or regenerate with "
        f"src/custom/make_synthetic.py."
    )
    _meta_df = pd.concat([pd.read_csv(p) for p in _paths], ignore_index=True)

    # Cross-check against the events the pipeline will actually epoch.  With
    # custom_proc set, epochs come from raw.annotations in the derivative FIF,
    # so prefer that file when it exists and fall back to the canonical
    # events.tsv otherwise.
    def _count_pipeline_events() -> int | None:
        try:
            import sys

            sys.path.insert(0, str(_here.parents[3] / "src"))
            from custom.preprocessing._io import (  # noqa: PLC0415
                count_condition_events_in_raw,
                count_condition_events_in_tsv,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"    - metadata: event cross-check unavailable ({exc})")
            return None

        # Restrict to the task run: the empty-room derivative sits in the same
        # directory and would otherwise be picked first and report zero trials.
        deriv = sorted(
            Path(BIDS_DIR).glob(
                f"derivatives/{_ANALYSIS_NAME}/sub-{SUBJECT}/ses-{SESSION}/meg/"
                f"*task-{task}_*proc-{custom_proc}*_raw.fif"
            )
        )
        if deriv:
            import mne  # noqa: PLC0415

            raw = mne.io.read_raw_fif(deriv[0], preload=False, verbose="error")
            return count_condition_events_in_raw(raw, _trial_conditions)[0]

        tsv = sorted(
            Path(BIDS_DIR).glob(
                f"sub-{SUBJECT}/ses-{SESSION}/meg/*task-{task}*_events.tsv"
            )
        )
        if tsv:
            return count_condition_events_in_tsv(tsv[0], _trial_conditions)[0]
        return None

    _n_events = _count_pipeline_events()
    if _n_events is not None and _n_events != len(_meta_df):
        raise RuntimeError(
            f"Behavioural metadata has {len(_meta_df)} rows but the pipeline "
            f"will epoch {_n_events} '{'/'.join(_trial_conditions)}' events. "
            f"Regenerate the dataset so the two agree."
        )
    print(f"    - metadata: {len(_meta_df)} trials")

    epochs_custom_metadata = _meta_df
    del _meta_df


# ============================================================================
# PIPELINE CONTROL & PATHS
# ============================================================================

interactive = False
n_jobs = int(os.environ.get("MAX_WORKERS", 1))
process_empty_room = True

bids_root = BIDS_DIR
deriv_root = f"{bids_root}/derivatives/{_ANALYSIS_NAME}"
subjects_dir = SUBJECTS_DIR

subjects = [SUBJECT]
sessions = [SESSION]
ch_types = ["mag"]  # OPM sensors are magnetometers


# ============================================================================
# SPATIAL FILTERING
# ============================================================================

use_maxwell_filter, _do_HFC, _do_ZCA, _st_only = False, False, False, False

if _spatial_filter.lower() == "maxwell":
    use_maxwell_filter = True
elif _spatial_filter.upper() == "HFC":
    _do_HFC = True
elif _spatial_filter.upper() == "ZCA":
    _do_ZCA = True
elif _spatial_filter.lower() != "none":
    raise ValueError(f'Unknown _spatial_filter value: "{_spatial_filter}".')

print(f"    - spatial filter: {_spatial_filter}")

# HFC: order 1 removes a uniform field, order 3 also removes its first- and
# second-order gradients.  The simulation injects both, so order 3 has work
# to do here.
_hfc_order = 3

_zca_method = "zca"
_zca_ext_order = 3
_zca_threshold = 0.50
_gedai_threshold = 0.50

mf_int_order = 10
mf_ext_order = 2
mf_st_duration = 30.0
mf_st_correlation = 0.90
mf_reference_run = "01"
mf_extra_kws = {"ignore_ref": True, "st_overlap": True, "st_only": _st_only}
mf_esss = 0
mf_esss_reject = None
mf_cal_missing = "warn"
mf_ctc_missing = "warn"

find_flat_channels_meg = use_maxwell_filter
find_noisy_channels_meg = use_maxwell_filter
find_bad_channels_extra_kws = {"ignore_ref": True}


# ============================================================================
# CUSTOM FLAGS
# ============================================================================

_manual_channels = False  # no display in a headless dev environment
_bad_ICs = True
_manual_ica = False
_skip_on_deriv = True

_bad_segments_params = {
    "1": {
        "channel_threshold": 0.20,
        "noise_channel_threshold": 0.50,
        "segment_len_sec": 1.0,
    },
    "2": {
        "channel_threshold": 0.05,
        "noise_channel_threshold": 0.30,
        "segment_len_sec": 0.5,
    },
}

# Head-position regressors are shipped as misc channels on the synthetic
# subject; off by default because the simulated drift is uninformative.
_regress = False
_regress_preds = ["x_head", "y_head", "distance"]
_regress_lags = 1


# ============================================================================
# BREAKS, FILTERING, RESAMPLING
# ============================================================================

# The generator inserts a 9 s rest between the two blocks.
find_breaks = True
min_break_duration = 6
t_break_annot_start_after_previous_event = 1.5
t_break_annot_stop_before_next_event = 1.5

l_freq = 0.1
h_freq = 40.0
bandpass_extra_kws = {"fir_window": "blackman"}

notch_freq = None
notch_extra_kws = {"method": "spectrum_fit", "fir_window": "blackman"}

epochs_decim = 2


# ============================================================================
# ARTIFACT REJECTION
# ============================================================================

event_repeated = "drop"
reject = dict(mag=6e-12)


# ============================================================================
# ICA
# ============================================================================

spatial_filter = "ica"
ica_algorithm = "picard-extended_infomax"
ica_l_freq = np.max([l_freq, 3.0])
ica_max_iterations = 512
ica_n_components = 48

ica_decim = 2

# Deliberately loose: the ocular and cardiac artifacts are exactly what ICA
# needs to see, so rejecting the epochs that contain them before fitting would
# be self-defeating. The bad-segment annotations already keep the planted
# interference bursts out of the fit.
ica_reject = dict(mag=5e-11)
ica_ecg_threshold = 0.10
ica_eog_threshold = 3.0

# The synthetic subject carries eye_nmf1-3 as EOG channels (the simulation puts
# real ocular artifact on the sensors), so EOG detection has a genuine target.
# There is no ECG channel; MNE synthesises one from the magnetometers, which is
# enough to catch the simulated cardiac component.
ica_use_eog_detection = True
ica_use_ecg_detection = True
ica_plot_component_properties = "all"

# No corrmap templates ship with the synthetic dataset, so "corrmap_eog" /
# "corrmap_ecg" are left out of _ica_metrics below.
_corrmap_template_dir = None
_n_eog_templates = 0
_n_ecg_templates = 0

# Per-IC scores fed into the unified PCA -> GESD component detection. Set to
# None for all available scores, or [] to disable.
_ica_metrics = [
    "log_hf_ratio",
    "log_line_ratio",
    "temporal_kurtosis_sqrt",
    "autocorr_fisher_z",
    "spectral_slope",
    "spatial_kurtosis_sqrt",
    "log_mean_abs_gradient",
    "eog",
    "ecg",
]

_bad_ICs_alpha = 0.05


# ============================================================================
# DECODING
# ============================================================================
# Off by default: ~40 trials is too few for the result to mean anything, and it
# roughly doubles the runtime.  Turn it on to exercise the code path.

decode = False
_run_decoding = False
decoding_csp = False
decoding_time_generalization = False
_decoder_scoring = "roc_auc"
_decoder_chance = 0.5
_decoder_group_column = "block"
_decoder_contrasts = [c["name"] for c in contrasts]


# ============================================================================
# NOISE COVARIANCE & SOURCE ESTIMATION
# ============================================================================

# The dataset ships an empty-room recording containing the same environmental
# interference and sensor noise as the task run, but no brain.
noise_cov = "emptyroom"

run_source_estimation = True
freesurfer_verbose = False

# oct6 is what the phantom's ico5 cortical surfaces support (4098 sources per
# hemisphere out of 10242 vertices).
spacing = "oct6"
mindist = 5
adjust_coreg = False
inverse_method = "eLORETA"


# ============================================================================
# BEAMFORMER (LCMV)
# ============================================================================

_run_beamformer = True
_beamformer_source_space = ["volume", "surface"]

_beamformer_volume_pos = 8.0
_beamformer_volume_mindist = mindist
_beamformer_volume_bem_conductivity = (0.3,)
_beamformer_volume_bem_ico = 4
_beamformer_volume_cache = True

_beamformer_cov_method = "shrunk"
_beamformer_reg = 0.05
_beamformer_pick_ori = {"volume": "vector", "surface": "max-power"}
_beamformer_weight_norm = {
    "volume": "unit-noise-gain-invariant",
    "surface": "nai",
}
_beamformer_surf_ori = True
_beamformer_depth = None
_beamformer_rank = "data"
_beamformer_rank_tol = 1e-6
_beamformer_rank_tol_kind = "relative"
_reduce_rank = False
_beamformer_output_type = "time"
_beamformer_power_tmin = 0.0
_beamformer_power_tmax = epochs_tmax
_beamformer_save_filters = True
_beamformer_add_to_report = generate_reports
_beamformer_report_n_time_points = 30


# ============================================================================
# COREG DIAGNOSTICS
# ============================================================================

_run_coreg_diagnostics = False
_coreg_diag_run_alignment = True
_coreg_diag_run_bem = True
_coreg_diag_run_headpoint = True
_coreg_diag_run_sensitivity = True
_coreg_diag_output_formats = ["png"]
_coreg_diag_dpi = 120
_coreg_diag_figsize = (8, 8)
_coreg_diag_make_gif = False
_coreg_diag_sensitivity_modes = ["free"]
_coreg_diag_use_nilearn = False
_coreg_diag_bem_conductivity = (0.3,)
_coreg_diag_bem_ico = 4
_coreg_diag_src_spacing = spacing
