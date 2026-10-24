# Preprocessing walkthrough

This tutorial explains each preprocessing step in the pipeline, what it does,
and how to configure it.

## Pipeline order

The `run_preproc.sh` script runs these steps in sequence:

```text
init -> annotate_breaks -> select_trial_response
    -> bad_segments_1 -> bad_channels -> manual_channel
    -> apply_hfc -> apply_zca -> MNE preprocessing -> ica_convergence -> bad_ICs
    -> manual_ica -> MNE ICA apply -> bad_epochs
```

`annotate_breaks` saves `BAD_break` annotations to proc-init before segment
and channel scoring, for every analysis whose config sets `find_breaks`. It
exports `MNE_OPM_BREAKS_ALREADY_ANNOTATED=1` on success (and `bad_segments_1`,
the fallback, exports `MNE_OPM_STAGE1_BREAKS_SAVED=1`); a config should turn
its own `find_breaks` off when either is set so mne-bids-pipeline does not add
a duplicate copy. The first bad-segment pass uses conservative settings before
Maxwell filtering; the standard wrapper does not run stage 2.

Each custom step is a module in `custom.preprocessing` with a `run(cfg)`
function. They can be run individually via:

```bash
python src/custom/custom_preproc.py --analysis=<step> --config=/path/to/config.py
```

## Bad channel detection (`bad_channels`)

Automatically identifies bad channels using the Generalized Extreme Studentized
Deviate (GESD) test from osl-ephys. Channels with abnormal variance, kurtosis,
or correlation are flagged.

**Always enabled** -- no config flag required.

**What it does:**
1. Loads raw MEG data from BIDS
2. Runs GESD-based statistical tests on channel properties
3. Marks detected channels as bad in the BIDS sidecar files

## Manual channel inspection (`manual_channel`)

Opens an interactive GUI for visually inspecting and marking bad channels.

**Config flag:** `_manual_channels = True`

**What it does:**
1. Loads raw data with previously detected bad channels highlighted
2. Opens the MNE Qt browser for interactive inspection
3. Saves updated bad channel list back to BIDS

:::{note}
Requires a display. Set `_manual_channels = False` for headless environments.
:::

## Homogeneous field correction (`apply_hfc`)

Applies HFC projections to suppress environmental interference. HFC exploits
the fact that uniform (homogeneous) magnetic fields are not produced by brain
sources and can be projected out.

**Config flag:** `_do_HFC = True`

**What it does:**
1. Computes HFC projection vectors using `mne.preprocessing.compute_proj_hfc`
2. Adds projections to the raw data
3. Saves updated data to BIDS derivatives

## Adaptive multipole models (`apply_amm`)

Models the field the array measures with prolate spheroidal harmonics fitted
to the array (Tierney et al., 2024, https://doi.org/10.1002/hbm.26596):
internal harmonics for the brain and external harmonics for interference.
The data are projected onto the internal space after removing the external
space, and, with `_amm_corr_lim < 1`, interference that neither model
captures is removed by a temporal extension, as in tSSS.  Uses MNE's
`mne.preprocessing.apply_amm` / `compute_proj_amm`, an independent
implementation of SPM's `spm_opm_amm` checked against SPM itself
(`tests/test_amm.py`).

**Config flag:** `_do_AMM = True` (set by `_spatial_filter = "AMM"`)

**Parameters:** `_amm_li` (9), `_amm_le` (2), `_amm_corr_lim` (1.0, off),
`_amm_window` (10 s), `_amm_coord_frame` ("head"), `_amm_center_basis`
(False; True centres the basis across channels as SPM does).

**What it does:**
1. Fits a prolate spheroid to the positions of the good channels
2. Builds the internal and external harmonic bases and the projector
3. Adds the projection as SSP projectors and applies it, so the forward model
   uses the same projection
4. Optionally runs the temporal extension on windows of `_amm_window` seconds
5. Applies the task's projection to the empty-room recording
6. Saves updated data to BIDS derivatives

## ZCA spatial filter (`zca_filter` / `apply_zca`)

Applies a Zero-phase Component Analysis filter that uses a forward model and
external SSS basis to separate signal and noise subspaces via generalized
eigendecomposition.

**Config flag:** `_do_ZCA = True`

**What it does:**
1. Computes a forward model for the signal subspace
2. Computes external SSS basis for the noise subspace
3. Performs generalized eigendecomposition to identify signal/noise components
4. Creates projection vectors from the noise subspace
5. Applies projections to the data

## Bad segment detection (`bad_segments_1`)

Detects and annotates bad data segments using osl-ephys tools.

**Always enabled** -- no config flag required.

**What it does:**
1. Loads raw data
2. Optionally runs `mne.preprocessing.annotate_amplitude` when the stage
   parameters include an `amplitude` dict. Its thresholds apply to
   sample-to-sample differences, so `peak` catches jumps such as flux jumps;
   this adds `BAD_peak` / `BAD_flat` annotations and marks channels that exceed
   `bad_percent` as bad
3. Runs segment-based artifact detection using the configured stage-1 window
4. Annotates bad segments in the raw data
5. Saves back to BIDS

```python
_bad_segments_params = {
    "1": dict(
        channel_threshold=0.20,
        noise_channel_threshold=0.50,
        segment_len_sec=1.0,
        amplitude=dict(peak=dict(mag=1e-10), bad_percent=100),
    ),
}
```

## MNE preprocessing

After the custom steps, the pipeline hands off to mne-bids-pipeline for
standard preprocessing (filtering, resampling, etc.):

```bash
mne_bids_pipeline --steps=preprocessing --config=$CONFIG_PATH
```

## ICA convergence check (`ica_convergence`)

Runs whenever `spatial_filter = "ica"`. Reads the ICA solution fitted by
mne-bids-pipeline, logs whether it converged and how many of the allowed
iterations (`ica_max_iterations`) it used, and saves
`ICA/<basename>_convergence.png` next to the other ICA figures. A fit that did
not converge raises a warning, or an error with
`_ica_fail_on_nonconvergence = True`.

## Automatic ICA (`bad_ICs`)

Combines configured ICA scores with PCA, then runs GESD on the retained PCs.

**Config flag:** `_bad_ICs = True` (also requires `spatial_filter = "ica"`)

`_bad_ICs_include_preexcluded = True` keeps components already in `ica.exclude`
in the PCA fit and GESD test. Set it to `False` to fit and test only the
remaining components. The components TSV retains each tested metric's raw
score for all components; omitted candidates have no new PC score.

**What it does:**
1. Loads the ICA solution computed by mne-bids-pipeline
2. Computes the configured temporal, spatial, and correlation scores
3. Flags outlying components by PCA→GESD
4. Saves updated exclusion list to BIDS

## Manual ICA review (`manual_ica`)

Opens an interactive GUI for reviewing and selecting ICA components to exclude.

**Config flag:** `_manual_ica = True` (also requires `spatial_filter = "ica"`)

**What it does:**
1. Loads ICA solution with auto-labeled components highlighted
2. Opens interactive plots of component topographies and time courses
3. Allows manual inclusion/exclusion of components
4. Saves the final exclusion list to BIDS

## Bad epoch detection (`bad_epochs`)

Detects and drops bad epochs using GESD-based statistical tests.

**Always enabled** -- no config flag required.

**What it does:**
1. Loads epoched data from derivatives
2. Runs GESD test on epoch-level summary statistics
3. Marks bad epochs
4. Saves cleaned epochs

## Running individual steps

You can run any step independently:

```bash
# Just bad channel detection
python src/custom/custom_preproc.py --analysis=bad_channels --config=config.py

# Just HFC
python src/custom/custom_preproc.py --analysis=apply_hfc --config=config.py

# Just auto ICA
python src/custom/custom_preproc.py --analysis=bad_ICs --config=config.py
```

This is useful for re-running a single step after adjusting config settings
without repeating the entire pipeline.
