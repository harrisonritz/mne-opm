# Preprocessing walkthrough

This tutorial explains each preprocessing step in the pipeline, what it does,
and how to configure it.

## Pipeline order

The `run_preproc.sh` script runs these steps in sequence:

```text
bad_channels -> manual_channel -> apply_hfc -> apply_zca
    -> bad_segments -> MNE preprocessing -> auto_ica
    -> manual_ica -> MNE ICA apply -> bad_epochs
```

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

## Bad segment detection (`bad_segments`)

Detects and annotates bad data segments using osl-ephys tools.

**Always enabled** -- no config flag required.

**What it does:**
1. Loads raw data
2. Runs segment-based artifact detection (default 1-second segments)
3. Annotates bad segments in the raw data
4. Saves back to BIDS

## Reference channel regression (`regress_ref`)

Regresses out signals captured by reference MEG sensors from the primary
sensors. Reference sensors are far from the scalp and primarily measure
environmental noise.

**Config flag:** `_regress_ref = True`

**What it does:**
1. Loads raw data with both MEG and reference channels
2. Fits a regression model from reference to MEG channels
3. Subtracts predicted noise from MEG channels
4. Saves cleaned data to BIDS

## MNE preprocessing

After the custom steps, the pipeline hands off to mne-bids-pipeline for
standard preprocessing (filtering, resampling, etc.):

```bash
mne_bids_pipeline --steps=preprocessing --config=$CONFIG_PATH
```

## Automatic ICA (`auto_ica`)

Automatically labels ICA components by correlating them with reference sensors.

**Config flag:** `_auto_ica = True` (also requires `spatial_filter = "ica"`)

**What it does:**
1. Loads the ICA solution computed by mne-bids-pipeline
2. Correlates ICA components with reference channel signals
3. Labels highly correlated components as artifacts
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
python src/custom/custom_preproc.py --analysis=auto_ica --config=config.py
```

This is useful for re-running a single step after adjusting config settings
without repeating the entire pipeline.
