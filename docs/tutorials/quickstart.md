# Quick start

This tutorial walks through running the full pipeline on a single subject,
from raw data to preprocessed output.

## 1. Organize your raw data

Place your raw data under a directory following the expected layout:

```text
/data/MyExperiment/
    raw/
        exp_001/
            session1_task/
                20250101_120000_meg.fif
            20250101_130000_noise/
                20250101_130000_meg.fif
            dicom/
                ...
```

Key rules:
- Subject folders must end with `_<NNN>` (3-digit, zero-padded)
- Task run folders must end with `_task`
- Empty-room folders must end with `_noise`
- MEG files must match `*_meg.fif`

## 2. Create a configuration file

Create a config directory with a Python config file:

```text
/configs/MyExperiment/
    config-MyAnalysis.py
    bids/
        sub-001_config-bids.py
```

The analysis config (`config-MyAnalysis.py`) is a plain Python file that sets
variables consumed by the pipeline:

```python
# config-MyAnalysis.py

# BIDS settings
bids_root = "/data/MyExperiment/bids"
deriv_root = "/data/MyExperiment/bids/derivatives/MyAnalysis"
subjects = ["001"]
sessions = ["01"]
task = "mytask"

# Preprocessing
process_empty_room = True
l_freq = 0.1
h_freq = 100.0

# Custom steps
_do_HFC = True
_do_ZCA = False
_regress = False
_regress_preds = ["ref_meg"]
_manual_channels = True
_auto_ica = True
_manual_ica = True
spatial_filter = "ica"
```

The BIDS config (`sub-001_config-bids.py`) defines per-subject conversion
settings. See {doc}`configuration` for details.

## 3. Convert to BIDS

```bash
sh mne-opm.sh bids \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs
```

This creates the BIDS directory structure under `/data/MyExperiment/bids/`.

## 4. Run anatomical processing

If you have DICOMs, convert them to NIfTI first:

```bash
sh mne-opm.sh nifti \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs
```

Then run FreeSurfer:

```bash
sh mne-opm.sh freesurfer \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs \
    --fs /Applications/freesurfer/8.0.0 \
    --workers 4
```

And coregistration:

```bash
sh mne-opm.sh coreg \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs
```

## 5. Run preprocessing

```bash
sh mne-opm.sh preproc \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs \
    --analysis MyAnalysis
```

This runs the full preprocessing pipeline:
1. Bad channel detection (automatic + manual)
2. HFC projections
3. Bad segment detection
4. MNE preprocessing (filtering, resampling)
5. ICA (automatic labeling + manual review)
6. Bad epoch detection

Interactive steps (manual channel marking, manual ICA) will open GUI windows.
Make sure you have a display available, or set `_manual_channels = False` and
`_manual_ica = False` in your config for headless operation.

## 6. Run sensor/source analysis

```bash
sh mne-opm.sh sensor \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs \
    --analysis MyAnalysis

sh mne-opm.sh source \
    --exp MyExperiment \
    --sub 001 \
    --data /data \
    --config /configs \
    --analysis MyAnalysis \
    --fs /Applications/freesurfer/8.0.0
```

## Next steps

- {doc}`configuration` -- detailed guide to writing config files
- {doc}`preprocessing` -- walkthrough of each preprocessing step
