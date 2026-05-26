# Configuration guide

mne-opm uses plain Python files for configuration, following the same pattern
as mne-bids-pipeline. This guide covers how to write and organize config files.

## Config file basics

A config file is a `.py` file that defines variables at the module level. The
pipeline imports it and exposes all variables as attributes on a
`SimpleNamespace` object.

```python
# config-CSI.py

bids_root = "/data/TSXpilot/bids"
deriv_root = "/data/TSXpilot/bids/derivatives/CSI"

subjects = ["007"]
sessions = ["01"]
task = "TSX"

process_empty_room = True
```

When loaded, these become `cfg.bids_root`, `cfg.subjects`, etc.

## Required settings

Every analysis config should define at minimum:

`bids_root`
: Path to the BIDS dataset root.

`deriv_root`
: Path to the derivatives output directory.

`subjects`
: List of subject IDs (without the `sub-` prefix).

`sessions`
: List of session IDs (without the `ses-` prefix).

`task`
: Task name used in BIDS filenames.

## MNE-BIDS-pipeline settings

Since mne-opm builds on mne-bids-pipeline, your config files can include any
setting recognized by mne-bids-pipeline. Common ones:

```python
# Filtering
l_freq = 0.1        # high-pass filter (Hz)
h_freq = 100.0      # low-pass filter (Hz)

# Epoching
epochs_tmin = -0.2
epochs_tmax = 0.5
baseline = (None, 0)

# ICA
spatial_filter = "ica"
ica_max_iterations = 500
```

See the [mne-bids-pipeline settings reference](https://mne.tools/mne-bids-pipeline/stable/settings/)
for the full list.

## Custom preprocessing flags

mne-opm adds several flags (prefixed with `_`) to control custom steps:

| Flag                | Type   | Default | Description                              |
|---------------------|--------|---------|------------------------------------------|
| `_skip_on_deriv`    | `bool` | `False` | Skip steps when derivatives already exist |
| `_do_HFC`           | `bool` | `False` | Apply homogeneous field correction        |
| `_do_ZCA`           | `bool` | `False` | Apply ZCA spatial filter                  |
| `_regress`          | `bool` | `False` | Regress out sensor signals                |
| `_regress_preds`    | `list` | `[]`    | Channel names/types to use as predictors  |
| `_manual_channels`  | `bool` | `False` | Enable interactive bad channel GUI        |
| `_auto_ica`         | `bool` | `False` | Enable automatic ICA labeling             |
| `_manual_ica`       | `bool` | `False` | Enable interactive ICA review             |

:::{note}
ICA-related flags (`_auto_ica`, `_manual_ica`) also require
`spatial_filter = "ica"` to be set.
:::

## Per-subject BIDS config

BIDS conversion uses a separate per-subject config at:

```text
<config_dir>/bids/sub-<NNN>_config-bids.py
```

This file defines how raw data maps to BIDS structure for a specific subject.

## Config file organization

Typical layout:

```text
configs/
    MyExperiment/
        config-CSI.py              # main analysis config
        config-ERP.py              # alternative analysis config
        bids/
            sub-001_config-bids.py # per-subject BIDS mapping
            sub-002_config-bids.py
```

## Loading configs programmatically

You can load config files from Python using the `load_config` utility:

```python
from custom.preprocessing import load_config

cfg = load_config("/path/to/config-CSI.py")
print(cfg.subjects)   # ['007']
print(cfg.bids_root)  # '/data/TSXpilot/bids'
```

## Tips

- Config files are plain Python, so you can use expressions, conditionals, and
  imports to build settings dynamically.
- Use `_skip_on_deriv = True` during development to avoid re-running completed
  steps.
- Set `_manual_channels = False` and `_manual_ica = False` for fully automated
  (headless) runs.
