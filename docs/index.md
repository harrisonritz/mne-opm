# mne-opm

**OPM-MEG preprocessing pipeline built on MNE-Python, mne-bids-pipeline, and OSL-ephys.**

mne-opm provides a modular, BIDS-native pipeline for preprocessing optically pumped
magnetometer (OPM) MEG data. It wraps battle-tested tools from the MNE ecosystem with
OPM-specific steps such as homogeneous field correction (HFC), ZCA spatial filtering,
reference-channel regression, and automated/manual artifact rejection.

## Key features

- **Modular analyses** -- each preprocessing step (bad channels, HFC, ICA, etc.) is an
  independent module with a consistent `BaseAnalysis` interface.
- **CLI-driven** -- run any stage from the command line via `mne-opm.sh` or
  `custom_preproc.py`.
- **BIDS-native** -- reads and writes data in Brain Imaging Data Structure format using
  mne-bids.
- **Configuration files** -- all settings live in plain Python config files, inspired by
  mne-bids-pipeline.
- **Interactive and automated** -- supports both hands-free batch processing and
  interactive GUI steps for manual inspection.

## Quick links

- {doc}`installation` -- get up and running
- {doc}`usage` -- pipeline overview and CLI reference
- {doc}`tutorials/index` -- step-by-step walkthroughs
- {doc}`api/index` -- Python API reference

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
tutorials/index
api/index
```
