# API reference

The Python API is organized under the `custom` package:

- `custom.preprocessing` -- preprocessing analysis modules and shared utilities
- `custom.custom_preproc` -- CLI dispatcher and analysis registry
- `custom.osl` -- osl-ephys pipeline, from preprocessing to beamforming
- `custom.run_osl` -- osl-ephys pipeline CLI

Most users interact with mne-opm through the CLI (`mne-opm.sh`,
`custom_preproc.py`, or `run_osl.py`). The Python API is useful for scripting,
extending the pipeline, or integrating mne-opm into larger workflows.

```{toctree}
:maxdepth: 2

preprocessing
osl
utilities
```
