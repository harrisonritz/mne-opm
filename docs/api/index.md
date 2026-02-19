# API reference

The Python API is organized under the `custom` package:

- `custom.preprocessing` -- preprocessing analysis modules and shared utilities
- `custom.custom_preproc` -- CLI dispatcher and analysis registry

Most users interact with mne-opm through the CLI (`mne-opm.sh` or
`custom_preproc.py`). The Python API is useful for scripting, extending the
pipeline, or integrating mne-opm into larger workflows.

```{toctree}
:maxdepth: 2

preprocessing
utilities
```
