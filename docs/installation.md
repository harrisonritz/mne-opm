# Installation

## Prerequisites

- **Python 3.13+** (the project pins `~=3.13.0` in `pyproject.toml`)
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **FreeSurfer** (required for anatomical processing and source-space analyses)

## Install uv

If you don't already have uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clone and install

```bash
git clone https://github.com/harrisonritz/mne-opm.git
cd mne-opm
uv sync
```

`uv sync` creates a virtual environment in `.venv/` and installs all dependencies
(including the custom forks of MNE-Python, mne-bids, mne-bids-pipeline, and
osl-ephys listed in `pyproject.toml`).

## Custom forks

mne-opm depends on patched versions of several upstream packages. These are
installed automatically by uv from the `[tool.uv.sources]` section in
`pyproject.toml`:

| Package            | Fork branch |
|--------------------|-------------|
| MNE-Python         | `mne-opm`   |
| mne-bids           | `mne-opm`   |
| mne-bids-pipeline  | `mne-opm`   |
| osl-ephys          | default     |

You don't need to install these manually -- `uv sync` handles everything.

## Verify the installation

Activate the virtual environment and check that the key packages are importable:

```bash
source .venv/bin/activate
python -c "import mne; import mne_bids; import custom.preprocessing; print('All imports OK')"
```

## FreeSurfer setup

Several pipeline stages (coregistration, BEM, source space) require FreeSurfer.
Install it following the [FreeSurfer wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall),
then point the pipeline to your install:

```bash
# Either export the variable...
export FREESURFER_HOME=/Applications/freesurfer/8.0.0

# ...or pass it on the CLI
sh mne-opm.sh freesurfer --fs /Applications/freesurfer/8.0.0 ...
```

## Development install

To also install testing and documentation dependencies:

```bash
uv sync --group dev --group docs
```

Run the test suite:

```bash
uv run pytest
```
