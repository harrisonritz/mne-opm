# mne-opm

[![Documentation Status](https://readthedocs.org/projects/mne-opm/badge/?version=latest)](https://mne-opm.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-67%25-yellowgreen.svg)](https://github.com/harrisonritz/mne-opm)

OPM-MEG preprocessing pipeline based on MNE-Python, mne-bids-pipeline, and OSL-ephys. This repository contains
utility scripts, custom pre-processing glue code, and run scripts to convert data to BIDS, run preprocessing,
inspect and curate bad channels/epochs, run coregistration and FreeSurfer, and prepare sensor/source outputs.

**[Full documentation on ReadTheDocs](https://mne-opm.readthedocs.io)**

This README documents how to set up the environment, prepare your data and configuration, and run the pipeline
using the provided run scripts.

[![DOI](https://zenodo.org/badge/962744712.svg)](https://doi.org/10.5281/zenodo.17328630)


## Contents
- `src/custom/` - custom preprocessing helpers and OSL wrappers
- `src/run/` - shell wrappers that run high-level pipeline stages (`run_all.sh`, `run_preproc.sh`, etc.)
- `synthetic/` - a fully synthetic BIDS subject plus ready-to-run configs, so the pipeline can be exercised end to end without real data (see [`synthetic/README.md`](synthetic/README.md))
- `local-mne-opm.sh` - convenience script to set defaults and invoke specific pipeline stages
- `mne-opm.sh` - main CLI wrapper to run a single pipeline stage with explicit arguments
- `install.sh` - environment creation and installation helper
- `pyproject.toml` - Python project metadata


## Requirements

- macOS / Linux (development tested on macOS)
- conda or micromamba for environment management
- Python 3.10+ (check `pyproject.toml` for exact requirements)
- MNE-Python, mne-bids-pipeline, osl-ephys and other scientific dependencies (installed by `install.sh`)


## Installation (quick)

use uv to create the invronment and install Python dependencies.
See installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/). 

```zsh
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The pipeline should install the necessary packages the first time it is run, and then will use the local environment from then on.


## Data layout expectations

Raw data and outputs must conform to a few naming/layout conventions used by the run scripts:

- Raw experiment folder: `${DATA_BASE}/${EXPERIMENT}/raw/` contains per-subject folders.
- For BIDS conversion, raw data folders should include `_task` or `_noise` in their names (e.g., `20250321_140828_noise/*_meg.fif`).
- DICOM-to-NIfTI conversion (`run_nifti.sh`) expects DICOMs in `${RAW_DIR}/<subject>/dicom` and writes NIfTI files to `${RAW_DIR}/<subject>/anat`.
- After conversion, files with `T1w` or `T2w` in their names are auto-renamed to include `_t1w` or `_t2w`.
- BIDS output root is `${DATA_BASE}/${EXPERIMENT}/bids` and derivatives live under `bids/derivatives`.
- FreeSurfer SUBJECTS_DIR is `${DATA_BASE}/${EXPERIMENT}/bids/derivatives/freesurfer/subjects`.

If you provide trial/behavioral metadata, place it under `raw/<subject>/metadata` as CSV files; configs can point to these.


## Example directory tree and formatting requirements

Example layout for a single experiment (TSXpilot) and subject 007. Adjust names and paths to your project.

```
/data/TSXpilot/
	raw/
		exp_007/                       # subject folder must end with _007 (3-digit, zero-padded)
			dicom/                       # DICOMs (input to run_nifti.sh)
				...
			anat/                        # NIfTI outputs created by run_nifti.sh
				<files>_t1w.nii.gz         # must include suffix _t1w
				<files>_t2w.nii.gz         # must include suffix _t2w (optional)
			session1_task/               # task run folder; must end with _task
				20250321_123456_meg.fif    # raw MEG FIF; name can vary but must match *_meg.fif
			session2_task/
				20250321_125500_meg.fif
			20250321_140828_noise/       # empty-room; folder must end with _noise
				20250321_140828_meg.fif
			metadata/                    # optional per-subject metadata
				sub-007_behavior.csv       # metadata files should start with sub-<NNN>_
				sub-007_events.csv
			eyetracking/
				recording.asc              # Eyelink ASCII file (*.asc) detected anywhere under subject

	bids/
		sub-007/
			ses-01/
				meg/
					sub-007_ses-01_task-<task>_run-01_meg.fif
					sub-007_ses-01_task-noise_run-01_meg.fif     # if empty-room exists
				anat/
					sub-007_ses-01_T1w.nii.gz
					sub-007_ses-01_T2w.nii.gz
		derivatives/
			freesurfer/
				subjects/
					sub-007_ses-01/...
			<ANALYSIS>/                  # e.g., CSI; pipeline derivatives live here

	configs/
		TSXpilot/
			config-CSI.py
			bids/
				sub-007_config-bids.py     # used by run_bids.sh
```

Formatting requirements enforced/assumed by the scripts:

- Subject folder naming (raw): must end with `_<NNN>` where `NNN` is a 3-digit, zero-padded subject id (e.g., `exp_007`).
- Task runs (raw): run folders must end with `_task` and contain files matching `*_meg.fif`.
- Empty-room (raw): folder must end with `_noise` and contain `*_meg.fif`.
- DICOMs (raw): stored under `dicom/`; `run_nifti.sh` writes NIfTI to `anat/` and renames files to include `_t1w` / `_t2w`.
- Anatomical NIfTI files (raw): must include suffix `_t1w.nii*` and optionally `_t2w.nii*` (detected by glob).
- Eye-tracking: Eyelink ASCII `.asc` file located anywhere inside the subject directory is detected and aligned.
- Metadata: per-subject files should be prefixed `sub-<NNN>_` (e.g., `sub-007_*`) for consistent discovery.
- BIDS: per-subject BIDS config at `CONFIG_DIR/bids/sub-<SUBJECT>_config-bids.py`.
- Sessions: default is `ses-01`; FreeSurfer/coreg scripts temporarily format subject as `sub-<NNN>_ses-<SESSION>`.


## Environment variables and configuration

This pipeline uses environment variables to locate data and configuration. The minimal variables you should set
before running are:

- `ROOT_DIR` - root of this repository (typically the project folder)
- `EXPERIMENT` - short experiment name (used to find config directory)
- `ANALYSIS` - analysis name, appended to derivative folders (e.g., CSI)
- `SUBJECT` - subject id (e.g., 007)
- `CONFIG_DIR` - path to config folder (usually `<analysis repo>/analysis/config`)
- `RAW_DIR`, `BIDS_DIR`, `SUBJECTS_DIR` - data paths used by the pipeline

You can set these directly in your shell or use `local-mne-opm.sh` to generate and export them for you. Example usage of
the helper script is shown below.


## CLI wrapper: mne-opm.sh (recommended)

Run a single pipeline stage with explicit arguments (no pre-set defaults). Valid `<pipeline>` options:

`nifti | bids | freesurfer | coreg | preproc | sensor | source | all | func | anat`

Usage (from repo root):

```zsh
mne-opm.sh <pipeline> \
	--exp TSXpilot \
	--sub 007 \
	--data /path/to/data/TSXpilot \
	--config /path/to/config/TSXpilot \
	--analysis CSI \
	--fs /Applications/freesurfer/8.0.0 \
	--t1w /path/to/T1w.nii.gz   # optional; auto-detected if under BIDS
```

Notes:
- The script exports environment variables used by the run scripts (ROOT_DIR, CONFIG_DIR, RAW/BIDS paths, etc.).
- `--analysis` selects `config-<ANALYSIS>.py` for most stages; BIDS uses a per-subject config at `CONFIG_DIR/bids/sub-<SUBJECT>_config-bids.py`.
- Set `--workers` to control FreeSurfer parallelism (`MAX_WORKERS`).
- Ensure FreeSurfer is installed and `--fs` points to your install; the scripts source `SetUpFreeSurfer.sh` when needed.


## Pipeline stages (what each script does)

- `run_bids.sh`  : Convert raw files into BIDS structure (expects raw file naming conventions)
- `run_freesurfer.sh` : Run FreeSurfer recon-all for anatomical processing
- `run_coreg.sh`: Coregister sensor and anatomical spaces
- `run_preproc.sh`: Runs the preprocessing flow including OSL wrappers and manual inspection steps
- `run_sensor.sh` / `run_source.sh`: Sensor- and source-level processing steps used for analysis

Inside `run_preproc.sh` you will see a combination of calls to:
- `python src/custom/custom_preproc.py --analysis=...` for custom OSL / helper scripts
- `mne_bids_pipeline --steps=... --config=...` for standardized MNE-BIDS pipeline steps

Details by stage:

- `run_nifti.sh`
	- Uses `dcm2niix` to convert DICOMs from `${RAW_DIR}/<subject>/dicom` to NIfTI in `${RAW_DIR}/<subject>/anat`.
	- Skips if the NIfTI directory already exists.
	- Auto-renames files containing `T1w`/`T2w` to include `_t1w`/`_t2w` suffixes.

- `run_bids.sh`
	- Loads a per-subject BIDS config: `${CONFIG_DIR}/bids/sub-${SUBJECT}_config-bids.py`.
	- Calls `src/custom/format_bids.py` to create/validate the BIDS structure.

- `run_freesurfer.sh`
	- Requires T1w (and optionally T2w) images. Paths can be auto-detected from BIDS or provided via `--t1w`, `--t2w`.
	- Runs `recon-all` with `-parallel -openmp ${MAX_WORKERS}`.
	- Builds watershed BEM surfaces afterward.

- `run_coreg.sh`
	- Temporarily sets `SUBJECT=sub-<id>_ses-<SESSION>` for MNE/FreeSurfer conventions.
	- Calls `src/custom/auto_coreg.py` for automated head<->MRI coregistration.

- `run_preproc.sh`
	- Sets `CONFIG_PATH=${CONFIG_DIR}/config-${ANALYSIS}.py`.
	- Calls custom steps: `bad_segments`, `bad_channels`, `manual_channel` via `custom_preproc.py`.
	- Runs `mne_bids_pipeline --steps=preprocessing` followed by manual ICA selection and application (ICA/SSP/PTP reject).
	- Detects bad epochs and prepares source space prerequisites (evoked, cov, BEM solution, source space, forward).

- `run_sensor.sh`
	- Runs the sensor-level steps via `mne_bids_pipeline --steps=sensor` using your config.

- `run_source.sh`
	- Sources FreeSurfer env then runs `mne_bids_pipeline --steps=source`.


## Config files

Project-specific configuration files are stored in your `CONFIG_DIR` (e.g., `analysis/config/<EXPERIMENT>`).
Each config is a plain Python file that sets variables consumed by the pipeline, e.g.:

- `bids_root`, `deriv_root`, `subjects`, `sessions`, `task`, `process_empty_room`
- preprocessing options: `l_freq`, `h_freq`, Maxwell filter settings, bad-channel/epoch options
- custom flags: `_skip_on_deriv`, `_manual_bads`, `_manual_ica`

Example config snippet (from this repo's configs):

```python
_skip_on_deriv = True  # if True, skip steps when derivatives exist
process_empty_room = True
subjects = ['007']
task = 'TSX'  # experiment task name
```

Notes:
- Config files are plain Python. The pipeline imports them as modules to populate a `SimpleNamespace`.
- If you need dict-like behavior (for example calling `.pop()`), convert the namespace with `vars(cfg)`.

Typical config file layout under your config directory:

```
CONFIG_DIR/
	<EXPERIMENT>/
		config-<ANALYSIS>.py        # used by most stages
		bids/
			sub-<SUBJECT>_config-bids.py  # used by run_bids.sh
```


## Custom scripts and manual steps

The repository includes custom helpers in `src/custom/` to integrate OSL/Ephephys steps (bad segment detection, bad
channel detection, manual ICA selection, etc.). These are invoked by `custom_preproc.py`. When running manual
inspection steps, the scripts will typically open interactive plots - run from an X-enabled environment or save
outputs to disk when running headless.


## Common workflows

- Run full automated preprocessing for one subject:

```bash
mne-opm.sh preproc --exp TSXpilot --sub 007 --config /path/to/config
```

- Run the whole pipeline against the bundled synthetic subject (no real data needed):

```bash
bash mne-opm.sh preproc --exp synth --sub 001 --session 01 --analysis trial \
    --data synthetic/datasets --config synthetic/config
```


## Tips and gotchas

- FreeSurfer: set `--fs` to your install root or export `FREESURFER_HOME` beforehand.
  - Make sure that none of the paths used by freesurfer have spaces in them
- Some stages rely on `SESSION` (default `01`) for subject formatting in coreg; ensure it’s set in your environment if needed.
- Interactive steps require a display; set `MPLBACKEND=agg` to render to files when running headless.
- To re-run a stage that checks for existing derivatives, delete the target derivative folder or set `_skip_on_deriv = False` in your config.


## Troubleshooting

- If the pipeline says derivatives exist and you want to re-run, either remove the derivatives folder or set
	`_skip_on_deriv = False` in your config and re-run.
- For plotting/interactive steps, ensure your environment supports GUI output or configure plots to save to disk.


## Contributing

Contributions are welcome. Please follow the project's style, run tests if present, and create a PR against the
`main` branch. For large changes, open an issue first describing the proposal.


## License and citation

Check the repository root for a LICENSE file. When using this pipeline in publications, please cite MNE-Python,
mne-bids-pipeline, and any other libraries used.


## Contact / support

Open an issue in this repo with reproducible steps to replicate problems and include the full stack trace and the
config file you used.


