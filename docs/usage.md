# Usage

## Pipeline overview

mne-opm is organized as a sequence of pipeline stages, each handled by a shell
script in `src/run/`. The main entry point is `mne-opm.sh`, which sets up the
environment and dispatches to the appropriate stage.

```text
nifti -> bids -> freesurfer -> coreg -> preproc -> sensor -> source
```

You can run stages individually or chain them with `all` / `func` / `anat`.

## CLI reference: mne-opm.sh

```bash
sh mne-opm.sh <pipeline> \
    --exp <experiment> \
    --sub <subject> \
    --data <data_directory> \
    --config <config_directory> \
    [--analysis <analysis_name>] \
    [--session <session>] \
    [--fs <freesurfer_home>] \
    [--workers <n>] \
    [--fail-on-first-crash]
```

### Required arguments

`pipeline`
: Stage to run. One of: `nifti`, `bids`, `freesurfer`, `coreg`, `preproc`,
  `sensor`, `source`, `beamformer`, `all`, `func`, `anat`.

`--exp`
: Experiment name (e.g., `TSXpilot`). Used to locate config files and data.

`--sub`
: Subject ID as a 3-digit number (e.g., `007`). Zero-padded.

`--data`
: Root of your data directory. The pipeline expects
  `<data>/<experiment>/raw/` and `<data>/<experiment>/bids/`.

`--config`
: Root of your config directory. The pipeline looks for
  `<config>/<experiment>/config-<analysis>.py`.

### Optional arguments

`--analysis`
: Analysis name (e.g., `CSI`). Selects `config-<analysis>.py` and sets the
  derivative output folder.

`--session`
: Session label (default: `01`).

`--fs`
: Path to FreeSurfer installation root.

`--workers`
: Number of parallel workers for FreeSurfer.

`--fail-on-first-crash`
: Stop the pipeline immediately on the first error.

## Pipeline stages

### nifti

Converts DICOMs to NIfTI using `dcm2niix`. Reads from
`<raw>/<subject>/dicom/` and writes to `<raw>/<subject>/anat/`.

### bids

Converts raw MEG data into BIDS format. Uses a per-subject config file at
`<config>/bids/sub-<subject>_config-bids.py`.

### freesurfer

Runs FreeSurfer `recon-all` for cortical reconstruction and BEM watershed.
Requires T1w (and optionally T2w) anatomical images.

### coreg

Coregisters sensor space to MRI space. Uses automatic fitting with manual
verification.

### preproc

The main preprocessing stage. Runs the following steps in order:

1. **bad_channels** -- statistical detection of bad channels (GESD)
2. **manual_channel** -- interactive GUI for manual bad channel marking
3. **apply_hfc** -- homogeneous field correction projections
4. **apply_zca** -- zero-phase component analysis spatial filter
5. **bad_segments** -- detect and annotate bad data segments
6. **MNE preprocessing** -- filtering, resampling via mne-bids-pipeline
7. **auto_ica** -- automatic ICA component labeling
8. **manual_ica** -- interactive ICA component review
9. **MNE ICA application** -- apply ICA, SSP, peak-to-peak rejection
10. **bad_epochs** -- GESD-based bad epoch detection

### sensor / source

Sensor-level and source-level analyses via mne-bids-pipeline.

### beamformer

LCMV beamformer source reconstruction.

Reconstructs a volume source space, a cortical-surface source space, or both in
one pass (`_beamformer_source_space`); the two share the data and noise
covariance, so running both costs one extra forward and one extra filter set.

Two settings matter more than the rest for group-level quality:

- **`noise_cov`** should be `"emptyroom"`, not `"ad-hoc"`. An ad-hoc covariance
  is isotropic, so its eigenvectors are the canonical channel basis — when MNE
  truncates it to the data rank it discards *whole channels* rather than a
  meaningful subspace. It also reduces `weight_norm="nai"` to a constant.
  Requires `process_empty_room = True` and a prior
  `mne_bids_pipeline --steps=sensor/make_cov` run, whose covariance and rank
  sidecar the beamformer then reuses.
- **Orientation** differs by source space, so `_beamformer_pick_ori` and
  `_beamformer_weight_norm` accept a dict keyed by source space. A surface space
  has a cortical normal to anchor the max-power sign to (enabled by
  `_beamformer_surf_ori`); a volume grid does not — MNE falls back to the
  +Z / superior direction, which is noise-driven for tangentially oriented
  sources. Volume spaces are therefore fit with `pick_ori="vector"`, and
  `TSX_OPM/analysis/source/beamformer_volume.py` reads the components out along
  the fsaverage cortical normals when projecting to the surface.

`_beamformer_rank = "data"` takes the element-wise minimum of the info rank, a
data-driven SVD rank, and the noise covariance's stored rank. Neither of the first
two works alone: `"info"` only sees the SSS bookkeeping in `info['proc_history']`
and so misses what ICA removed, while a data-driven SVD at MNE's default
`tol="auto"` (`n_dim * max_s * eps_float64`, ~1e-13 relative) counts the SSS- and
ICA-nulled directions that return from a float32 FIF at ~1e-7 relative — which is
why `tol="auto"` reports a rank *above* the info rank on Maxwell-filtered data.
The tolerance for the data-driven part is `_beamformer_rank_tol` (default `1e-6`,
relative). Cross-check a subject with:

```bash
python src/custom/rank_check.py --config=/path/to/config.py clean
```

## Custom preprocessing CLI

Individual preprocessing steps can also be run directly:

```bash
python src/custom/custom_preproc.py \
    --analysis=<step> \
    --config=/path/to/config.py
```

Available steps: `regress`, `bad_segments`, `bad_channels`,
`manual_channel`, `apply_hfc`, `zca_filter`, `apply_zca`, `bad_epochs`,
`auto_ica`, `manual_ica`, `coreg`.

## Data layout

Raw data and outputs must follow these conventions:

```text
<data_root>/<experiment>/
    raw/
        <subject_folder>/        # must end with _<NNN> (e.g., exp_007)
            dicom/               # DICOMs for NIfTI conversion
            anat/                # NIfTI anatomical images
            <run>_task/          # task runs (folder ends with _task)
            <run>_noise/         # empty-room (folder ends with _noise)
            metadata/            # optional behavioral CSV files
    bids/
        sub-<NNN>/
            ses-01/
                meg/
                anat/
        derivatives/
            freesurfer/subjects/
            <analysis>/          # pipeline output
```

See the [README](https://github.com/harrisonritz/mne-opm#data-layout-expectations)
for full naming requirements.

## Environment variables

The pipeline uses these environment variables (set automatically by
`mne-opm.sh`):

| Variable        | Description                                |
|-----------------|--------------------------------------------|
| `ROOT_DIR`      | Repository root directory                  |
| `EXPERIMENT`    | Experiment name                            |
| `SUBJECT`       | Subject ID (e.g., `007`)                   |
| `ANALYSIS`      | Analysis name                              |
| `SESSION`       | Session label (default `01`)               |
| `CONFIG_DIR`    | Path to config directory                   |
| `RAW_DIR`       | Raw data directory                         |
| `BIDS_DIR`      | BIDS output directory                      |
| `SUBJECTS_DIR`  | FreeSurfer subjects directory              |
| `MAX_WORKERS`   | Parallel workers for FreeSurfer            |
