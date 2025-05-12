# mne-opm

Development branch for mne-opm, with bleeding-edge version of opm-compatible pipeline
_Things are still being updated_


## pipeline setup
1. set-up your environment variables for your local pathing (using the `env-vars.env.example` template to generate `env-vars.env`)
2. create the conda environment by following the steps in `install.sh`


## data setup
- for bids conversion, raw data must have `_task` or `_noise` appended to the data folders (e.g., `20250321_140828_noise/*_meg.fif`)
- anatomical files must have `_t1w` appended
- any metadata should be in `raw/*/metadata`


## preprocessing steps (`src/_run`)
0. run_bids
1. run_freesurfer
2. run_coreg
3. run_preproc (parallel to freesurfer/coreg)
4. run_sensor (parallel to freesurfer/coreg)
5. run_source


