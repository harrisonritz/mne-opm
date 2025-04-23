# mne-opm

Development branch for mne-opm, with bleeding-edge version of opm-compatible pipeline

## TODO
- extend beyond preprocessing (freesurfer, recon, sensor, source)
- incorporate osl preprocessing (segment and channel detection before preproc?)?
- add `plot_sources` to ica outputs
- use `.py` config files for bids



## pipeline setup
1. set-up your environment variables for your local pathing (used `env-vars.env.example` template to generate `env-vars.env`)

2. create the conda environment
	- **Reccomended**: install from included environment file (`conda env create -f environment.yml`)
	- follow the steps in `old_install.sh`


## preprocessing setup
1. set-up your subject-specific bids and preproc config files  (e.g., `src/config/AV-pilot/sub-xxx`)
	- these may be set-up to  

2. convert your OPM data to BIDS (example in `src/_run/run_pipeline.sh`, but update the paths)
	- BIDS conversion based on PNI's OPM data format
	
3. run preprocessing steps (example in `src/_run/run_pipeline.sh`, but update the paths)


## data setup
- for bids conversion, raw data must have `_task` or `_noise` appended to the data folders (e.g., `20250321_140828_noise/*_meg.fif`)
- anatomical files must have `_t1w` appended



## example
- download the example dataset and add to the root folder: https://www.dropbox.com/scl/fo/mkq0xbvxl8lbf1z2fx9eo/AHgnTCyyTBqNeqQUWGSlMjU?rlkey=qorma6meq0ki2u7w3wpujkyhy&dl=0 
- you may have to reformat the data using the `data setup` above