# mne-opm

Download the example dataset and add to the root folder: https://www.dropbox.com/scl/fo/mkq0xbvxl8lbf1z2fx9eo/AHgnTCyyTBqNeqQUWGSlMjU?rlkey=qorma6meq0ki2u7w3wpujkyhy&dl=0 **NOTE: may not be formatting properly**


# data formatting
- for bids conversion, raw data must have `_task` or `_noise` appended to the data folders (e.g., `20250321_140828_noise/20250321_140828_*.fif`)
- anatomical files must have `_t1w` appended
- format `env-vars.env.example` with your local paths, and save as `env-vars.env`


## Preprocessing steps

1. set-up your subject-specific config files  (e.g., `src/config/AV-pilot/sub-xxx`)

2. convert your OPM data to BIDS (example in `src/_run/run_pipeline.sh`, but update the paths)
	- BIDS conversion based on PNI's OPM data format
	
3. run preprocessing steps (example in `src/_run/run_pipeline.sh`, but update the paths)
