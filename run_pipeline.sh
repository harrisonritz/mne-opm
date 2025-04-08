
## activate environment ----------------------------------------
conda activate mne-opm


## format data for bids ----------------------------------------
# edit the configuration settings in `src/opm_format_bids.py`
# call the right location for opm_format_bids

python /Users/hr0283/Projects/mne-opm/src/opm_format_bids.py "/Users/hr0283/Projects/mne-opm/config/sub-004/config-bids_sub-004.yml"


## run preproc pipeline ----------------------------------------
# edit the configuration settings in `config`
# call the right location for `config_preproc`

mne_bids_pipeline --steps=preprocessing --config="/Users/hr0283/Projects/mne-opm/config/sub-004/config-preproc_sub-004.py"


## run freesurfer pipeline ----------------------------------------
# edit the configuration settings in `config/config_preproc_sub-004.py`
# call the right location for `config_preproc`

# does not not work yet
# mne_bids_pipeline --steps=freesurfer --config="/Users/hr0283/Projects/mne-opm/config/sub-004/config-freesurfer_sub-004.py"



## run source recon pipeline ----------------------------------------
# edit the configuration settings in `config/config_preproc_sub-004.py`
# call the right location for `config_preproc`


# does not not work yet
# mne_bids_pipeline --steps=source --config="/Users/hr0283/Projects/mne-opm/config/sub-004/config-source_sub-004.py"

