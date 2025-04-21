
## activate environment ----------------------------------------
conda init
conda activate mne-opm


## run preproc pipeline ----------------------------------------
# edit the configuration settings in `config`

mne_bids_pipeline --steps=preprocessing --config="../config/TSX-pilot/sub-004/config-preproc_sub-004.py"


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

