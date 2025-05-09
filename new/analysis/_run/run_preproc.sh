## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg

export SUBJECT=004
config_path="../config/TSX-pilot/sub-${SUBJECT}/config-preproc_sub-${SUBJECT}.py"


## TODO: run osl_ephys preprocessing steps before mne_bids_pipeline ----------------------------------------
# echo "Running OSL preprocessing for bad segments detection..."
python ../scripts/aux_preproc.py --analysis=badsegment --config=$config_path

# echo "Running OSL preprocessing for bad channels detection..."
# python ../scripts/preproc/osl_preproc.py badchannels --config=$config_path


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings in `config`
# echo "Running MNE BIDS pipeline..."
# mne_bids_pipeline --steps=preprocessing --config=$config_path


## TODO:  run osl_ephys post-processing after mne_bids_pipeline ----------------------------------------
# echo "Running OSL post-processing for bad epochs detection..."
# python ../scripts/preproc/osl_preproc.py badepochs --config=$config_path
