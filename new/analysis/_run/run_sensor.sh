## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg

subject=003
config_path="../config/TSX-pilot/sub-${subject}/config-sensor_sub-${subject}.py"


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings in `config`
echo "Running MNE BIDS pipeline..."
mne_bids_pipeline --steps=sensor --config=$config_path
