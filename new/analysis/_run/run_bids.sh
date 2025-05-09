# run BIDS configuration
# Harrison Ritz 2025

## activate environment ----------------------------------------
conda init
conda activate mne-opm

subject=004


## format data for bids ----------------------------------------
python ../scripts/format_bids.py --config="../config/TSX-pilot/sub-${subject}/config-bids_sub-${subject}.py"