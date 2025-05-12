# run BIDS configuration
# Harrison Ritz 2025

## activate environment ----------------------------------------
conda init
conda activate mne-opm

export SUBJECT=$1


## format data for bids ----------------------------------------
python ../scripts/format_bids.py --config="../config/example/sub-${SUBJECT}/config-bids_sub-${SUBJECT}.py"