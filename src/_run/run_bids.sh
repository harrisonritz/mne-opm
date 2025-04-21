

## activate environment ----------------------------------------
# source ../../env-vars.env might not need
conda init
conda activate mne-opm


## format data for bids ----------------------------------------
# edit the configuration settings in `config`

# python ${TSX_ROOT}/analysis/scripts/opm_format_bids.py "/Users/hr0283/Projects/mne-opm/config/sub-004/config-bids_sub-004.yml"
python ../scripts/preproc/opm_format_bids.py "../config/AV-pilot/sub-004/config-bids_sub-004.yml"
