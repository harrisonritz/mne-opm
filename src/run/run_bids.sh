# run BIDS configuration
# Harrison Ritz 2025


## activate environment ----------------------------------------
# source activate base
source activate mne-opm


export CONFIG_PATH="$CONFIG_DIR/bids/sub-${SUBJECT}_config-bids.py"


## BIDS FORMATTING ----------------------------------------
echo ""
echo "RUNNING BIDS FORMATTING --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "SUBJECT: $SUBJECT"
echo "CONFIG PATH: $CONFIG_PATH"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------------------------------"
echo ""

python $ROOT_DIR/src/custom/format_bids.py --config=$CONFIG_PATH