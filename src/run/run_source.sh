# run sensor-level analysis
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


# freesurfer variables
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export FS_ALLOW_DEEP=1


## RUN PREPROCESSING ------------------
echo ""
echo "RUNNING PREPROCESSING --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "ANALYSIS: $ANALYSIS"
echo "SUBJECT: $SUBJECT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings at `CONFIG_PATH`
echo "Running MNE BIDS pipeline..."
mne_bids_pipeline --steps=source --config=$CONFIG_PATH