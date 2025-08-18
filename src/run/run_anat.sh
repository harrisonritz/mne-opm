# run all
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


## RUN ALL ------------------
echo ""
echo "-------------------------- RUNNING ALL --------------------------"
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


echo ""
echo "======================= NIFTI =============================================="
echo ""
source "$ROOT_DIR/src/run/run_nifti.sh"


echo ""
echo "======================= FREESURFER =============================================="
echo ""
source "$ROOT_DIR/src/run/run_freesurfer.sh"


echo ""
echo "======================= COREG =============================================="
echo ""
source "$ROOT_DIR/src/run/run_coreg.sh"

