# run automatic coregistration
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg

old_sub=$SUBJECT
export SUBJECT=${SUBJECT_NUM}_ses-${SESSION}


## COREGISTRATION ----------------------------------------
echo ""
echo "RUNNING COREGISTRATION --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "SUBJECT: $SUBJECT"
echo "SESSION: $SESSION"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""


python $ROOT_DIR/src/custom/auto_coreg.py


# revert to original subject formatting
export SUBJECT=$old_sub
