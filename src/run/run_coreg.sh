# run automatic coregistration
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg


# ## import variables
# export EXPERIMENT=$1
# if [ ! ${EXPERIMENT} ]; then
#   echo "Error: please provide a experiment name"
#   exit 1
# fi

# export SUBJECT="sub-$2_ses-$SESSION"
# if [ ! ${SUBJECT} ]; then
#   echo "Error: please provide a subject number"
#   exit 1
# fi


# ## activate environment file
# source ../experiments/$EXPERIMENT/$EXPERIMENT.env
# if [ ! ${ROOT_DIR} ]; then
#   echo "Error: please check that the environmental variables are available at '../{$EXPERIMENT}.env'"
#   exit 1
# fi

# export RAW_DIR=$(eval echo $RAW_DIR)
# export BIDS_DIR=$(eval echo $BIDS_DIR)
# export SUBJECTS_DIR=$(eval echo $SUBJECTS_DIR)


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
