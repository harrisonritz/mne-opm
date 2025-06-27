# run sensor-level analysis
# Harrison Ritz 2025


## activate environment ----------------------------------------
source activate mne-opm

## fixed variables
export MPLBACKEND=agg
# export SESSION=01


## import variables
# export EXPERIMENT=$1
# if [ ! ${EXPERIMENT} ]; then
#   echo "Error: please provide a experiment name"
#   exit 1
# fi

# export ANALYSIS=$2
# if [ ! ${ANALYSIS} ]; then
#   echo "Error: please provide an analysis name"
#   exit 1
# fi

# export SUBJECT=$3
# if [ ! ${SUBJECT} ]; then
#   echo "Error: please provide a subject number"
#   exit 1
# fi


## activate environment file
# source ../experiments/$EXPERIMENT/$EXPERIMENT.env
# if [ ! ${ROOT_DIR} ]; then
#   echo "Error: please check that the environmental variables are available at '../{$EXPERIMENT}.env'"
#   exit 1
# fi



# set config
# export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"
# export RAW_DIR=$(eval echo $RAW_DIR)
# export BIDS_DIR=$(eval echo $BIDS_DIR)
# export SUBJECTS_DIR=$(eval echo $SUBJECTS_DIR)

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
mne_bids_pipeline --steps=sensor --config=$CONFIG_PATH
