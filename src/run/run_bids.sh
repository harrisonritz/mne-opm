# run BIDS configuration
# Harrison Ritz 2025


## activate environment ----------------------------------------
# source activate base
source activate mne-opm

## import variables
# export EXPERIMENT=$1
# if [ ! ${EXPERIMENT} ]; then
#   echo "format: sh run_bids.sh EXPERIMENT SUBJECT"  
#   echo "please provide a experiment name"
#   exit 1
# fi

# export SUBJECT=$2
# if [ ! ${SUBJECT} ]; then
#   echo "format: sh run_bids.sh EXPERIMENT SUBJECT"
#   echo "Error: please provide a subject number"
#   exit 1
# fi


# ## activate environment file
# source ./config/$EXPERIMENT/$EXPERIMENT.env
# if [ ! ${ROOT_DIR} ]; then
#   echo "Error: please check that the environmental variables are available at 'config/{$EXPERIMENT}.env'"
#   exit 1
# fi


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