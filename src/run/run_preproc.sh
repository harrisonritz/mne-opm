# run preprocessing
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg
export SESSION=01


## import variables
export EXPERIMENT=$1
if [ ! ${EXPERIMENT} ]; then
  echo "Error: please provide a experiment name"
  exit 1
fi

export ANALYSIS=$2
if [ ! ${ANALYSIS} ]; then
  echo "Error: please provide an analysis name"
  exit 1
fi

export SUBJECT=$3
if [ ! ${SUBJECT} ]; then
  echo "Error: please provide a subject number"
  exit 1
fi


## activate environment file
source ../experiments/$EXPERIMENT/$EXPERIMENT.env
if [ ! ${ROOT_DIR} ]; then
  echo "Error: please check that the environmental variables are available at '../{$EXPERIMENT}.env'"
  exit 1
fi



# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"
export RAW_DIR=$(eval echo $RAW_DIR)
export BIDS_DIR=$(eval echo $BIDS_DIR)
export SUBJECTS_DIR=$(eval echo $SUBJECTS_DIR)

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


# echo ""
# echo "======================= OSL: bad segment =============================================="
# echo ""
# python ../src/aux_preproc.py --analysis=bad_segments --config=$CONFIG_PATH



# echo ""
# echo "======================= OSL: bad channels =============================================="
# echo ""
# python ../src/aux_preproc.py --analysis=bad_channels --config=$CONFIG_PATH



# echo ""
# echo "======================= Manual: bad channel =============================================="
# echo ""
# python ../src/aux_preproc.py --analysis=manual_channel --config=$CONFIG_PATH



# echo ""
# echo "======================= MNE: preprocessing =============================================="
# echo ""
# mne_bids_pipeline --steps=preprocessing --config=$CONFIG_PATH



# echo ""
# echo "======================= Manual: ICA selection =============================================="
# echo ""
# python ../src/aux_preproc.py --analysis=manual_ica --config=$CONFIG_PATH



# echo ""
# echo "======================= MNE: apply ICA =============================================="
# echo ""
# mne_bids_pipeline --steps=preprocessing/apply_ica,preprocessing/apply_ssp,preprocessing/ptp_reject --config=$CONFIG_PATH



# echo ""
# echo "======================= OSL: bad epochs =============================================="
# echo ""
# python ../src/aux_preproc.py --analysis=bad_epochs --config=$CONFIG_PATH



echo ""
echo "======================= MNE: prep source space =============================================="
echo ""
mne_bids_pipeline --steps=sensor/make_evoked,sensor/make_cov,source/make_bem_solution,source/setup_source_space,source/make_forward --config=$CONFIG_PATH



