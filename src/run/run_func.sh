# run all
# Harrison Ritz 2025


## set config ----------------------------------------
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"
export MPLBACKEND=agg


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
echo "======================= BIDS =============================================="
echo ""
source "$ROOT_DIR/src/run/run_bids.sh"


echo ""
echo "======================= COREG =============================================="
echo ""
source "$ROOT_DIR/src/run/run_coreg.sh"


echo ""
echo "======================= PREPROCESSING =============================================="
echo ""
source "$ROOT_DIR/src/run/run_preproc.sh"


echo ""
echo "======================= SENSOR =============================================="
echo ""
source "$ROOT_DIR/src/run/run_sensor.sh"


# echo ""
# echo "======================= SOURCE =============================================="
# echo ""
# source "$ROOT_DIR/src/run/run_source.sh"


echo ""
echo "======================= BEAMFORMER =============================================="
echo ""
source "$ROOT_DIR/src/run/run_beamformer.sh"
