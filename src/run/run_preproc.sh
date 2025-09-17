# run preprocessing
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm

## fixed variables
export MPLBACKEND=agg

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -euo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

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



echo ""
echo "======================= MNE: reference regression =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=regress_ref --config=$CONFIG_PATH



echo ""
echo "======================= OSL: bad segment =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_segments --config=$CONFIG_PATH



echo ""
echo "======================= OSL: bad channels =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_channels --config=$CONFIG_PATH



echo ""
echo "======================= Manual: bad channel =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=manual_channel --config=$CONFIG_PATH



echo ""
echo "======================= MNE: preprocessing =============================================="
echo ""
mne_bids_pipeline --steps=preprocessing --config=$CONFIG_PATH



echo ""
echo "======================= Manual: ICA selection =============================================="
echo ""
python  $ROOT_DIR/src/custom/custom_preproc.py --analysis=manual_ica --config=$CONFIG_PATH



echo ""
echo "======================= MNE: apply ICA =============================================="
echo ""
mne_bids_pipeline --steps=preprocessing/apply_ica,preprocessing/apply_ssp,preprocessing/ptp_reject --config=$CONFIG_PATH



echo ""
echo "======================= OSL: bad epochs =============================================="
echo ""
python  $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_epochs --config=$CONFIG_PATH



echo ""
echo "======================= MNE: prep source space =============================================="
echo ""
mne_bids_pipeline --steps=sensor/make_evoked,sensor/make_cov,source/make_bem_solution,source/setup_source_space,source/make_forward --config=$CONFIG_PATH



