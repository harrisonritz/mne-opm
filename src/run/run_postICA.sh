# run preprocessing
# Harrison Ritz 2025


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


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



echo ""
echo "======================= BEAMFORMER =============================================="
echo ""
source "$ROOT_DIR/src/run/run_beamformer.sh"