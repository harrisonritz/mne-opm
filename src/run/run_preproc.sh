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


# timing helpers
PIPELINE_START=$SECONDS
_fmt_time() {
	local secs=$1
	printf "%dh %02dm %02ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}
_print_timing() {
	local step_name=$1
	local step_start=$2
	local step_elapsed=$((SECONDS - step_start))
	local total_elapsed=$((SECONDS - PIPELINE_START))
	echo ""
	echo "[TIMING] ${step_name}: $(_fmt_time $step_elapsed) | total: $(_fmt_time $total_elapsed)"
	echo ""
}


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


STEP_START=$SECONDS
echo ""
echo "======================= INIT: clear stale custom derivatives =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=init --config=$CONFIG_PATH
_print_timing "INIT: clear stale custom derivatives" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= CUSTOM: select first response per trial =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=select_trial_response --config=$CONFIG_PATH
_print_timing "CUSTOM: select first response per trial" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= custom: regress  =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=regress --config=$CONFIG_PATH
_print_timing "custom: regress" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= OSL: bad segment 1 =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_segments_1 --config=$CONFIG_PATH
_print_timing "OSL: bad segment 1" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= OSL: bad channels =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_channels --config=$CONFIG_PATH
_print_timing "OSL: bad channels" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= Manual: bad channel =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=manual_channel --config=$CONFIG_PATH
_print_timing "Manual: bad channel" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= CUSTOM: homogeneous field correction (HFC) =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=apply_hfc --config=$CONFIG_PATH
_print_timing "CUSTOM: homogeneous field correction (HFC)" $STEP_START


STEP_START=$SECONDS
echo ""
echo "======================= CUSTOM: Common Spatial Filter (ZCA) =============================================="
echo ""
python $ROOT_DIR/src/custom/custom_preproc.py --analysis=apply_zca --config=$CONFIG_PATH
_print_timing "CUSTOM: Common Spatial Filter (ZCA)" $STEP_START


STEP_START=$SECONDS
echo ""
echo "======================= MNE: preprocessing =============================================="
echo ""
mne_bids_pipeline --steps=preprocessing --config=$CONFIG_PATH
_print_timing "MNE: preprocessing" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= CUSTOM: Automatic ICA rejection =============================================="
echo ""
python  $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_ICs --config=$CONFIG_PATH
_print_timing "CUSTOM: automatic ICA rejection" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= Manual: ICA selection =============================================="
echo ""
python  $ROOT_DIR/src/custom/custom_preproc.py --analysis=manual_ica --config=$CONFIG_PATH
_print_timing "Manual: ICA selection" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= MNE: apply ICA =============================================="
echo ""
mne_bids_pipeline --steps=preprocessing/apply_ica,preprocessing/apply_ssp,preprocessing/ptp_reject --config=$CONFIG_PATH
_print_timing "MNE: apply ICA" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= OSL: bad epochs =============================================="
echo ""
python  $ROOT_DIR/src/custom/custom_preproc.py --analysis=bad_epochs --config=$CONFIG_PATH
_print_timing "OSL: bad epochs" $STEP_START



STEP_START=$SECONDS
echo ""
echo "======================= MNE: prep source space =============================================="
echo ""
mne_bids_pipeline --steps=sensor/make_evoked,sensor/make_cov,source/make_bem_solution,source/setup_source_space,source/make_forward --config=$CONFIG_PATH
_print_timing "MNE: prep source space" $STEP_START




STEP_START=$SECONDS
echo ""
echo "======================= BEAMFORMER =============================================="
echo ""
source "$ROOT_DIR/src/run/run_beamformer.sh"
_print_timing "BEAMFORMER" $STEP_START



