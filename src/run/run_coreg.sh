# run automatic coregistration
# Harrison Ritz 2025


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"

# FreeSurfer requires SUBJECT in sub-XX_ses-XX format
# MNE requires SUBJECT as just XX
# Swap for coreg step, then restore
old_sub=$SUBJECT
export SUBJECT=${SUBJECT_NUM}_ses-${SESSION}


## COREGISTRATION ----------------------------------------
echo ""
echo "RUNNING COREGISTRATION --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "SUBJECT: $SUBJECT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "--------------------------"
echo ""


# Run coregistration via custom preprocessing dispatcher
python $ROOT_DIR/src/custom/custom_preproc.py \
    --analysis=coreg \
    --config=$CONFIG_PATH


# Restore original subject formatting for MNE
export SUBJECT=$old_sub


## prep source space ----------------------------------------
echo ""
echo "======================= MNE: prep source space =============================================="
echo ""
mne_bids_pipeline --steps=source/make_bem_solution,source/setup_source_space --config=$CONFIG_PATH


