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

# SUBJECT stays in MNE format (just the number, e.g., "008") for config loading
# coreg.py constructs the FreeSurfer subject name (sub-XX_ses-YY) internally
# Zero-pad subject to 3 digits for consistency
SUBJECT_PADDED=$(printf "%03d" $((10#${SUBJECT})))
export SUBJECT=$SUBJECT_PADDED


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


## prep source space ----------------------------------------
echo ""
echo "======================= MNE: prep source space =============================================="
echo ""
mne_bids_pipeline --steps=source/make_bem_solution,source/setup_source_space --config=$CONFIG_PATH


