# run coregistration diagnostics
# Harrison Ritz 2026

# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -euo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi


## RUN COREG DIAGNOSTICS ------------------
echo ""
echo "RUNNING COREG DIAGNOSTICS --------------------------"
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


# Render BEM, alignment, head-point distance and sensitivity-map figures
# and save them under {deriv_root}/sub-XX/ses-YY/meg/coreg_diagnostics/.
echo ""
echo "======================= COREG DIAGNOSTICS =============================================="
echo ""
python "$ROOT_DIR/src/custom/coreg_diagnostics.py" --config="$CONFIG_PATH"
