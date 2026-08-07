# run the osl-ephys pipeline
# Harrison Ritz 2025


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi


# The osl-ephys pipeline is configured with YAML rather than the Python configs
# used by the mne-bids-pipeline stages, and lives in its own config subdirectory.
export CONFIG_PATH="${OSL_CONFIG_PATH:-$CONFIG_DIR/osl/$ANALYSIS.yaml}"

# preproc | source | all | collate.  Set with --stage, or export OSL_STAGE.
OSL_STAGE="${OSL_STAGE:-all}"


# timing helpers
STAGE_START=$SECONDS
_fmt_time() {
	local secs=$1
	printf "%dh %02dm %02ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}


## RUN OSL-EPHYS ------------------
echo ""
echo "RUNNING OSL-EPHYS PIPELINE --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "ANALYSIS: $ANALYSIS"
echo "SUBJECT: $SUBJECT"
echo "STAGE: $OSL_STAGE"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""


python "$ROOT_DIR/src/custom/run_osl.py" \
	--stage="$OSL_STAGE" \
	--config="$CONFIG_PATH"

OSL_EXIT=$?

echo ""
echo "[TIMING] osl-ephys ${OSL_STAGE}: $(_fmt_time $((SECONDS - STAGE_START)))"
echo ""

if [ "$OSL_EXIT" -ne 0 ]; then
	echo "[osl] stage '${OSL_STAGE}' failed with exit status ${OSL_EXIT}"
fi

# Make the stage's status the caller's status.  This script is sourced by
# mne-opm.sh, so a subshell that exits with the code is the way to set $?
# without terminating the caller.
( exit "$OSL_EXIT" )
