# run automatic coregistration
# Harrison Ritz 2025

## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg

export SUBJECT="sub-${1}_ses-01"
export TASK=TSXpilot
export SESSION=01


## format data for bids ----------------------------------------
echo "running coregistration for subject $SUBJECT"

python ../scripts/auto_coreg.py
