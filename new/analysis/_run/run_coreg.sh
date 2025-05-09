# run automatic coregistration
# Harrison Ritz 2025

## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg

export SUBJECT="sub-004_ses-01"
export TASK=TSXpilot
export SESSION=01


# export FREESURFER_HOME=/Applications/freesurfer/8.0.0 			# freesurfer location
# export FS_ALLOW_DEEP=1 											# NOTE: beta & might require NVIDIA GPU
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export SUBJECTS_DIR=/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/freesurfer/subjects # save location


## format data for bids ----------------------------------------
python ../scripts/auto_coreg.py
