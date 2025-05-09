## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg

# parameters
subject=003
config_path="../config/TSX-pilot/sub-${subject}/config-source_sub-${subject}.py"

# # freesurfer variables
export FREESURFER_HOME=/Applications/freesurfer/8.0.0 			# freesurfer location
export FS_ALLOW_DEEP=1 											# NOTE: beta & might require NVIDIA GPU
source $FREESURFER_HOME/SetUpFreeSurfer.sh


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings in `config`
echo "Running MNE BIDS pipeline..."
mne_bids_pipeline --steps=source --config=$config_path