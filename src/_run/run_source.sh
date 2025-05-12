## activate environment ----------------------------------------
conda init
conda activate mne-opm
export MPLBACKEND=agg


# parameters ----------
export SUBJECT=$1
# ---------------------


# run checks
if [ ! ${SUBJECT} ]; then
  echo "Error: please provide a subject number"
  exit 1
fi


# set config
# config_path="../config/TSX-pilot/sub-${SUBJECT}/config-source_sub-${SUBJECT}.py"
config_path="../config/example/generic/config-source.py"


# # freesurfer variables
export FREESURFER_HOME=/Applications/freesurfer/8.0.0 			# freesurfer location
export FS_ALLOW_DEEP=1 											# NOTE: beta & might require NVIDIA GPU
source $FREESURFER_HOME/SetUpFreeSurfer.sh


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings in `config`
echo "Running MNE BIDS pipeline..."
mne_bids_pipeline --steps=source --config=$config_path