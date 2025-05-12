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
# config_path="../config/TSX-pilot/sub-${SUBJECT}/config-sensor_sub-${SUBJECT}.py"
config_path="../config/example/generic/config-sensor.py"


## run mne_bids_pipeline ----------------------------------------
# edit the configuration settings in `config`
echo "Running MNE BIDS pipeline..."
mne_bids_pipeline --steps=sensor --config=$config_path
