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
# config_path="../config/TSX-pilot/sub-${SUBJECT}/config-preproc_sub-${SUBJECT}.py"
config_path="../config/example/generic/config-preproc.py"


## find bad segments (OSL) ----------------------------------------
echo "---------------------- Running OSL preprocessing for bad segments detection ----------------------"
python ../scripts/aux_preproc.py --analysis=bad_segments --config=$config_path


## find bad channels (OSL) ----------------------------------------
echo "---------------------- Running OSL preprocessing for bad channels detection ----------------------"
python ../scripts/aux_preproc.py --analysis=bad_channels --config=$config_path


## manually select channels  ----------------------------------------
echo "---------------------- Manual bad channel selection ----------------------"
python ../scripts/aux_preproc.py --analysis=manual_channel --config=$config_path


## run mne_bids_pipeline ----------------------------------------
echo "---------------------- Running MNE BIDS pipeline ----------------------"
mne_bids_pipeline --steps=preprocessing --config=$config_path


## manually select ICA components  ----------------------------------------
echo "---------------------- Manual ICA selection ----------------------"
python ../scripts/aux_preproc.py --analysis=manual_ica --config=$config_path


## run mne_bids_pipeline ----------------------------------------
echo "---------------------- Re-Running MNE BIDS pipeline to remove ICA components ----------------------"
mne_bids_pipeline --steps=preprocessing/apply_ica,preprocessing/apply_ssp,preprocessing/ptp_reject --config=$config_path


# ## find bad epochs (OSL) ----------------------------------------
echo "---------------------- Running OSL post-processing for bad epochs detection ----------------------"
python ../scripts/aux_preproc.py --analysis=bad_epochs --config=$config_path