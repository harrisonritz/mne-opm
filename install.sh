# create env ----------------------------------------
conda create --name=mne-opm 


# activate env ----------------------------------------
conda activate mne-opm


# install mne-python ----------------------------------------
pip3 install git+https://github.com/harrisonritz/mne-python.git@colocated_topo


# install mne-bids ----------------------------------------
pip3 install https://github.com/mne-tools/mne-bids/zipball/main


# install harrison's mne-bids-pipeline ----------------------------------------
pip3 install git+https://github.com/harrisonritz/mne-bids-pipeline@compute_rank


# install harrison's mne-python again (revert to my mne version) ----------------------------------------
pip3 install git+https://github.com/harrisonritz/mne-python.git@colocated_topo


# check installation
python -c "import mne; mne.sys_info()"
