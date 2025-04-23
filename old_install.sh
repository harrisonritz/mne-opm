# create env ----------------------------------------
conda create --name=mne-opm python=3.12


# activate env ----------------------------------------
conda activate mne-opm


# install basic packages ----------------------------------------
pip3 install pandas


# install mne-python ----------------------------------------
# shouldnt need to install ahead of time, but can try this in case you're having issues
# pip3 install git+https://github.com/mne-tools/mne-python.git


# install mne-bids ----------------------------------------
# pip3 install --no-cache-dir https://github.com/mne-tools/mne-bids/zipball/main
pip3 install git+https://github.com/mne-tools/mne-bids.git


# install harrison's mne-bids-pipeline ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-bids-pipeline.git


# install harrison's mne-python again (revert to my mne version) ----------------------------------------
# this has to be installed last
pip3 install git+https://github.com/harrisonritz/mne-python.git@colocated_topo


# check installation
python -c "import mne; mne.sys_info()"
