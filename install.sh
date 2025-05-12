# create env ----------------------------------------
conda create --name=mne-opm python=3.12


# activate env ----------------------------------------
conda activate mne-opm


# basic install
# add packages here


# install dev mne-bids ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-bids.git


# install dev mne-bids-pipeline ----------------------------------------
# pip3 install git+https://github.com/mne-tools/mne-bids-pipeline.git
pip3 install git+https://github.com/harrisonritz/mne-bids-pipeline.git@custom_metadata


# install dev osl-ephys
pip3 install git+https://github.com/harrisonritz/osl-ephys.git


# install dev mne-python ----------------------------------------
# pip3 install git+https://github.com/mne-tools/mne-python.git
pip3 install git+https://github.com/harrisonritz/mne-python.git@opm-maxwell-er


# check installation
python -c "import mne; mne.sys_info()"


