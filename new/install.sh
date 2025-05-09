# create env ----------------------------------------
conda create --name=mne-opm python=3.12


# activate env ----------------------------------------
conda activate mne-opm


# install dev mne-bids ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-bids.git


# install dev mne-bids-pipeline ----------------------------------------
# pip3 install git+https://github.com/mne-tools/mne-bids-pipeline.git
# pip install -e /Users/hr0283/Projects/mne-bids-pipeline
pip3 install git+https://github.com/harrisonritz/mne-bids-pipeline.git@custom_metadata

# install dev osl-ephys
pip3 install git+https://github.com/OHBA-analysis/osl-ephys.git


# install dev mne-python ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-python.git


# check installation
python -c "import mne; mne.sys_info()"


