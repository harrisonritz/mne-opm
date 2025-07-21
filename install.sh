# create env ----------------------------------------
conda create --name=mne-opm python=3.12


# activate env ----------------------------------------
conda activate mne-opm


# basic install
# add packages here
pip3 install dotenv
pip3 install ipykernel


# mne-bids (dev) ----------------------------------------
pip3 install git+https://github.com/harrisonritz/mne-bids.git@T2w


# mne-bids-pipeline (dev) ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-bids-pipeline.git


# osl-ephys (dev) ----------------------------------------
pip3 install git+https://github.com/harrisonritz/osl-ephys.git


# mne-python (dev) ----------------------------------------
# re-run if you update other packages, as they will set to a specific mne version
# pip3 install git+https://github.com/mne-tools/mne-python.git
pip3 install git+https://github.com/harrisonritz/mne-python.git@mne-opm


# check installation
python -c "import mne; mne.sys_info()"


