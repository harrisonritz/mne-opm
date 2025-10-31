# old installation

# NOTE: use UV instead
echo "USE UV INSTEAD!!"
return


# create env ----------------------------------------
conda create --name=mne-opm python=3.12


# activate env ----------------------------------------
conda activate mne-opm


# add packages here
pip3 install dcm2niix
# pip3 install dotenv # probably don't need
pip3 install ipykernel
pip3 install mne-qt-browser


# pygam ----------------------------------------
# pip3 install git+https://github.com/harrisonritz/pyGAM.git@mne-opm

# pip3 install scikit-sparse
# pip3 install nose
# pip3 install progressbar


# mne-bids (dev) ----------------------------------------
# pip3 install git+https://github.com/mne-tools/mne-bids.git
pip3 install git+https://github.com/harrisonritz/mne-bids.git@mne-opm


# mne-bids-pipeline (dev) ----------------------------------------
pip3 install git+https://github.com/mne-tools/mne-bids-pipeline.git


# osl-ephys (dev) ----------------------------------------
pip3 install git+https://github.com/harrisonritz/osl-ephys.git


# mne-python (dev) ----------------------------------------
# NOTE: re-run if you update other packages, as they will set to a specific mne version
pip3 install git+https://github.com/harrisonritz/mne-python.git@mne-opm


# check installation
python -c "import mne; mne.sys_info()"


