# %% ZCA test

import mne
from preprocessing._config import load_config
from mne_bids import BIDSPath, get_head_mri_trans


# %% get paths

info = mne.io.read_info()
bem = mne.read_bem_solution()
src = mne.setup_source_space()
trans = get_head_mri_trans()


# %% compute forward

fwd = mne.make_forward_solution(
    info, trans=trans, src=src, bem=bem, mindist=cfg.mindist
)



# %% get inverse model

