# %%
import matplotlib
import mne
import mne_bids


fn = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/bids/sub-008/ses-01/meg/sub-008_ses-01_task-TSXpilot_run-01_meg.fif"

# %% LOAD BIDS --------------------------------------------------------------------

print('standard raw')
raw = mne.io.read_raw_fif(fn, preload=True)
print(f'example OPM: {raw.info["chs"][raw.ch_names.index("T2 2L Z")]}')
print(f'xpos_right: {raw.info["chs"][raw.ch_names.index("xpos_right")]}')

raw.info



print('mne_bids raw')
# NOT LOADING COIL TYPES FOR EYETRACKING!
raw = mne_bids.read_raw_bids(
    bids_path=mne_bids.BIDSPath(
        subject="008",
        session="01",
        task="TSXpilot",
        run="01",
        root="/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/bids",
    ),
)

print(f'xpos_right: {raw.info["chs"][raw.ch_names.index("xpos_right")]}')
print(f'x_head: {raw.info["chs"][raw.ch_names.index("x_head")]}')

raw.info


chinds = mne.pick_types(raw.info, meg=False, ref_meg=False, eyetrack=True, exclude='bads')


# %% SAVE BID --------------------------------------------------------------------

print('standard raw')
raw = mne.io.read_raw_fif(fn, preload=True)
# print(f'example OPM: {raw.info["chs"][raw.ch_names.index("T2 2L Z")]}')
# print(f'xpos_right: {raw.info["chs"][raw.ch_names.index("xpos_right")]}')

raw.info


test_path = mne_bids.BIDSPath(
    root='/Users/hr0283/Projects/mne-opm/src/dev/bids_test',
    subject='test',
    session='01',
    task='test',
    run='01',
    suffix='meg',
    extension='.fif',
    )

save_path = mne_bids.write_raw_bids(
    raw.copy().crop(tmax=120),
    test_path,
    overwrite=True,
    allow_preload=True,
    format="FIF",
)




print('mne_bids raw')
# NOT LOADING COIL TYPES FOR EYETRACKING!
raw = mne_bids.read_raw_bids(save_path, verbose='INFO')

# print(f'xpos_right: {raw.info["chs"][raw.ch_names.index("xpos_right")]}')

raw.info



# %% PLOT --------------------------------------------------------------------

%matplotlib qt

stims = raw.copy().crop(tmax=100).plot()

stims = raw.copy().crop(tmax=100).get_data(picks='stim')
# %%