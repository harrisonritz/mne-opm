# %%
import matplotlib
import mne
import numpy as np

# %% load data
raw = mne.io.read_raw_fif(
    "/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/TSXpilot/raw/2025-07-11_TSX-Pilot_008/20250711_140149_cMEG_Data_task/20250711_140149_meg.fif",
    preload=True,
)
eye_path = "/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/TSXpilot/raw/2025-07-11_TSX-Pilot_008/eyetracker/E008.asc"

# %%

# load data
print("Loading eye-tracking data from: ", eye_path)
eye = mne.io.read_raw_eyelink(eye_path, create_annotations=True)


# set calibration info
cal = mne.preprocessing.eyetracking.read_eyelink_calibration(eye_path)[0]
cal["screen_resolution"] = (1920, 1080)
cal["screen_size"] = (0.53, 0.3)
cal["screen_distance"] = 0.9
mne.preprocessing.eyetracking.convert_units(eye, calibration=cal, to="radians")

# align from events
eye_events, eye_id = mne.events_from_annotations(eye, regexp="response")
eye_shape = eye_events.shape[0]
print("eye events: ", eye_shape)

mag_events, mag_id = mne.events_from_annotations(raw, regexp="BNC")
mag_shape = mag_events.shape[0]
print("mag events: ", mag_shape)

eye_0 = 0 if eye_shape < mag_shape else eye_shape - mag_shape
mag_0 = 0 if mag_shape < eye_shape else mag_shape - eye_shape

eye_times = eye_events[eye_0:, 0] / eye.info["sfreq"]
mag_times = mag_events[mag_0:, 0] / raw.info["sfreq"]

mne.preprocessing.realign_raw(raw, eye, mag_times, eye_times, verbose="error")


# add channels to raw
raw.add_channels([eye], force_update_info=True)
del eye


if "xpos_right" in raw.ch_names:
    mne.preprocessing.eyetracking.set_channel_types_eyetrack(
        raw,
        {
            "xpos_right": ("eyegaze", "rad", "right", "x"),
            "ypos_right": ("eyegaze", "rad", "right", "y"),
            "pupil_right": ("pupil", "rad", "right"),
        },
    )
else:
    mne.preprocessing.eyetracking.set_channel_types_eyetrack(
        raw,
        {
            "xpos_left": ("eyegaze", "rad", "left", "x"),
            "ypos_left": ("eyegaze", "rad", "left", "y"),
            "pupil_left": ("pupil", "rad", "left"),
        },
    )


# %%

raw.info["chs"][-6]

# set channel type to eyetrack  # noqa: F821
# include channel type?
#       https://vscode.dev/github/harrisonritz/mne-opm/blob/devmne-opm/lib/python3.12/site-packages/mne/_fiff/meas_info.py#L448
#       mne > _fiff > meas_info.py > _unit2human


# %%
