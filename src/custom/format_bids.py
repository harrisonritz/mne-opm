## Convert Cerca OPM data to BIDS format.
# Harrison Ritz (2025)


# TODO
# - update tigger/annotation mapping


# %% import -------------------------------------------------------------------

# from dotenv import load_dotenv, find_dotenv
# import matplotlib
# import sys
import mne
import mne_bids
import os
import numpy as np
import yaml
import glob
import argparse
from types import SimpleNamespace
from mne_bids_pipeline._config_import import _update_config_from_path
from mne._fiff.meas_info import _merge_info

import mne_qt_browser

mne.set_config("MNE_BROWSER_BACKEND", "qt")

# %% import parameters


def set_bids_params(config_path=""):
    # set-up configuration ==========================================================================================================
    print("\n\n\n[loading configuration]\n")
    RAW_DIR = f"{os.environ.get('RAW_DIR')}"
    BIDS_DIR = f"{os.environ.get('BIDS_DIR')}"

    # Create flat configuration as SimpleNamespace with all parameters at top level
    config = SimpleNamespace(
        # Directory paths
        raw_dir=RAW_DIR,
        bids_dir=BIDS_DIR,
        # Session information
        ids=0,
        task="",
        session="",
        # Trigger information
        rename_annot=True,
        trigger_desc={},
        response_desc={},
        # Recording information
        line_freq=60.0,
        bads=[],
    )

    # Load config file if provided
    if config_path:
        print(f"\n\nloading config from Python file: {config_path}\n")
        # Use mne_bids_pipeline's function to update config from Python file
        try:
            _update_config_from_path(config=config, config_path=config_path)
        except Exception as e:
            print(
                f"error loading config from Python file: {e},\ncreating new config from template"
            )
            template_path = os.path.join(
                os.path.dirname(config_path), "TEMPLATE_config-bids.py"
            )
            # make a copy of the .py file at template_path, and save to config_path
            with open(template_path, "r") as template_file:
                template_content = template_file.read()
            with open(config_path, "w") as config_file:
                config_file.write(template_content)
            _update_config_from_path(config=config, config_path=config_path)

    # Return the config directly as SimpleNamespace (no need to convert to dict anymore)
    print('\nconfig:"\n', config)

    return config


def convert_triggers(raw, cfg):
    """
    Converts trigger data from multiple channels into a combined trigger channel and annotations.

    Parameters:
        raw (mne.io.Raw): Raw data to process
        cfg (SimpleNamespace): Configuration with trigger_desc and response_desc

    Returns:
        mne.io.Raw: Raw data with updated annotations
    """

    print("\n\n\nconverting triggers ----------------------\n")
    # Extract data from trigger channels
    trigger_channels = [f"Trigger {i}" for i in range(1, 9)]

    trigger_data = []
    for ch_name in trigger_channels:
        trigger_data.append(raw.get_data(ch_name))

    # Stack the channels together (shape: n_channels x n_timepoints)
    stacked_triggers = np.vstack(trigger_data)

    # convert to binary
    stacked_triggers[stacked_triggers < 2] = 0
    stacked_triggers[stacked_triggers > 2] = 1

    # convert to binary pattern to integer at each time point
    powers_of_two = (
        2 ** np.arange(len(trigger_channels))[:, np.newaxis]
    )  # Column vector
    integer_triggers = np.sum(stacked_triggers * powers_of_two, axis=0).astype(int)

    # add combined trigger to raw
    raw.add_channels(
        [
            mne.io.RawArray(
                integer_triggers.reshape(1, -1),
                mne.create_info(["Trigger Combined"], raw.info["sfreq"], ["stim"]),
            )
        ],
        force_update_info=True,
    )

    # extract events
    events = mne.find_events(
        raw, stim_channel="Trigger Combined", min_duration=0.001, consecutive=True
    )

    # convert to annotation
    new_annotations = mne.annotations_from_events(
        events,
        event_desc=cfg.trigger_desc,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )

    # Remove annotations related to trigger channels
    old_annotations = raw.copy().annotations
    trigger_ch_mask = np.ones(len(old_annotations), dtype=bool)
    for i, description in enumerate(old_annotations.description):
        for ch_name in trigger_channels:
            if ch_name in description:
                trigger_ch_mask[i] = False
                break

    # Keep only annotations not related to trigger channels
    old_annotations = mne.Annotations(
        onset=old_annotations.onset[trigger_ch_mask],
        duration=old_annotations.duration[trigger_ch_mask],
        description=old_annotations.description[trigger_ch_mask],
        orig_time=old_annotations.orig_time,
    )

    # Set the new annotations
    raw.set_annotations(old_annotations + new_annotations)

    # rename response annotations
    raw.annotations.rename(cfg.response_desc)

    print("Trigger & Response conversion completed.\n----------\n")
    return raw


def bids_conversion(cfg):
    """
    Converts raw OPM MEG data files to BIDS format using configuration parameters provided in the cfg namespace.
    This function performs the following steps:
        1. Extracts necessary configuration parameters such as subject ID, session information, and task name.
        2. Locates the empty room file, task file(s), and anatomical scan file using glob pattern matching.
        3. Reads and processes the empty room file if found. The empty room raw data is read, its line
           frequency updated based on cfg.line_freq, bad channels set, and then
           written to a BIDS-compatible directory structure.
        4. Processes task files:
            - Reads the raw data and updates its metadata (line frequency and subject information).
            - Appends the processed raw data for later concatenation.
        5. Concatenates the individual raw run data into a single raw object and prints the recording duration.
        6. Converts triggers and renames annotations if cfg.rename_annot is True using the
           convert_triggers function.
        7. Sets bad channels from the configuration.
        8. Writes the concatenated raw data to the BIDS directory with empty room reference if available.
        9. If an anatomical scan is found, writes the anatomical image to the BIDS structure.

    Parameters:
        cfg (SimpleNamespace): A configuration namespace containing settings required for the conversion
            with all parameters as top-level attributes.

    Returns:
        None
    """

    # %% convert to BIDS ---------------------------------------------------------

    subj = cfg.ids
    task = cfg.task

    emptyroom_path = glob.glob(
        os.path.join(cfg.raw_dir, f"*_{subj:03}", "*_noise", "*_meg.fif")
    )  # Take the first match
    emptyroom_path = (
        emptyroom_path[0] if emptyroom_path else False
    )  # Take the first match or False if not found

    task_path = glob.glob(
        os.path.join(cfg.raw_dir, f"*_{subj:03}", "*_task", "*_meg.fif")
    )  # Take the first match
    task_path = task_path if task_path else False
    if not task_path:
        raise FileNotFoundError(
            f"No task files found for subject {subj} in {cfg.raw_dir}. Please check the directory structure."
        )

    # T1w
    t1w_path = glob.glob(
        os.path.join(cfg.raw_dir, f"*_{subj:03}", "*", "*_t1w.nii*")
    )  # Take the first match
    t1w_path = (
        t1w_path[0] if t1w_path else False
    )  # Take the first match or False if not found

    # T2w
    t2w_path = glob.glob(
        os.path.join(cfg.raw_dir, f"*_{subj:03}", "*", "*_t2w.nii*")
    )  # Take the first match
    t2w_path = (
        t2w_path[0] if t2w_path else False
    )  # Take the first match or False if not found

    # eyetracking
    eye_path = glob.glob(
        os.path.join(cfg.raw_dir, f"*_{subj:03}", "*", "*.asc")
    )  # Take the first match
    eye_path = (
        eye_path[0] if eye_path else False
    )  # Take the first match or False if not found

    raw_list = list()
    print(
        "\nparticipant: ",
        subj,
        "\ntask: ",
        task,
        "\ndata dir: ",
        cfg.raw_dir,
        "\nbids dir: ",
        cfg.bids_dir,
        "\ntask path: ",
        task_path,
        "\nemptyroom path: ",
        emptyroom_path,
        "\nT1w path: ",
        t1w_path,
        "\nT2w path: ",
        t2w_path,
        "\nEye-tracking path: ",
        eye_path,
        "\n--------\n\n",
    )

    # Process empty room data ------------------------------------------------

    if emptyroom_path:
        raw_empty_room = mne.io.read_raw_fif(emptyroom_path)
        raw_empty_room.info["line_freq"] = cfg.line_freq

        # set bad channels
        if cfg.bads:
            raw_empty_room.info["bads"] = cfg.bads

        # make bids path
        emptyroom_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg.session,
            task="noise",
            root=cfg.bids_dir,
        )

        # Write empty room data to BIDS
        mne_bids.write_raw_bids(
            raw_empty_room,
            emptyroom_bids_path,
            allow_preload=True,
            overwrite=True,
            events=None,
            format="FIF",
        )

    # Concatenate data for this subject -----------------------------------------

    for rr, (fn) in enumerate(task_path):
        # get path
        raw_rr = mne.io.read_raw_fif(fn)

        raw_rr.info["line_freq"] = cfg.line_freq
        raw_rr.info["subject_info"] = {
            "id": int(subj),
            "his_id": f"{subj:03}",
        }

        # append raw to list
        raw_list.append(raw_rr)
        del raw_rr

    # Concatenate raws for all runs of this subject
    raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")
    # crop
    if cfg.crop > 0:
        print(f"*****Cropping first {cfg.crop} seconds of raw data")
        raw.crop(tmin=cfg.crop, tmax=None)
        print(f"raw after cropping: {raw.first_time}")

    recording_duration = raw.times[-1] - raw.times[0]
    print(
        f"\n\n*************\nRecording duration for subject {subj}: {(recording_duration / 60):.2f} minutes\n*************\n\n"
    )

    # Rename annotations
    if cfg.rename_annot:
        raw = convert_triggers(raw, cfg)

    # set bad channels
    if cfg.bads:
        raw.info["bads"] = cfg.bads

    # get eyetracking data TODO: split off into separate function --------------------------------------------------------
    if eye_path:
        print("\n\n\nformatting eyetracker ----------------------\n")

        # load data
        print("\nLoading eye-tracking data from: ", eye_path, "...")
        eye = mne.io.read_raw_eyelink(
            eye_path, create_annotations=True, apply_offsets=True, find_overlaps=True
        )

        # set calibration info
        print("\nCalibrating recording...")
        try:
            cals = mne.preprocessing.eyetracking.read_eyelink_calibration(eye_path)
            print(f"found {len(cals)}, using first one")
            cal = cals[0]  # take first calibration
        except Exception as e:
            print("***** error reading eyelink calibration:", e)
            print("warning: assuming zero calibration error")
            cal = mne.preprocessing.eyetracking.Calibration(
                onset=0,
                model="HV13",
                eye="right",
                avg_error=0.0,
                max_error=0.0,
                positions=None,
                offsets=None,
                gaze=None,
            )
        cal["screen_resolution"] = (1920, 1080)
        cal["screen_size"] = (0.606, 0.341)
        cal["screen_distance"] = 0.895
        print("calibration:")
        print(cal)
        mne.preprocessing.eyetracking.convert_units(eye, calibration=cal, to="radians")

        # set onset times
        # raw.set_meas_date(None)
        # raw._cropped_samp = 0
        # eye.set_meas_date(None)
        # eye._cropped_samp = 0

        # interpolate blinks

        # for ch_name in eye.ch_names:
        #     ch_data = eye[ch_name][0]
        #     nans = np.isnan(ch_data).sum()
        #     print(f"Channel '{ch_name}': {nans} NaN timepoints")

        # print()
        # mne.preprocessing.eyetracking.interpolate_blinks(eye,
        #                                                  buffer=(0.05, 0.1),
        #                                                  match=['BAD_blink'],
        #                                                  interpolate_gaze=True,
        #                                                 )

        buffer = 0.1
        buffer_samp = int(buffer * eye.info["sfreq"])
        print("\nInterpolating remaining nans (buffer = ", buffer, " sec)...")
        orig_nan = np.isnan(eye.get_data()).any(axis=0)
        data = eye.get_data()  # Get all data at once
        for ch_idx, ch_name in enumerate(eye.ch_names):
            # get nan indices
            ch_data = data[ch_idx, :]

            # Create masks for NaN values and their buffer regions before and after
            nan_mask = np.isnan(ch_data)  # Current NaN positions

            # Mask for positions where any of the next buffer_samp samples are NaN
            future_nan_mask = np.zeros_like(nan_mask, dtype=bool)
            for i in range(1, buffer_samp + 1):
                if i < len(ch_data):
                    future_nan_mask[:-i] |= nan_mask[i:]

            # Mask for positions where any of the previous buffer_samp samples are NaN
            past_nan_mask = np.zeros_like(nan_mask, dtype=bool)
            for i in range(1, buffer_samp + 1):
                if i < len(ch_data):
                    past_nan_mask[i:] |= nan_mask[:-i]

            # Combine all masks
            nan_mask = nan_mask | future_nan_mask | past_nan_mask

            # Skip if no NaN values
            if not np.any(nan_mask):
                continue

            # Skip if all values are NaN
            if np.all(nan_mask):
                print(
                    f"Warning: All values are NaN for channel '{ch_name}', skipping interpolation"
                )
                continue

            # Get valid (non-NaN) data points
            nan_indices = np.where(nan_mask)[0]
            valid_indices = np.where(~nan_mask)[0]

            # Only interpolate if we have enough non-NaN values
            if len(valid_indices) > 1:
                # Linear interpolation using valid timepoints and data
                interpolated_values = np.interp(
                    nan_indices,  # x-coordinates where we want interpolated values
                    valid_indices,  # x-coordinates of known data points
                    ch_data[valid_indices],  # y-coordinates of known data points
                )
                # from scipy.interpolate import CubicSpline
                # cs = CubicSpline(valid_indices, ch_data[valid_indices])
                # interpolated_values = cs(nan_indices)

                # Update the data in place
                data[ch_idx, nan_indices] = interpolated_values
            elif len(valid_indices) == 1:
                # If only one valid point, fill NaNs with that value
                data[ch_idx, nan_indices] = ch_data[valid_indices[0]]
                print(
                    f"Warning: Only one valid data point for channel '{ch_name}', using constant interpolation"
                )
            else:
                print(
                    f"Warning: No valid data points for interpolation in channel '{ch_name}'"
                )

        # Update the raw object with interpolated data
        eye._data = data

        print("Removing 'BAD_' from BAD_blink.")
        eye.annotations.rename({"BAD_blink": "blink"})

        #  create NaN EOG channel
        print("\nAdding a NaN channel...")
        nan_channel = (
            np.convolve(orig_nan, np.hanning(int(0.05 * eye.info["sfreq"])), "same")[
                np.newaxis, :
            ]
            * 1e-5
        )
        nan_info = mne.create_info(["eye_nan"], eye.info["sfreq"], ch_types="eog")
        nan_array = mne.io.RawArray(
            nan_channel, nan_info, first_samp=eye.first_samp, copy="auto"
        )
        # eye.add_channels([nan_array], force_update_info=True)

        # create blink EOG channel
        print("\nAdding a blink channel...")
        blink_channel = np.zeros(eye._data.shape[1])
        for aa, (ann,) in enumerate(zip(eye.annotations)):
            if ann["description"] == "blink":
                onset = int((ann["onset"] - eye.first_time) * eye.info["sfreq"])
                duration = int(np.ceil(ann["duration"] * eye.info["sfreq"]))
                blink_channel[onset : onset + duration] = 1.0
        blink_channel = (
            np.convolve(
                blink_channel, np.hanning(int(0.05 * eye.info["sfreq"])), "same"
            )[np.newaxis, :]
            * 1e-5
        )
        blink_info = mne.create_info(["blink"], eye.info["sfreq"], ch_types="eog")
        blink_array = mne.io.RawArray(
            blink_channel, blink_info, first_samp=eye.first_samp, copy="auto"
        )
        # eye.add_channels([blink_array], force_update_info=True)

        # create saccade EOG channel
        print("\nAdding a saccade channel...")
        saccade_channel = np.zeros(eye._data.shape[1])
        for aa, (ann,) in enumerate(zip(eye.annotations)):
            if ann["description"] == "saccade":
                onset = int((ann["onset"] - eye.first_time) * eye.info["sfreq"])
                duration = int(np.ceil(ann["duration"] * eye.info["sfreq"]))
                saccade_channel[onset : onset + duration] = 1.0
        saccade_channel = (
            np.convolve(
                saccade_channel, np.hanning(int(0.05 * eye.info["sfreq"])), "same"
            )[np.newaxis, :]
            * 1e-5
        )
        saccade_info = mne.create_info(["saccade"], eye.info["sfreq"], ch_types="eog")
        saccade_array = mne.io.RawArray(
            saccade_channel, saccade_info, first_samp=eye.first_samp, copy="auto"
        )
        # eye.add_channels([saccade_array], force_update_info=True)

        # Use NMF on the NaN, blink, saccade channels, and add each component as an EOG channel
        print("\nAdding eye channels to eye-tracking data using NMF...")
        nmf_data = np.vstack([nan_channel, blink_channel, saccade_channel])
        nmf_data = np.clip(nmf_data, 0.0, None)  # ensure non-negativity
        n_comp = min(3, nmf_data.shape[0])
        try:
            from sklearn.decomposition import NMF

            nmf = NMF(
                n_components=n_comp, init="nndsvda", random_state=99, max_iter=500
            )
            W = nmf.fit_transform(nmf_data)  # shape: (3, n_comp)
            H = nmf.components_  # shape: (n_comp, n_times)
            for k in range(n_comp):
                comp_ts = H[k][np.newaxis, :]
                comp_info = mne.create_info(
                    [f"eye_nmf{k + 1}"], eye.info["sfreq"], ch_types="eog"
                )
                comp_array = mne.io.RawArray(
                    comp_ts, comp_info, first_samp=eye.first_samp, copy="auto"
                )
                eye.add_channels([comp_array], force_update_info=True)
            del W, H, nmf
        except Exception as e:
            # Graceful fallback to SVD if sklearn is not available or NMF fails to converge
            print(f"NMF unavailable or failed ({e}); falling back to SVD components.")
            u, s, vh = np.linalg.svd(nmf_data, full_matrices=False)
            for k in range(n_comp):
                comp_ts = (s[k] * vh[k, :])[np.newaxis, :]
                comp_info = mne.create_info(
                    [f"eye_pc{k + 1}"], eye.info["sfreq"], ch_types="eog"
                )
                comp_array = mne.io.RawArray(
                    comp_ts, comp_info, first_samp=eye.first_samp, copy="auto"
                )
                eye.add_channels([comp_array], force_update_info=True)
            del u, s, vh
        del nan_array, blink_array, saccade_array, nmf_data
        print("done adding eye-tracking channels, new info:")
        print(eye.info)

        # align from events
        eye_events, eye_id = mne.events_from_annotations(eye, regexp="stim_onset")
        eye_shape = eye_events.shape[0]

        raw_events, raw_id = mne.events_from_annotations(raw, regexp="trial")
        raw_shape = raw_events.shape[0]

        raw_onset = 0 if raw_shape < eye_shape else raw_shape - eye_shape
        raw_times = (raw_events[raw_onset:, 0] / raw.info["sfreq"]) - raw.first_time

        eye_onset = 0 if eye_shape < raw_shape else eye_shape - raw_shape
        eye_times = (eye_events[eye_onset:, 0] / eye.info["sfreq"]) - eye.first_time

        # realign the raw data --------
        print("\nRealigning eye-tracking data to OPM...")
        mne.preprocessing.realign_raw(raw, eye, raw_times, eye_times, verbose=True)

        def print_info(raw, eye, idx):
            print(f"\n\n[{idx}]******** raw info *************")
            # report number of CSI
            n_csi = 0
            for ann in raw.annotations:
                if ann["description"] == "CSI":
                    n_csi += 1
            print("number of CSI: ", n_csi)
            print(raw.annotations[:10].description)
            print(raw.annotations[:10].onset)
            print(raw.annotations[-10:].description)
            print(raw.annotations[-10:].onset)
            print("raw onset:", raw.first_time, raw.first_samp)
            print("eye onset:", eye.first_time, eye.first_samp)

        # print_info(raw,eye,505)

        # reset first_samp to zero ------------------------------------
        raw_ann = raw.annotations
        raw_ann.onset -= raw.first_time
        raw = mne.io.RawArray(raw._data, raw.info, first_samp=0, copy="both")
        raw.set_annotations(raw_ann)

        eye_ann = eye.annotations
        eye_ann.onset -= eye.first_time
        eye = mne.io.RawArray(eye._data, eye.info, first_samp=0, copy="both")
        eye.set_annotations(eye_ann)

        # add eye to raw ------------------------------------
        raw.add_channels([eye], force_update_info=True)
        # print_info(raw,eye,530)

        # update coil types
        if "DIN" in raw.ch_names:
            raw.drop_channels("DIN")
        if "xpos_right" in raw.ch_names:
            mne.preprocessing.eyetracking.set_channel_types_eyetrack(
                raw,
                {
                    "xpos_right": ("eyegaze", "rad", "right", "x"),
                    "ypos_right": ("eyegaze", "rad", "right", "y"),
                    "pupil_right": ("pupil", "rad", "right"),
                },
            )
        if "xpos_left" in raw.ch_names:
            mne.preprocessing.eyetracking.set_channel_types_eyetrack(
                raw,
                {
                    "xpos_left": ("eyegaze", "rad", "left", "x"),
                    "ypos_left": ("eyegaze", "rad", "left", "y"),
                    "pupil_left": ("pupil", "rad", "left"),
                },
            )

        print(
            "\nupdated info ----------------------\n",
            raw.info,
            "\n----------------------\n",
        )

        # plot annotations ------------------------------------
        show_eye_plots = False
        if show_eye_plots:
            raw_copy = raw.copy()
            cfg.eye_annotations = ["blink", "stim_onset"]
            eye_anot = eye.annotations.copy()
            mask = np.array(
                [desc in cfg.eye_annotations for desc in eye_anot.description]
            )

            print("eye ann onset: ", eye_anot.onset[mask])

            new_eye_anot = mne.Annotations(
                onset=eye_anot.onset[mask],
                duration=eye_anot.duration[mask],
                description=eye_anot.description[mask],
            )

            raw_copy.set_annotations(raw.annotations + new_eye_anot)

            # print summary of new_eye_anot
            print(f"\nEye-tracking annotations to be added ({len(new_eye_anot)}): ")
            for desc in np.unique(new_eye_anot.description):
                count = np.sum(new_eye_anot.description == desc)
                print(f"  {desc}: {count}")
            print()

            raw_copy.crop(0, 600).plot(
                duration=10, block=True, precompute=True, scalings=dict(mag=1e-11)
            )
            del raw_copy

            # # remove annotations
            # rm_list=[]
            # for aa, (ann,) in enumerate(zip(raw.annotations)):
            #     if ann["description"] in cfg.eye_annotations:
            #         rm_list.append(aa)
            # if rm_list:
            #     raw.annotations.delete(rm_list)

            # plot gaze
            gaze_epochs = mne.make_fixed_length_epochs(raw)
            mne.viz.eyetracking.plot_gaze(gaze_epochs, sigma=10, calibration=cal)
            del gaze_epochs

        del eye

    # Write to BIDS -----------------------------------------------------------
    # set bids path
    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=cfg.session,
        task=task,
        run="01",
        root=cfg.bids_dir,
    )

    # write raw data to BIDS
    if emptyroom_path:
        mne_bids.write_raw_bids(
            raw,
            bids_path,
            allow_preload=True,
            overwrite=True,
            format="FIF",
            empty_room=emptyroom_bids_path,
        )
    else:
        mne_bids.write_raw_bids(
            raw,
            bids_path,
            allow_preload=True,
            overwrite=True,
            format="FIF",
        )

    # Write anatomical image to BIDS --------------------------------------
    if t1w_path:
        anat_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg.session,
            suffix="T1w",
            root=cfg.bids_dir,
        )

        mne_bids.write_anat(
            image=t1w_path,
            bids_path=anat_bids_path,
            overwrite=True,
            verbose=True,
        )

        print("\n-------------\nsaved t1w: ", t1w_path)

    if t2w_path:
        anat_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg.session,
            suffix="T2w",
            root=cfg.bids_dir,
        )

        mne_bids.write_anat(
            image=t2w_path,
            bids_path=anat_bids_path,
            overwrite=True,
            verbose=True,
        )

        print("saved t2w: ", t2w_path)
    print()


# %% main ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OPM data to BIDS format")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        help="Path to the Python or YAML configuration file",
        default="",
    )

    # For backward compatibility, allow positional argument as well
    parser.add_argument(
        "config_pos",
        nargs="?",
        type=str,
        default="",
        help="Path to the configuration file (positional, for backward compatibility)",
    )

    args = parser.parse_args()

    # Use the named argument if provided, otherwise use positional
    config_path = args.config_path if args.config_path else args.config_pos

    print("config path: ", config_path)
    cfg = set_bids_params(config_path)
    bids_conversion(cfg)

    print("\n\n\nDONE!\n\n\n")
